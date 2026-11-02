import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StdioServerTransport } from "@modelcontextprotocol/sdk/server/stdio.js";
import { z } from "zod";
import { Pool } from "pg";
import { config } from "dotenv";
import { createHash, randomBytes } from "crypto";

config();

const pool = new Pool({
  database: process.env.DB_NAME ?? "osm",
  user: process.env.DB_USER ?? process.env.USER,
  host: process.env.DB_HOST ?? "localhost",
  port: Number(process.env.DB_PORT ?? 5432),
});

const PHOTON_URL = process.env.PHOTON_URL ?? "http://127.0.0.1:2322";
const GRAPHHOPPER_URL = process.env.GRAPHHOPPER_URL ?? "http://127.0.0.1:8989";

const MAX_LIMIT = 50;                 // max limit, limit_per_a
const MAX_SET = 100;                  // max places per join set / make_geom references
const KNN_NETWORK_CANDIDATES = 100;   // straight-line candidates knn_search_network routes
const GH_CONCURRENCY = 20;            // GraphHopper requests knn_search_network keeps in flight
const GH_TIMEOUT_MS = 10000;          // per GraphHopper request
const STREET_GAP_M = 100;             // m; max gap chaining same-named lines into one street
const REVERSE_FAR_M = 50;             // m; reverse_geocode warns beyond this
const MIN_STRETCH_M = 20;             // m; shorter route stretches are dropped
const MERGED_FROM_SHOWN = 5;          // merged_from parts listed before "..."
const TMP_TTL = "interval '1 hour'";  // tmp entity lifetime
const COORD_DP = 7;                   // lat/lon decimals (~1 cm)
const MEASURE_DP = 3;                 // decimals of distances, areas, durations, bearings

// ---------------------------------------------------------------------------
// Types and output formatting
// ---------------------------------------------------------------------------

type EntityId = string; // source_table:osm_id or tmp:<kind>-<8 hex>
type TagFilter = Record<string, string | string[]>; // default exclude only

type Point = { lat: number; lon: number };

const UNITS = ["m", "km", "mi"] as const;
type Unit = (typeof UNITS)[number];
const PROFILES = ["car", "bike", "foot"] as const;
type Profile = (typeof PROFILES)[number];

interface LocationObj {
  entity_id: EntityId;
  name: string | null;
  address: string | null;
  lat: number;
  lon: number;
  tags: Record<string, unknown>;
  perimeter_m?: number;
  area_m2?: number;
  distance_m?: number;
  duration_min?: number;
  geodesic_distance_m?: number;
  bearing_deg?: number;
  direction?: string;
  merged_from?: EntityId[];
  merged_count?: number;
}
type Place = Pick<LocationObj, "lat" | "lon" | "name" | "address"> & Partial<LocationObj>;

const round = (v: number, dp: number) => Math.round(v * 10 ** dp) / 10 ** dp;
const measure = (v: number | null | undefined) => (v == null ? undefined : round(v, MEASURE_DP));

// Drops null, undefined and {} so only informative keys reach the LLM.
function pruneNulls<T extends Record<string, unknown>>(obj: T): T {
  return Object.fromEntries(
    Object.entries(obj).filter(([, v]) => v != null && !(typeof v === "object" && !Array.isArray(v) && !Object.keys(v as object).length))
  ) as T;
}

const textResult = (body: Record<string, unknown>) => ({ content: [{ type: "text" as const, text: JSON.stringify(body) }] });
const prunedResult = (body: Record<string, unknown>) => textResult(pruneNulls(body));
const sentences = (...parts: (string | undefined | false)[]) => parts.filter(Boolean).join(" ") || undefined;

// Extra <name>_<unit> key beside the untouched <name>_m; nothing for m.
function convertMeters(meters: number, unit: Unit | undefined, name: string): Record<string, number> {
  if (!unit || unit === "m") return {};
  return { [`${name}_${unit}`]: round(meters / (unit === "km" ? 1000 : 1609.344), 6) };
}

function compassDirection(bearingDeg: number, rose: number): string {
  const names = {
    4: ["N", "E", "S", "W"],
    8: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
    16: ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"],
  }[rose]!;
  return names[Math.round(bearingDeg / (360 / names.length)) % names.length];
}

// Straight-line measures in key order; bearing only when the place is apart from the reference.
function distanceAndBearing(distanceM: number | null | undefined, bearingDeg: number | null | undefined, units?: Unit): Partial<LocationObj> {
  const d = measure(distanceM);
  const apart = d != null && d > 0 && bearingDeg != null;
  return pruneNulls({
    geodesic_distance_m: d,
    ...(d == null ? {} : convertMeters(d, units, "geodesic_distance")),
    bearing_deg: apart ? round(bearingDeg, MEASURE_DP) % 360 : undefined,
    direction: apart ? compassDirection(bearingDeg, 8) : undefined,
  });
}

const routingMeasures = (meters: number, minutes: number, units?: Unit) =>
  ({ distance_m: round(meters, MEASURE_DP), ...convertMeters(meters, units, "distance"), duration_min: round(minutes, MEASURE_DP) });

// Extra measures go before the merge parts (location key order).
const withMeasures = ({ merged_from, merged_count, ...loc }: LocationObj, measures: Record<string, unknown>) =>
  ({ ...loc, ...measures, ...pruneNulls({ merged_from, merged_count }) });

// ---------------------------------------------------------------------------
// Input schemas and checks
// ---------------------------------------------------------------------------

// An explicit null (common from LLM clients) means omitted.
const optionalArg = <T extends z.ZodTypeAny>(schema: T) => z.preprocess((v) => v ?? undefined, schema.optional());
const orDefault = <T extends z.ZodTypeAny>(schema: T, dflt: z.input<T>) => z.preprocess((v) => v ?? undefined, schema.optional().default(dflt as any));

// Some LLM clients send an object or array argument as its JSON text; parse it, anything else is left for the schema to judge.
const fromJsonText = <T extends z.ZodTypeAny>(schema: T) =>
  z.preprocess((v) => {
    if (typeof v !== "string") return v;
    try {
      return JSON.parse(v);
    } catch {
      return v;
    }
  }, schema);

// Fresh instances per use: a shared one is advertised as a $ref, inheriting the other arg's default.
const PointSchema = z.object({ lat: z.number(), lon: z.number() }).describe("Latitude and longitude in WGS84 decimal degrees.");
const placeRef = () =>
  fromJsonText(
    z
      .union([z.object({ entity_id: z.string() }).strict(), z.object({ lat: z.number(), lon: z.number() }).strict()])
      .describe("Place reference: an `entity_id` from an earlier tool result, or `lat` and `lon` coordinates.")
  );
type PlaceRef = z.infer<ReturnType<typeof placeRef>>;
const limitArg = (dflt: number) => orDefault(z.number(), dflt).describe("Maximum results to return. Default " + dflt + ". Allowed values: 1-" + MAX_LIMIT + ".");
const profileArg = () => orDefault(z.enum(PROFILES), "car").describe("Travel mode for the road network. Default `car`. Allowed values: `car`, `bike`, `foot`.");
const unitsArg = () => orDefault(z.enum(UNITS), "m").describe("Additional distance unit; meters are always included. Default `m`. Allowed values: `m`, `km`, `mi`.");
const includeArg = () =>
  orDefault(z.string(), "").describe(
    'Place filter. key=value, key=a|b (any of), key=* (tag present), key~text (contains), key!=value; combine with and, or, not and ( ); and binds tighter than or. Quote values with spaces or special characters. * is a wildcard; case ignored. Omitted or empty = no inclusion filter.'
  );
const excludeArg = () =>
  optionalArg(z.string()).describe(
    'Same grammar as `include`; a matching place is dropped. Omitted or "" = default (drops roads, trees, fences, landuse and street furniture, except keys named in `include`); a given `exclude` replaces it; `none` keeps all.'
  );
const radiusArg = () => orDefault(z.number(), 0).describe("Distance from the reference place's shape in meters. Default 0 (inside or touching). Allowed values: 0 or greater.");

function checkPoint(p: Point, name: string): Point {
  const bad = (axis: "lat" | "lon", range: number, v: number) =>
    new Error(`${name}.${axis} must be a number in [-${range}, ${range}] (got ${v}). Pass {lat, lon} in decimal degrees${axis === "lat" && Math.abs(p.lon) <= 90 ? "; lat and lon look swapped" : ""}.`);
  if (!Number.isFinite(p.lat) || Math.abs(p.lat) > 90) throw bad("lat", 90, p.lat);
  if (!Number.isFinite(p.lon) || Math.abs(p.lon) > 180) throw bad("lon", 180, p.lon);
  return p;
}

function checkLimit(limit: number, name = "limit"): void {
  if (!Number.isInteger(limit) || limit < 1 || limit > MAX_LIMIT) throw new Error(`${name} must be an integer from 1 to ${MAX_LIMIT} (got ${limit}).`);
}

function checkNonNegative(v: number, name: string): void {
  if (!(v >= 0)) throw new Error(`${name} must be a number >= 0 (got ${v}).`);
}

const expiredHint = (id: string) => (id.trim().startsWith("tmp:") ? " A tmp:* entity expires 1 hour after creation; call the tool that made it again." : "");

// Awaits every task, then throws the first failure in argument order (deterministic error text).
async function allInOrder<T>(tasks: Promise<T>[]): Promise<T[]> {
  const settled = await Promise.allSettled(tasks);
  const failed = settled.find((s): s is PromiseRejectedResult => s.status === "rejected");
  if (failed) throw failed.reason;
  return settled.map((s) => (s as PromiseFulfilledResult<T>).value);
}

// ---------------------------------------------------------------------------
// Entity ids
// ---------------------------------------------------------------------------

const OSM_TABLES = ["point", "line", "polygon"] as const;
type OsmTable = (typeof OSM_TABLES)[number];
const TABLE_FOR: Record<OsmTable, string> = { point: "planet_osm_point", line: "planet_osm_line", polygon: "planet_osm_polygon" };

const fitsBigint = (digits: string) => BigInt(digits) <= 9223372036854775807n && BigInt(digits) >= -9223372036854775808n;
const OSM_ENTITY_RE = /^(point|line|polygon):(-?\d+)$/;
const isEntityId = (s: string) => s.startsWith("tmp:") || (OSM_ENTITY_RE.test(s) && fitsBigint(s.split(":")[1]));

function parseEntityId(entityId: EntityId): { table: OsmTable | "tmp"; id: string } {
  if (entityId.startsWith("tmp:")) return { table: "tmp", id: entityId };
  const [table, id] = entityId.split(":");
  return { table: table as OsmTable, id: BigInt(id).toString() };
}

// Canonical form (point:0123 -> point:123) of a well-formed entity_id.
const canonicalId = (entityId: EntityId) => {
  const { table, id } = parseEntityId(entityId);
  return table === "tmp" ? id : `${table}:${id}`;
};

const existsIn = async (table: OsmTable, id: bigint) => !!(await pool.query(`SELECT 1 FROM ${TABLE_FOR[table]} WHERE osm_id = $1 LIMIT 1`, [id])).rowCount;

// n -> point; w -> line, else polygon; r -> polygon, else line (relations are stored with negated ids).
async function osmIdToEntityId(osmId: string): Promise<EntityId | null> {
  const [, kind, digits] = /^([nwr])(\d+)$/.exec(osmId)!;
  if (!fitsBigint(digits)) return null;
  const n = BigInt(digits);
  if (kind === "n") return `point:${n}`;
  const id = kind === "r" ? -n : n;
  for (const table of kind === "w" ? (["line", "polygon"] as const) : (["polygon", "line"] as const)) {
    if (await existsIn(table, id)) return `${table}:${id}`;
  }
  return null;
}

// ---------------------------------------------------------------------------
// SQL fragments
// ---------------------------------------------------------------------------

// Perimeter (polygon ring or line length) and area in meters via geography, so any city works.
const measuresSql = (geom: string) =>
  `CASE WHEN ST_GeometryType(${geom}) LIKE '%Polygon%' THEN ST_Perimeter((${geom})::geography)
        WHEN ST_GeometryType(${geom}) LIKE '%LineString%' THEN ST_Length((${geom})::geography) END AS perimeter_m,
   CASE WHEN ST_GeometryType(${geom}) LIKE '%Polygon%' THEN ST_Area((${geom})::geography) END AS area_m2`;

// An OSM entity is all rows under its id (osm2pgsql splits some relations and long ways).
const entityGeomSql = (table: OsmTable, idSql: string) =>
  `(SELECT ST_CollectionHomogenize(ST_Collect(way)) FROM ${TABLE_FOR[table]} WHERE osm_id = ${idSql})`;

// One location row of an OSM entity: columns + tags as row_json (rows of one id share them), lat/lon and measures of its whole shape.
const osmEntitySql = (table: OsmTable, idSql: string) =>
  `SELECT '${table}:' || t.osm_id AS entity_id, ST_Y(ST_PointOnSurface(e.way)) AS lat, ST_X(ST_PointOnSurface(e.way)) AS lon, ${measuresSql("e.way")},
          to_jsonb(t) - 'tags' - 'way' || jsonb_build_object('tags', hstore_to_json(t.tags)) AS row_json, e.way
   FROM (SELECT * FROM ${TABLE_FOR[table]} WHERE osm_id = ${idSql} LIMIT 1) t, LATERAL (SELECT ${entityGeomSql(table, "t.osm_id")} AS way) e`;

const pointSql = (params: unknown[], p: Point) => {
  params.push(p.lon, p.lat);
  return `ST_SetSRID(ST_MakePoint($${params.length - 1}, $${params.length}), 4326)`;
};

// Planar box holding everything within meters of geom (a lon degree shrinks with latitude).
const bboxAroundSql = (geom: string, meters: string) =>
  `ST_Expand(${geom}, ${meters} / (110000 * cos(radians(least(85, greatest(abs(ST_YMin(${geom})), abs(ST_YMax(${geom}))) + 1)))))`;

// Direction of the shortest line from geometry a to b (0 = north, clockwise); agrees with ST_Distance on geography.
const shortestBearingSql = (a: string, b: string) =>
  `(SELECT degrees(ST_Azimuth(ST_StartPoint(l)::geography, ST_EndPoint(l)::geography)) FROM (SELECT ST_ShortestLine((${a})::geography, (${b})::geography)::geometry AS l) s)`;

// ---------------------------------------------------------------------------
// Tmp entities (routes, isochrones, make_geom results, merged streets): rows older than TMP_TTL are ignored and swept on insert
// ---------------------------------------------------------------------------

async function ensureTables(): Promise<void> {
  await pool.query(`
    CREATE TABLE IF NOT EXISTS tmp_geom (
      id text PRIMARY KEY,
      kind text NOT NULL,
      geom geometry(Geometry, 4326) NOT NULL,
      anchor_lat double precision NOT NULL,
      anchor_lon double precision NOT NULL,
      args jsonb,
      created_at timestamptz NOT NULL DEFAULT now()
    );
    CREATE INDEX IF NOT EXISTS tmp_geom_created_idx ON tmp_geom (created_at);
    CREATE INDEX IF NOT EXISTS planet_osm_point_osm_id_idx ON planet_osm_point (osm_id);
    CREATE INDEX IF NOT EXISTS planet_osm_line_osm_id_idx ON planet_osm_line (osm_id);
    CREATE INDEX IF NOT EXISTS planet_osm_polygon_osm_id_idx ON planet_osm_polygon (osm_id);
  `);
}

// Stores geomSql (over params) as a new tmp:<kind>-<8 hex>; anchorSql (over g) gives its lat/lon. Null when the shape is empty.
async function insertTmpGeom(kind: string, geomSql: string, params: unknown[], args: Record<string, unknown>, anchorSql = "ST_PointOnSurface(g)"): Promise<EntityId | null> {
  await pool.query(`DELETE FROM tmp_geom WHERE created_at < now() - ${TMP_TTL}`);
  const n = params.length;
  for (let attempt = 0; attempt < 3; attempt++) {
    const id = `tmp:${kind}-${randomBytes(4).toString("hex")}`;
    try {
      const { rowCount } = await pool.query(
        `INSERT INTO tmp_geom (id, kind, anchor_lat, anchor_lon, geom, args)
         SELECT $${n + 1}, $${n + 2}, ST_Y(a), ST_X(a), g, $${n + 3} FROM (SELECT g, ${anchorSql} AS a FROM (SELECT ${geomSql} AS g) s) s
         WHERE NOT ST_IsEmpty(g)`,
        [...params, id, kind, JSON.stringify(args)]
      );
      return rowCount ? id : null;
    } catch (err: any) {
      if (err.code !== "23505") throw err; // unique_violation: retry with a fresh id
    }
  }
  throw new Error("Failed to allocate a unique tmp entity id after 3 attempts");
}

// ---------------------------------------------------------------------------
// Locations
// ---------------------------------------------------------------------------

const CORE_COLS = new Set(["osm_id", "name", "tags", "way_area", "z_order"]);
const ADDR_KEYS = ["addr:housenumber", "addr:street", "addr:city", "addr:state", "addr:postcode"];
const PHOTON_ADDR_PROPS = ["housenumber", "street", "city", "state", "postcode"];
const US_STATE_CODES: Record<string, string> = {
  Alabama: "AL", Alaska: "AK", Arizona: "AZ", Arkansas: "AR", California: "CA", Colorado: "CO", Connecticut: "CT",
  Delaware: "DE", "District of Columbia": "DC", Florida: "FL", Georgia: "GA", Hawaii: "HI", Idaho: "ID", Illinois: "IL",
  Indiana: "IN", Iowa: "IA", Kansas: "KS", Kentucky: "KY", Louisiana: "LA", Maine: "ME", Maryland: "MD",
  Massachusetts: "MA", Michigan: "MI", Minnesota: "MN", Mississippi: "MS", Missouri: "MO", Montana: "MT",
  Nebraska: "NE", Nevada: "NV", "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
  "North Carolina": "NC", "North Dakota": "ND", Ohio: "OH", Oklahoma: "OK", Oregon: "OR", Pennsylvania: "PA",
  "Rhode Island": "RI", "South Carolina": "SC", "South Dakota": "SD", Tennessee: "TN", Texas: "TX", Utah: "UT",
  Vermont: "VT", Virginia: "VA", Washington: "WA", "West Virginia": "WV", Wisconsin: "WI", Wyoming: "WY",
};

// Folds fixed columns into tags, then moves the address parts out; fallback fills parts missing from the DB.
function addressAndTags(row: Record<string, any>, fallback: Record<string, string> = {}) {
  const tags: Record<string, any> = { ...(row.tags ?? {}) };
  for (const [k, v] of Object.entries(row)) if (!CORE_COLS.has(k) && v != null) tags[k] = String(v);
  const missingParts = ADDR_KEYS.some((k) => !tags[k]);
  const [num, street, city, rawState, zip] = ADDR_KEYS.map((k) => tags[k] || fallback[k]);
  const state = rawState && (US_STATE_CODES[rawState] ?? rawState); // full state name -> USPS code
  ADDR_KEYS.forEach((k) => delete tags[k]);
  const address = [[num, street].filter(Boolean).join(" "), city, [state, zip].filter(Boolean).join(" ")].filter(Boolean).join(", ");
  return { address: address || null, tags, missingParts };
}

// Address parts Photon has for this exact OSM object (matched by type and id); {} if none or on error.
const photonAddrCache = new Map<string, Promise<Record<string, string>>>();
function photonAddressParts(row: any): Promise<Record<string, string>> {
  const key = String(row.entity_id);
  if (!photonAddrCache.has(key)) {
    photonAddrCache.set(key, (async () => {
      try {
        const osmId = BigInt(row.row_json.osm_id);
        const type = key.startsWith("point:") ? "N" : osmId < 0n ? "R" : "W";
        const id = String(osmId < 0n ? -osmId : osmId);
        const features = await photon("/reverse", { lat: row.lat, lon: row.lon, limit: 5 }, AbortSignal.timeout(3000));
        const props = features?.find((f) => f.properties.osm_type === type && String(f.properties.osm_id) === id)?.properties;
        return props ? Object.fromEntries(ADDR_KEYS.map((tag, i) => [tag, props[PHOTON_ADDR_PROPS[i]]]).filter(([, v]) => v).map(([t, v]) => [t, String(v)])) : {};
      } catch {
        return {};
      }
    })());
  }
  return photonAddrCache.get(key)!;
}

// Row needs entity_id, lat, lon, row_json; optional perimeter_m, area_m2, geodesic_distance_m, bearing_deg, tmp_address, merged_from.
async function rowToLocation(row: any, units?: Unit): Promise<LocationObj> {
  let { address, tags, missingParts } = addressAndTags(row.row_json);
  if (missingParts && !String(row.entity_id).startsWith("tmp:")) ({ address, tags } = addressAndTags(row.row_json, await photonAddressParts(row)));
  const parts: EntityId[] | undefined = row.merged_from;
  return {
    entity_id: row.entity_id,
    name: row.row_json.name ?? null,
    address: address ?? row.tmp_address ?? null,
    lat: round(row.lat, COORD_DP),
    lon: round(row.lon, COORD_DP),
    tags,
    ...pruneNulls({ perimeter_m: measure(row.perimeter_m), area_m2: measure(row.area_m2) }),
    ...distanceAndBearing(row.geodesic_distance_m, row.bearing_deg, units),
    ...pruneNulls({ merged_from: parts && parts.length > MERGED_FROM_SHOWN ? [...parts.slice(0, MERGED_FROM_SHOWN), "..."] : parts, merged_count: parts?.length }),
  };
}

const toLocations = (rows: any[], units?: Unit) => Promise.all(rows.map((r) => rowToLocation(r, units)));

// The location of a well-formed entity_id, or null if it does not exist (or expired).
async function loadLocation(entityId: EntityId): Promise<LocationObj | null> {
  const { table, id } = parseEntityId(entityId);
  const sql =
    table === "tmp"
      ? `SELECT id AS entity_id, anchor_lat AS lat, anchor_lon AS lon, ${measuresSql("geom")},
                jsonb_build_object('tags', COALESCE(args->'tags', args, '{}'::jsonb), 'name', args->'name') AS row_json,
                args->'address' AS tmp_address, args->'merged_from' AS merged_from
         FROM tmp_geom WHERE id = $1 AND created_at > now() - ${TMP_TTL}`
      : `SELECT entity_id, lat, lon, perimeter_m, area_m2, row_json FROM (${osmEntitySql(table, "$1")}) e`;
  const { rows } = await pool.query(sql, [id]);
  return rows[0] ? rowToLocation(rows[0]) : null;
}

// A PlaceRef as a place; coordinates become an unnamed point.
async function resolvePlaceRef(ref: PlaceRef, name: string): Promise<Place> {
  if ("lat" in ref) return { ...checkPoint(ref, name), name: null, address: null };
  const id = ref.entity_id.trim();
  const loc = isEntityId(id) ? await loadLocation(id) : null;
  if (!loc) throw new Error(`Unrecognized entity_id "${ref.entity_id}" in ${name}. Expected an entity_id (point:123, polygon:-456), or pass {lat, lon}.${expiredHint(id)}`);
  return loc;
}

// SQL for a resolved PlaceRef's whole shape, pushing its params.
function shapeSql(ref: PlaceRef, params: unknown[]): string {
  if ("lat" in ref) return pointSql(params, ref);
  const { table, id } = parseEntityId(ref.entity_id.trim());
  params.push(id);
  return table === "tmp" ? `(SELECT geom FROM tmp_geom WHERE id = $${params.length})` : entityGeomSql(table, `$${params.length}`);
}

const isIsochrone = (p: Place) => !!p.entity_id?.startsWith("tmp:isochrone-");

// How a non-point place was read: lat/lon tools use an isochrone's origin or a point on a line or polygon; shape tools say what is measured from an isochrone's origin.
function referenceNote(label: string, p: Place, fromOrigin?: string): string | undefined {
  if (isIsochrone(p)) return `${label} is an isochrone, so ${fromOrigin ?? "its origin was used"}.`;
  if (!fromOrigin && (p.perimeter_m != null || p.area_m2 != null)) return `${label} is a line or polygon, so a point on it was used.`;
}

// ---------------------------------------------------------------------------
// Photon (geocode, reverse_geocode)
// ---------------------------------------------------------------------------

async function photon(path: string, query: Record<string, string | number>, signal?: AbortSignal): Promise<any[]> {
  const url = new URL(path, PHOTON_URL);
  for (const [k, v] of Object.entries(query)) url.searchParams.set(k, String(v));
  const res = await fetch(url, { signal });
  if (!res.ok) throw new Error(`Photon ${path === "/api" ? "geocode" : "reverse"} failed: ${res.status} ${await res.text()}`);
  return ((await res.json()) as { features: any[] }).features;
}

// Photon hits as our locations, one per OSM object (Photon emits one feature per tag); hits the DB lacks are dropped.
async function photonLocations(path: string, query: Record<string, string | number>): Promise<{ loc: LocationObj; osmKey: string }[]> {
  const out: { loc: LocationObj; osmKey: string }[] = [];
  const seen = new Set<string>();
  for (const { properties: p } of await photon(path, query)) {
    if (seen.has(`${p.osm_type}${p.osm_id}`)) continue;
    seen.add(`${p.osm_type}${p.osm_id}`);
    if (!p.osm_type || !p.osm_id) continue; // e.g. postcode features carry no OSM object
    const entityId = await osmIdToEntityId(`${String(p.osm_type).toLowerCase()}${p.osm_id}`);
    const loc = entityId && (await loadLocation(entityId));
    if (loc) out.push({ loc, osmKey: p.osm_key });
  }
  return out;
}

async function geocodeQuery(query: string, bias: Point | undefined, limit: number): Promise<LocationObj[]> {
  const hits = await photonLocations("/api", { q: query, limit, ...(bias ? { lat: bias.lat, lon: bias.lon } : {}) });
  const locations: LocationObj[] = [];
  for (const { loc, osmKey } of hits) {
    const place = osmKey === "highway" && loc.entity_id.startsWith("line:") && loc.name ? await mergedStreet(loc) : loc;
    if (!locations.some((l) => l.entity_id === place.entity_id)) locations.push(place); // several hits of one street
  }
  return locations;
}

// Distinct values: one stays plain, several become a sorted list.
function distinct(values: unknown[]): unknown {
  const d = [...new Set(values.filter((v) => v != null).flat() as string[])].sort();
  return d.length > 1 ? d : d[0] ?? null;
}

// The street holding the hit as one entity: same-named highway lines chained within STREET_GAP_M; one segment stays as is.
async function mergedStreet(hit: LocationObj): Promise<LocationObj> {
  const ids = (await pool.query(
    `WITH RECURSIVE s AS (
       SELECT osm_id, way FROM planet_osm_line WHERE osm_id = $3
       UNION
       SELECT l.osm_id, l.way FROM planet_osm_line l JOIN s
         ON l.name = $1 AND l.highway IS NOT NULL AND l.way && ${bboxAroundSql("s.way", "$2")} AND ST_DWithin(l.way::geography, s.way::geography, $2))
     SELECT DISTINCT osm_id FROM s ORDER BY osm_id`,
    [hit.name, STREET_GAP_M, parseEntityId(hit.entity_id).id]
  )).rows.map((r) => r.osm_id as string);
  if (ids.length < 2) return hit;
  // Same name and segment set = same street, so a repeat geocode reuses (and refreshes) its tmp entity
  const key = `street:${hit.name}:${createHash("sha1").update(ids.join(",")).digest("hex")}`;
  const reuse = await pool.query(`UPDATE tmp_geom SET created_at = now() WHERE kind = 'street' AND args->>'key' = $1 RETURNING id`, [key]);
  let id: EntityId | null = reuse.rows[0]?.id ?? null;
  if (!id) {
    const rows = (await pool.query(`SELECT to_jsonb(t) - 'tags' - 'way' || jsonb_build_object('tags', hstore_to_json(t.tags)) AS row_json FROM planet_osm_line t WHERE osm_id = ANY($1::bigint[])`, [ids])).rows;
    const segTags = rows.map((r) => addressAndTags(r.row_json).tags);
    const tags = Object.fromEntries([...new Set(segTags.flatMap(Object.keys))].sort().map((k) => [k, distinct(segTags.map((t) => t[k]))]));
    id = await insertTmpGeom("street", `(SELECT ST_LineMerge(ST_Collect(way)) FROM planet_osm_line WHERE osm_id = ANY($1::bigint[]))`, [ids], {
      key, name: hit.name, address: hit.address, tags, merged_from: ids.map((i) => `line:${i}`),
    });
  }
  return (await loadLocation(id!))!;
}

// Each hit with its distance and bearing from the point to its shape; nearest first, Photon's rank breaks ties.
async function reverseGeocodeQuery(point: Point, limit: number): Promise<LocationObj[]> {
  const locations: LocationObj[] = [];
  for (const { loc } of await photonLocations("/reverse", { lat: point.lat, lon: point.lon, limit })) {
    const { table, id } = parseEntityId(loc.entity_id) as { table: OsmTable; id: string };
    const params: unknown[] = [];
    const pt = pointSql(params, point);
    params.push(id);
    const g = entityGeomSql(table, `$${params.length}`);
    const { rows: [r] } = await pool.query(`SELECT ST_Distance(${pt}::geography, g::geography) AS d, ${shortestBearingSql(pt, "g")} AS az FROM (SELECT ${g} AS g) s`, params);
    locations.push({ ...loc, ...distanceAndBearing(r.d, r.az) });
  }
  return locations.sort((a, b) => a.geodesic_distance_m! - b.geodesic_distance_m!);
}

// ---------------------------------------------------------------------------
// GraphHopper (route, knn_search_network, make_isochrone)
// ---------------------------------------------------------------------------

interface GHRoute {
  distance_m: number;
  duration_min: number;
  points: [number, number][]; // [lon, lat]
  snapped: Point[]; // where origin and destination were snapped onto the network; the route runs between them
  steps: { street: string; distance_m: number }[]; // turn instructions, only when requested
}

const GH_DOWN = "GraphHopper (routing) is unreachable or failing. Retry the call shortly.";

// GraphHopper's JSON body; null when it rejects the request (4xx: no route, point not found); throws GH_DOWN on an outage.
async function graphhopper(path: string, query: [string, string][]): Promise<any | null> {
  const url = new URL(path, GRAPHHOPPER_URL);
  for (const [k, v] of query) url.searchParams.append(k, v);
  const res = await fetch(url, { signal: AbortSignal.timeout(GH_TIMEOUT_MS) }).catch(() => null);
  if (res && res.status >= 400 && res.status < 500) return null;
  const body = res?.ok ? await res.json().catch(() => null) : null;
  if (!body) throw new Error(GH_DOWN);
  return body;
}

// fn over items with at most n calls in flight, results in item order; stops starting calls after a failure.
async function mapLimit<T, R>(items: T[], n: number, fn: (item: T) => Promise<R>): Promise<R[]> {
  const out: R[] = new Array(items.length);
  let next = 0;
  const worker = async () => {
    while (next < items.length) {
      const i = next++;
      try {
        out[i] = await fn(items[i]);
      } catch (err) {
        next = items.length;
        throw err;
      }
    }
  };
  await Promise.all(Array.from({ length: Math.min(n, items.length) }, worker));
  return out;
}

async function ghRoute(origin: Point, destination: Point, profile: Profile, withSteps = false): Promise<GHRoute | null> {
  const body = await graphhopper("/route", [
    ["point", `${origin.lat},${origin.lon}`], ["point", `${destination.lat},${destination.lon}`], ["profile", profile], ["points_encoded", "false"],
    ...(withSteps ? [["instructions", "true"] as [string, string]] : []),
  ]);
  const path = body?.paths?.[0];
  if (!path) return null;
  return {
    distance_m: path.distance,
    duration_min: path.time / 60000,
    points: path.points.coordinates,
    snapped: path.snapped_waypoints.coordinates.map(([lon, lat]: number[]) => ({ lat, lon })),
    steps: (path.instructions ?? []).map((i: any) => ({ street: i.street_name ?? "", distance_m: i.distance })),
  };
}

// GeoJSON polygon reachable from origin within the budget; time is sent in whole seconds.
async function ghIsochrone(origin: Point, profile: Profile, budget: { max_duration_min?: number; max_distance_m?: number }): Promise<object | null> {
  const body = await graphhopper("/isochrone", [
    ["point", `${origin.lat},${origin.lon}`], ["profile", profile],
    budget.max_duration_min != null ? ["time_limit", String(Math.round(budget.max_duration_min * 60))] : ["distance_limit", String(budget.max_distance_m)],
  ]);
  return body?.polygons?.[0]?.geometry ?? null;
}

// Consecutive steps on one street summed into one.
function sameStreetRuns(steps: GHRoute["steps"]): GHRoute["steps"] {
  const runs: GHRoute["steps"] = [];
  for (const s of steps) {
    const last = runs[runs.length - 1];
    if (last?.street === s.street) last.distance_m += s.distance_m;
    else runs.push({ ...s });
  }
  return runs;
}

// "first (m) ... biggest middle (m) ... last (m)" over street stretches; stretches under MIN_STRETCH_M are dropped, then neighbours rejoin.
function routeSummary(steps: GHRoute["steps"]): string | undefined {
  const stretches = sameStreetRuns(sameStreetRuns(steps).filter((s) => s.distance_m >= MIN_STRETCH_M));
  if (!stretches.length) return undefined;
  const middle = stretches.slice(1, -1).sort((a, b) => b.distance_m - a.distance_m)[0];
  const last = stretches.length > 1 ? stretches[stretches.length - 1] : undefined;
  return [stretches[0], middle, last].filter((s) => s).map((s) => `${s!.street || "unnamed path"} (${Math.round(s!.distance_m)} m)`).join(" ... ");
}

// Straight-line distance in meters and initial bearing (null when the points coincide) on the WGS84 ellipsoid.
async function geodesicBetween(a: Point, b: Point): Promise<{ distance_m: number; bearing_deg: number | null }> {
  const params: unknown[] = [];
  const { rows: [r] } = await pool.query(
    `SELECT ST_Distance(a, b) AS d, degrees(ST_Azimuth(a, b)) AS az FROM (SELECT ${pointSql(params, a)}::geography AS a, ${pointSql(params, b)}::geography AS b) g`,
    params
  );
  return { distance_m: round(r.d, MEASURE_DP), bearing_deg: r.az == null ? null : round(r.az, MEASURE_DP) % 360 };
}

// ---------------------------------------------------------------------------
// Place search (range_search, knn_search, knn_search_network)
// ---------------------------------------------------------------------------

// Tag keys osm2pgsql stored as real columns, per table (read from the DB, so any style works); other keys live in the tags hstore.
const tableColumns: Record<OsmTable, Set<string>> = { point: new Set(), line: new Set(), polygon: new Set() };
async function loadTableColumns(): Promise<void> {
  const { rows } = await pool.query(
    `SELECT c.relname, a.attname FROM pg_attribute a JOIN pg_class c ON c.oid = a.attrelid
     WHERE c.oid = ANY($1::text[]::regclass[]) AND a.attnum > 0 AND NOT a.attisdropped`,
    [Object.values(TABLE_FOR)]
  );
  for (const table of OSM_TABLES) {
    tableColumns[table] = new Set(rows.filter((r) => r.relname === TABLE_FOR[table] && !["osm_id", "way", "tags", "z_order", "way_area"].includes(r.attname)).map((r) => r.attname));
  }
}

// Usual noise near a point; applies when exclude is omitted, minus the keys include names.
const DEFAULT_EXCLUDE: TagFilter = {
  highway: "*", natural: "*", barrier: "*", landuse: "*",
  amenity: ["bench", "waste_basket", "bicycle_parking", "parking_space", "parking_entrance"],
};
const DEFAULT_EXCLUDE_NOTE = 'Default exclude applied (roads, trees, fences, landuse, street furniture); pass exclude none to keep them.';

// Filter expression: tag tests (= != ~) combined with and / or / not.
// Runner-up grammar: compact_dsl (key=a,b with | & !), nearly tied on knn, range and join.
type FilterNode =
  | { op: "=" | "!=" | "~"; key: string; values: string[] }
  | { op: "and" | "or"; kids: FilterNode[] }
  | { op: "not"; kid: FilterNode };

const FILTER_FORM = 'Form: key=value, key=a|b, key=*, key~text, key!=value; join with and, or, not and ( ); quote values with spaces or special characters, e.g. name="Main St".';
const KEY_CHAR = /[A-Za-z0-9_:.\-]/;
const BARE_CHAR = /[^\s()"|]/;

// Recursive descent: or > and > not > ( ) | test. Returns null for a blank filter.
function parseFilter(text: string, label: string): FilterNode | null {
  let pos = 0;
  const fail = (expected: string): never => {
    const found = pos >= text.length ? "end of input" : `"${text[pos]}"`;
    throw new Error(`Invalid ${label} at position ${pos + 1}: expected ${expected}, found ${found}.\n  ${text}\n  ${" ".repeat(pos)}^\n${FILTER_FORM}`);
  };
  const ws = () => { while (pos < text.length && /\s/.test(text[pos])) pos++; };
  const keyword = (w: string) => {
    ws();
    const hit = new RegExp(`^${w}(?=[\\s(]|$)`, "i").test(text.slice(pos));
    if (hit) pos += w.length;
    return hit;
  };
  const piece = (): string => {
    if (text[pos] === '"') {
      let out = "";
      for (pos++; pos < text.length && text[pos] !== '"'; pos++) out += text[pos] === "\\" && pos + 1 < text.length ? text[++pos] : text[pos];
      if (text[pos] !== '"') fail('a closing "');
      pos++;
      return out;
    }
    const start = pos;
    while (pos < text.length && BARE_CHAR.test(text[pos])) pos++;
    if (pos === start) fail("a value (* = any)");
    return text.slice(start, pos);
  };
  const test = (): FilterNode => {
    const start = pos;
    while (pos < text.length && KEY_CHAR.test(text[pos])) pos++;
    if (pos === start) fail('a tag key, "not" or "("');
    const key = text.slice(start, pos);
    const op = (["!=", "=", "~"] as const).find((o) => text.startsWith(o, pos));
    if (!op) fail(`"=", "!=" or "~" after key "${key}"`);
    pos += op!.length;
    const values = [piece()];
    while (text[pos] === "|") { pos++; values.push(piece()); }
    return { op: op!, key, values };
  };
  const unary = (): FilterNode => {
    if (keyword("not")) return { op: "not", kid: unary() };
    ws();
    if (text[pos] !== "(") return test();
    pos++;
    const inner = chain("or");
    ws();
    if (text[pos] !== ")") fail('")", "and" or "or"');
    pos++;
    return inner;
  };
  const chain = (op: "and" | "or"): FilterNode => {
    const kids = [op === "or" ? chain("and") : unary()];
    while (keyword(op)) kids.push(op === "or" ? chain("and") : unary());
    return kids.length === 1 ? kids[0] : { op, kids };
  };
  ws();
  if (pos >= text.length) return null;
  const node = chain("or");
  ws();
  if (pos < text.length) fail('"and", "or" or end of input');
  return node;
}

const filterKeys = (n: FilterNode | null): Set<string> =>
  !n ? new Set() : "key" in n ? new Set([n.key]) : new Set([n.op === "not" ? n.kid : n.kids].flat().flatMap((k) => [...filterKeys(k)]));

// Canonical text of a parsed filter, echoed back so the caller can check its intent.
function renderFilter(n: FilterNode, parent?: "and" | "or" | "not"): string {
  if ("key" in n) return `${n.key}${n.op}${n.values.map((v) => (v && [...v].every((c) => BARE_CHAR.test(c)) ? v : JSON.stringify(v))).join("|")}`;
  if (n.op === "not") return `not ${renderFilter(n.kid, "not")}`;
  const text = n.kids.map((k) => renderFilter(k, n.op)).join(` ${n.op} `);
  return parent === "not" || (parent === "and" && n.op === "or") ? `(${text})` : text;
}

// An "and" over different keys is the usual cause of an empty first search.
const crossKeyAnd = (n: FilterNode | null): boolean =>
  !n || "key" in n ? false : n.op === "not" ? crossKeyAnd(n.kid) : (n.op === "and" && filterKeys(n).size > 1) || n.kids.some(crossKeyAnd);

// * is the only wildcard, so LIKE's own specials match literally.
const likePattern = (v: string) => v.replace(/[\\%_]/g, "\\$&").replace(/\*/g, "%");

// One table's SQL for a filter; each test is two-valued (a missing tag is false), so != and not are NULL-safe.
function filterSql(n: FilterNode, params: unknown[], columns: Set<string>): string {
  if (n.op === "not") return `NOT ${filterSql(n.kid, params, columns)}`;
  if ("kids" in n) return `(${n.kids.map((k) => filterSql(k, params, columns)).join(n.op === "and" ? " AND " : " OR ")})`;
  let col = `"${n.key}"`;
  if (!columns.has(n.key)) {
    params.push(n.key);
    col = `(tags -> $${params.length}::text)`;
  }
  params.push(n.values.map((v) => (n.op === "~" ? `%${likePattern(v)}%` : likePattern(v))));
  const match = `COALESCE(${col} ILIKE ANY($${params.length}::text[]), false)`;
  return n.op === "!=" ? `NOT ${match}` : match;
}

function tagWhereSql(include: FilterNode | null, exclude: FilterNode | null, params: unknown[], columns: Set<string>): string {
  return (include ? ` AND ${filterSql(include, params, columns)}` : "") + (exclude ? ` AND NOT ${filterSql(exclude, params, columns)}` : "");
}

interface SearchSpec {
  shape: string; // SQL over params: membership is within radiusM of it
  anchor?: string; // distance, order and bearing are measured from it; default shape
  params: unknown[];
  radiusM: number | null; // null = no radius, the nearest by anchor (knn)
  include: FilterNode | null;
  exclude: FilterNode | null;
  limit: number;
}

// Entities across point/line/polygon, nearest first by geodesic distance from the anchor, ties by smallest area then entity_id.
async function searchPlaces(s: SearchSpec): Promise<{ rows: any[]; totalFound: number }> {
  const params = [...s.params];
  const radius = s.radiusM ?? 0;
  // knn: planar <-> index order overfetches rows, the exact geodesic sort re-ranks
  // ponytail: distance-0 ties beyond the overfetch (places inside a large reference) are cut arbitrarily, not by area
  const overfetch = Math.max(s.limit * 10, s.limit + 200);
  const hits = OSM_TABLES.map((table) => {
    const where = tagWhereSql(s.include, s.exclude, params, tableColumns[table]);
    const cols = `'${table}' AS tbl, osm_id, ST_Distance(ref.a::geography, way::geography) AS d,
                  CASE WHEN ST_GeometryType(way) LIKE '%Polygon%' THEN ST_Area(way::geography) ELSE 0 END AS area`;
    return s.radiusM == null
      ? `SELECT ${cols} FROM (SELECT osm_id, way FROM ${TABLE_FOR[table]} t WHERE true${where} ORDER BY way <-> (SELECT a FROM ref) LIMIT ${overfetch}) t, ref`
      : `SELECT ${cols} FROM ${TABLE_FOR[table]} t, ref WHERE way && ref.box AND ${radius > 0 ? `ST_DWithin(ref.g::geography, way::geography, ${radius})` : "ST_Intersects(ref.g, way)"}${where}`;
  });
  const entity = OSM_TABLES.map((table) => `SELECT * FROM (${osmEntitySql(table, "top.osm_id")}) x WHERE top.tbl = '${table}'`).join(" UNION ALL ");
  const { rows } = await pool.query(
    `WITH ref AS MATERIALIZED (SELECT g, COALESCE(a, g) AS a, ${radius > 0 ? bboxAroundSql("g", String(radius)) : "g"} AS box
                               FROM (SELECT ${s.shape} AS g, ${s.anchor ?? "NULL::geometry"} AS a) r),
     -- one row per entity: its nearest row's distance, its rows' summed area
     top AS (SELECT tbl, osm_id, tbl || ':' || osm_id AS entity_id, min(d) AS d, sum(area) AS area, count(*) OVER () AS total_found
             FROM (${hits.join(" UNION ALL ")}) h GROUP BY tbl, osm_id
             ORDER BY min(d), sum(area), (tbl || ':' || osm_id) COLLATE "C" LIMIT ${s.limit})
     SELECT e.entity_id, e.lat, e.lon, e.perimeter_m, e.area_m2, e.row_json, top.d AS geodesic_distance_m, top.total_found,
            ${shortestBearingSql("(SELECT a FROM ref)", "e.way")} AS bearing_deg
     FROM top CROSS JOIN LATERAL (${entity}) e
     ORDER BY top.d, top.area, top.entity_id COLLATE "C"`,
    params
  );
  return { rows, totalFound: Number(rows[0]?.total_found ?? 0) };
}

// Default exclude as a filter, minus the keys include names.
const defaultExclude = (named: Set<string>): FilterNode | null => {
  const kids = Object.entries(DEFAULT_EXCLUDE).filter(([k]) => !named.has(k)).map(([key, v]): FilterNode => ({ op: "=", key, values: [v].flat() }));
  return kids.length ? { op: "or", kids } : null;
};

// Omitted or blank exclude means the default; `none` keeps everything.
const usesDefaultExclude = (exclude: string | undefined) => !exclude?.trim();

// Resolves the reference and searches around it; a tmp:isochrone is matched by its shape but measured from its origin.
// filterNote echoes the parsed filters (and hints when a cross-key "and" matched nothing).
async function searchRows(reference: PlaceRef, radiusM: number | null, include: string, exclude: string | undefined, limit: number) {
  const inc = parseFilter(include, "include");
  const exc = usesDefaultExclude(exclude) ? defaultExclude(filterKeys(inc)) : exclude!.trim().toLowerCase() === "none" ? null : parseFilter(exclude!, "exclude");
  const place = await resolvePlaceRef(reference, "reference");
  const params: unknown[] = [];
  const shape = shapeSql(reference, params);
  const anchor = isIsochrone(place) ? pointSql(params, place) : undefined;
  const { rows, totalFound } = await searchPlaces({ shape, anchor, params, radiusM, include: inc, exclude: exc, limit });
  const echo = [inc && `include ${renderFilter(inc)}`, !usesDefaultExclude(exclude) && exc && `exclude ${renderFilter(exc)}`].filter(Boolean).join("; ");
  const filterNote = sentences(
    echo && `Filter read as: ${echo}.`,
    !rows.length && crossKeyAnd(inc) && `"and" requires every condition, but include mixes keys (${[...filterKeys(inc)].join(", ")}); use "or" between alternatives.`
  );
  return { rows, totalFound, place, filterNote };
}

const searchNote = (place: Place) => referenceNote("Reference", place, "distances, bearings and order are measured from its origin");

const defaultExcludeNote = (exclude: string | undefined, count: number, limit: number) => (usesDefaultExclude(exclude) && count < limit ? DEFAULT_EXCLUDE_NOTE : undefined);

const FILTER_EXAMPLES = `Filter examples (include/exclude):
  shop=supermarket                                     only supermarkets
  leisure=park|playground                              parks or playgrounds
  amenity=cafe or shop=coffee                          alternatives under different keys
  amenity=bank|atm
  highway=bus_stop
  sport=tennis
  cuisine~burger                                       cuisine contains burger (also lists like "burger;american")
  name=Purdue*                                         name starts with Purdue
  name="Purdue University"                             quote values with spaces
  name=*                                               has a name (named places only)
  opening_hours=*                                      has opening hours
  tourism=hotel and (name~Marriott or name~Hilton) and wheelchair=yes
  include amenity=restaurant, exclude cuisine~burger   restaurants except burger places
  include amenity=parking, exclude access=private|customers   public parking`;

// ---------------------------------------------------------------------------
// Tools
// ---------------------------------------------------------------------------

const server = new McpServer({ name: "spatial-mcp-server", version: "2.0.0" });

const GEOCODE_NO_MATCH =
  "No match. Rephrase to the official, formal name a gazetteer like OpenStreetMap would store (full name, no nicknames or abbreviations), or use a street address (number, street, city).";
const GEOCODE_RANKED =
  "Ranked by relevance. If none fits the request, raise limit or rephrase. If several fit, choose the best fit for the user request based on your knowledge and context.";
const REVERSE_NONE = "No OSM place near this point. Check the coordinates (lat, lon order, inside the data area), or use range_search on this point.";
const newEntityNote = (what: string, latLon?: string) =>
  `New entity created for this ${what}; use its entity_id for further queries.${latLon ? ` Its lat/lon is ${latLon}.` : ""}`;

// lookup: an entity_id or OSM id to its location.
server.registerTool(
  "lookup",
  {
    description: "Find a place by `entity_id` or OSM ID.",
    inputSchema: {
      id: z.string().describe("Place identifier: `entity_id` (e.g. `point:123`, `polygon:-456`) or OSM ID (e.g. `n123`, `w123`, `r123`)."),
    },
  },
  async ({ id }) => {
    const raw = id.trim();
    const osmId = /^[nwr]\d+$/i.test(raw);
    const entityId = osmId ? await osmIdToEntityId(raw.toLowerCase()) : isEntityId(raw) ? raw : null;
    const location = entityId && (await loadLocation(entityId));
    if (!location) throw new Error(`Unrecognized id "${id}". Expected an entity_id (point:123, polygon:-456) or an OSM id (n123, w123, r123).${expiredHint(raw)}`);
    return prunedResult({ location, guidance: osmId ? `Use ${location.entity_id} as the entity_id in other tools.` : undefined });
  }
);

// geocode: Photon text search; named highway hits become whole merged streets.
server.registerTool(
  "geocode",
  {
    description:
      "Find places by name or address: cities, landmarks, POIs, buildings, streets, street addresses. " +
      "The returned `entity_id` can be used in other tools. For nearby searches use `range_search` or `knn_search`.",
    inputSchema: {
      query: z.string().describe("Place name or address."),
      location_bias: optionalArg(fromJsonText(PointSchema.describe(PointSchema.description + " Ranking hint only; use `ip_location` from context if present."))),
      limit: limitArg(5),
    },
  },
  async ({ query, location_bias, limit }) => {
    if (!query.trim()) throw new Error("query must be a non-empty place name or address.");
    checkLimit(limit);
    if (location_bias) checkPoint(location_bias, "location_bias");
    const locations = await geocodeQuery(query, location_bias, limit);
    const count = locations.length;
    const guidance = sentences(
      count === 0 ? GEOCODE_NO_MATCH : count > 1 || count === limit ? GEOCODE_RANKED : undefined,
      locations.some((l) => l.merged_from) && "A `tmp:*` result merges all segments of one street (see merged_from)."
    );
    return prunedResult({ count, locations, guidance });
  }
);

// reverse_geocode: Photon's places near a point, re-ranked by true distance to their shapes.
server.registerTool(
  "reverse_geocode",
  {
    description: "Find places closest to a point, nearest first. The returned `entity_id` can be used in other tools. For nearby searches use `range_search` or `knn_search`.",
    inputSchema: {
      point: fromJsonText(PointSchema),
      limit: limitArg(5),
    },
  },
  async ({ point, limit }) => {
    checkPoint(point, "point");
    checkLimit(limit);
    const locations = await reverseGeocodeQuery(point, limit);
    const count = locations.length;
    const nearest = locations[0]?.geodesic_distance_m ?? 0;
    const guidance =
      count === 0 ? REVERSE_NONE
      : nearest > REVERSE_FAR_M ? `Nearest place is ${Math.round(nearest)} m away, so the point may not be at a mapped place. Report the distance, or use range_search for a fixed radius.`
      : count === limit ? "Nearest first. Raise limit to see more."
      : undefined;
    return prunedResult({ count, locations, guidance });
  }
);

// bearing: initial geodesic bearing between two places' lat/lon.
server.registerTool(
  "bearing",
  {
    description: "Calculate the initial bearing and compass direction from `origin` to `destination`.",
    inputSchema: {
      origin: placeRef(),
      destination: placeRef(),
      rose: orDefault(z.union([z.literal(4), z.literal(8), z.literal(16)]), 8).describe("Number of compass points. Default 8. Allowed values: 4, 8, 16."),
    },
  },
  async ({ origin, destination, rose }) => {
    const from = await resolvePlaceRef(origin, "origin");
    const to = await resolvePlaceRef(destination, "destination");
    const { distance_m, bearing_deg } = await geodesicBetween(from, to);
    const guidance = sentences(bearing_deg == null && "Origin and destination coincide; no bearing.", referenceNote("Origin", from), referenceNote("Destination", to));
    return textResult({ bearing_deg, direction: bearing_deg == null ? null : compassDirection(bearing_deg, rose), geodesic_distance_m: distance_m, guidance });
  }
);

// route: GraphHopper path between two places' lat/lon, stored as a tmp:route.
server.registerTool(
  "route",
  {
    description: "Find a route from `origin` to `destination`. Returns a new `entity_id` for the path, usable in other tools.",
    inputSchema: {
      origin: placeRef(),
      destination: placeRef(),
      profile: profileArg(),
      units: unitsArg(),
    },
  },
  async ({ origin, destination, profile, units }) => {
    const from = await resolvePlaceRef(origin, "origin");
    const to = await resolvePlaceRef(destination, "destination");
    const route = await ghRoute(from, to, profile, true);
    if (!route) throw new Error(`No ${profile} route between origin and destination. Check the coordinates are inside the data area and on or near a road, or try another profile.`);
    const line = `LINESTRING(${route.points.map(([lon, lat]) => `${lon} ${lat}`).join(",")})`;
    const entity_id = await insertTmpGeom("route", "ST_GeomFromText($1, 4326)", [line], { origin, destination, profile });
    const geodesic = (await geodesicBetween(from, to)).distance_m;
    return prunedResult({
      entity_id,
      ...routingMeasures(route.distance_m, route.duration_min, units),
      geodesic_distance_m: geodesic,
      ...convertMeters(geodesic, units, "geodesic_distance"),
      summary: routeSummary(route.steps),
      guidance: sentences(newEntityNote("route"), referenceNote("Origin", from), referenceNote("Destination", to)),
    });
  }
);

const MAKE_GEOM_OPS = ["bbox", "convex_hull", "min_bounding_circle", "union", "intersection", "boundary"] as const;
const ENCLOSING_OPS = new Set(["bbox", "convex_hull", "min_bounding_circle"]);

// make_geom: one PostGIS op over the references' whole shapes, stored as a tmp entity.
server.registerTool(
  "make_geom",
  {
    description: "Create a new shape from the full shapes of the referenced places. Returns a new `entity_id` for it, usable in other tools.",
    inputSchema: {
      op: z.enum(MAKE_GEOM_OPS).describe(
        "`bbox`, `convex_hull`, `min_bounding_circle` enclose all `references`; `union` merges them; `intersection` = the area all share; `boundary` = the outline of their areas."
      ),
      references: fromJsonText(z.array(placeRef()).describe("Place references whose full shapes are used to build the result.")),
    },
  },
  async ({ op, references }) => {
    if (references.length < 1 || references.length > MAX_SET) throw new Error(`references holds ${references.length} places; give 1 to ${MAX_SET}.`);
    if (op === "intersection" && references.length < 2) throw new Error("intersection needs at least 2 references.");
    await allInOrder(references.map((r, i) => resolvePlaceRef(r, `references[${i}]`)));
    const params: unknown[] = [];
    const shapes = references.map((r) => shapeSql(r, params));
    const all = `ST_Collect(ARRAY[${shapes.join(", ")}])`;
    const geomSql = {
      bbox: `ST_Envelope(${all})`,
      convex_hull: `ST_ConvexHull(${all})`,
      min_bounding_circle: `ST_MinimumBoundingCircle(${all})`,
      union: `ST_Union(ARRAY[${shapes.join(", ")}])`,
      intersection: `ST_CollectionExtract(${shapes.reduce((a, b) => `ST_Intersection(${a}, ${b})`)})`, // its highest-dimension part
      boundary: `ST_Boundary(ST_CollectionExtract(ST_Union(ARRAY[${shapes.join(", ")}]), 3))`,
    }[op];
    const circle = op === "min_bounding_circle";
    const entity_id = await insertTmpGeom(op, geomSql, params, { op, references }, circle ? `(ST_MinimumBoundingRadius(${all})).center` : undefined);
    if (!entity_id) {
      const why = op === "intersection" ? "References do not overlap, so the intersection is empty" : "References have no area, so the boundary is empty";
      return textResult({ entity_id: null, lat: null, lon: null, guidance: `${why}; no entity was created.` });
    }
    const { lat, lon, perimeter_m, area_m2 } = (await loadLocation(entity_id))!;
    const flat = ENCLOSING_OPS.has(op) && area_m2 == null && `References span no area, so the ${op} is a ${perimeter_m == null ? "point" : "line"}.`;
    return prunedResult({ entity_id, lat, lon, perimeter_m, area_m2, guidance: sentences(newEntityNote(op, circle ? "its center" : "a point on it"), flat) });
  }
);

// make_isochrone: GraphHopper isochrone from origin's lat/lon, stored as a tmp:isochrone anchored at that point.
server.registerTool(
  "make_isochrone",
  {
    description:
      "Create the area reachable from `origin` within a travel time or distance on the road network. Returns a new `entity_id` for it, usable in other tools.",
    inputSchema: {
      origin: placeRef(),
      profile: profileArg(),
      max_duration_min: optionalArg(z.number()).describe("Travel time budget in minutes. Supply exactly one of this or `max_distance_m`. Allowed values: greater than 0."),
      max_distance_m: optionalArg(z.number()).describe("Travel distance budget in meters. Supply exactly one of this or `max_duration_min`. Allowed values: greater than 0."),
    },
  },
  async ({ origin, profile, max_duration_min, max_distance_m }) => {
    const budget = pruneNulls({ max_duration_min, max_distance_m });
    const given = Object.entries(budget);
    if (given.length !== 1) throw new Error("Pass exactly one of max_duration_min or max_distance_m.");
    const [[name, value]] = given;
    if (!((value as number) > 0)) throw new Error(`${name} must be a number > 0 (got ${value}).`);
    const from = await resolvePlaceRef(origin, "origin");
    const polygon = await ghIsochrone(from, profile, budget);
    if (!polygon) throw new Error(`No ${profile} isochrone from origin. Check the coordinates are inside the data area and on or near a road, or try another profile.`);
    const params: unknown[] = [JSON.stringify(polygon)];
    const anchor = pointSql(params, from);
    const entity_id = (await insertTmpGeom("isochrone", "ST_SetSRID(ST_GeomFromGeoJSON($1), 4326)", params, { origin, profile, [name]: value }, anchor))!;
    const { lat, lon, perimeter_m, area_m2 } = (await loadLocation(entity_id))!;
    return textResult({ entity_id, lat, lon, perimeter_m, area_m2, guidance: sentences(newEntityNote("isochrone", "its origin"), referenceNote("Origin", from)) });
  }
);

// range_search: places within radius_m of the reference shape.
server.registerTool(
  "range_search",
  {
    description: "Find places within `radius_m` of `reference`, nearest first. To relate two sets of places, pass their `entity_ids` to `spatial_join`.\n\n" + FILTER_EXAMPLES,
    inputSchema: {
      reference: placeRef(),
      radius_m: radiusArg(),
      include: includeArg(),
      exclude: excludeArg(),
      limit: limitArg(20),
      units: unitsArg(),
    },
  },
  async ({ reference, radius_m, include, exclude, limit, units }) => {
    checkNonNegative(radius_m, "radius_m");
    checkLimit(limit);
    const { rows, totalFound, place, filterNote } = await searchRows(reference, radius_m, include, exclude, limit);
    const locations = await toLocations(rows, units);
    const count = locations.length;
    const guidance = sentences(
      count === 0 ? "No matches. Increase radius_m or loosen include/exclude."
      : totalFound > count && `Showing the nearest ${count} of ${totalFound}. Raise limit, narrow include/exclude, or reduce radius_m.`,
      filterNote,
      searchNote(place),
      defaultExcludeNote(exclude, count, limit)
    );
    return prunedResult({ count, total_found: totalFound, locations, guidance });
  }
);

// knn_search: the limit nearest places to the reference shape.
server.registerTool(
  "knn_search",
  {
    description: "Find the k nearest matching places to `reference` by straight-line (geodesic) distance, nearest first. Set `limit` to k. To relate two sets of places, pass their `entity_ids` to `spatial_join`.\n\n" + FILTER_EXAMPLES,
    inputSchema: {
      reference: placeRef(),
      include: includeArg(),
      exclude: excludeArg(),
      limit: limitArg(5).describe("Number of nearest places to return (k). Default 5. Allowed values: 1-" + MAX_LIMIT + "."),
      units: unitsArg(),
    },
  },
  async ({ reference, include, exclude, limit, units }) => {
    checkLimit(limit);
    const { rows, place, filterNote } = await searchRows(reference, null, include, exclude, limit);
    const locations = await toLocations(rows, units);
    const count = locations.length;
    const guidance = sentences(
      count === 0 ? "No matches. Loosen include/exclude." : count < limit && `Only ${count} matches. Loosen include/exclude for more.`,
      filterNote,
      searchNote(place),
      defaultExcludeNote(exclude, count, limit)
    );
    return prunedResult({ count, locations, guidance });
  }
);

const byId = (a: string, b: string) => (a < b ? -1 : a > b ? 1 : 0);

// knn_search_network: the limit nearest by route among the straight-line nearest candidates.
server.registerTool(
  "knn_search_network",
  {
    description: "Find the k nearest matching places to `reference` by travel distance on the road network for the selected `profile`, nearest first. Set `limit` to k.\n\n" + FILTER_EXAMPLES,
    inputSchema: {
      reference: placeRef(),
      profile: profileArg(),
      include: includeArg(),
      exclude: excludeArg(),
      limit: limitArg(5).describe("Number of nearest places to return (k). Default 5. Allowed values: 1-" + MAX_LIMIT + "."),
      units: unitsArg(),
    },
  },
  async ({ reference, profile, include, exclude, limit, units }) => {
    checkLimit(limit);
    const place = await resolvePlaceRef(reference, "reference");
    const origin = { lat: place.lat, lon: place.lon };
    // No straight-line pruning: a snapped route can be shorter than the straight line
    const { rows, filterNote } = await searchRows(origin, null, include, exclude, KNN_NETWORK_CANDIDATES + 1);
    const candidates = rows.slice(0, KNN_NETWORK_CANDIDATES);
    const routes = await mapLimit(candidates, GH_CONCURRENCY, (r) => ghRoute(origin, r, profile));
    const routed = candidates
      .flatMap((row, i) => (routes[i] ? [{ row, route: routes[i]! }] : []))
      .sort((a, b) => a.route.distance_m - b.route.distance_m || a.route.duration_min - b.route.duration_min || byId(a.row.entity_id, b.row.entity_id));
    const best = routed.slice(0, limit);
    const last = best[best.length - 1];
    const leftOut = last ? candidates.filter((r, i) => !routes[i] && r.geodesic_distance_m < last.row.geodesic_distance_m).length : 0;
    // Cap note: a match past the candidates may still be nearer by road (its own snap distance is unknown)
    const next = rows[KNN_NETWORK_CANDIDATES];
    const capped = !!next && !!last && (best.length < limit || next.geodesic_distance_m - (await geodesicBetween(origin, last.route.snapped[0])).distance_m <= last.route.distance_m);
    const locations = await Promise.all(
      best.map(async ({ row, route }) =>
        withMeasures(await rowToLocation({ ...row, geodesic_distance_m: null, bearing_deg: null }), {
          ...routingMeasures(route.distance_m, route.duration_min, units),
          ...distanceAndBearing(row.geodesic_distance_m, row.bearing_deg, units),
        })
      )
    );
    const count = locations.length;
    const guidance = sentences(
      count === 0
        ? routes.length ? `No ${profile} route to any match. Check the reference is inside the data area and on or near a road, or try another profile.` : "No matches. Loosen include/exclude."
        : count < limit && `Only ${count} matches. Loosen include/exclude for more.`,
      leftOut > 0 && `Left out ${leftOut} ${leftOut === 1 ? "match" : "matches"} nearer by straight line than the last result: no ${profile} route.`,
      capped && `Routed only the ${KNN_NETWORK_CANDIDATES} nearest matches by straight line; a farther one may be nearer by road.`,
      filterNote,
      referenceNote("Reference", place),
      defaultExcludeNote(exclude, count, limit)
    );
    return prunedResult({ count, locations, guidance });
  }
);

const joinSet = () => fromJsonText(z.array(z.string()).describe("List of `entity_id` values from earlier tool results."));
type JoinSet = string[];

// The places of one set, loaded from the entity_ids of earlier search results.
async function resolveJoinSet(ids: JoinSet, name: string): Promise<{ locs: LocationObj[] }> {
  const unrecognized = (id: string) => new Error(`Unrecognized entity_id "${id}" in ${name}. Expected an entity_id (point:123, polygon:-456).${expiredHint(id)}`);
  const bad = ids.find((id) => !isEntityId(id.trim()));
  if (bad != null) throw unrecognized(bad);
  const canonical = [...new Set(ids.map((id) => canonicalId(id.trim())))];
  if (canonical.length > MAX_SET) throw new Error(`${name} holds ${canonical.length} places; at most ${MAX_SET}. Split the list or narrow the search.`);
  const locs = await Promise.all(canonical.map(loadLocation));
  const missing = canonical.find((_, i) => !locs[i]);
  if (missing) throw unrecognized(missing);
  return { locs: locs as LocationObj[] };
}

// One row per stored geometry of each listed entity, with its 1-based list index.
const joinSetSql = (param: string) => `
  SELECT e.eid, e.i, s.w, s.w::geography AS g FROM unnest(${param}::text[]) WITH ORDINALITY e(eid, i) CROSS JOIN LATERAL (
    ${OSM_TABLES.map((t) => `SELECT way FROM ${TABLE_FOR[t]} WHERE osm_id = CASE WHEN e.eid LIKE '${t}:%' THEN split_part(e.eid, ':', 2)::bigint END`).join(" UNION ALL ")}
    UNION ALL SELECT geom FROM tmp_geom WHERE id = e.eid) s(w)`;

// spatial_join: distances between every set_a and set_b place in one SQL pass, then pairs or set_a places per join_type.
server.registerTool(
  "spatial_join",
  {
    description: "Join two sets of places. For each `set_a` place, match `set_b` places within `max_distance_m`, keep its `limit_per_a` nearest matches, or apply both constraints. Omit both for all qualifying pairs. Matches are ordered nearest first. Build each set to cover the requested places, using `entity_id` values from `range_search`, `knn_search`, `knn_search_network` or earlier tool results. One call replaces separate searches and distance calculations for each place.",
    inputSchema: {
      set_a: joinSet(),
      set_b: joinSet(),
      join_type: orDefault(z.enum(["inner", "left", "semi", "anti"]), "inner").describe(
        "Join result: `inner` returns matching pairs; `left` also keeps unmatched `set_a` places (`location_b` null); `semi` returns `set_a` places with a match; `anti` returns those without one. Default `inner`. Allowed values: `inner`, `left`, `semi`, `anti`."
      ),
      max_distance_m: optionalArg(z.number()).describe("Maximum distance between the shapes of a `set_a` place and a `set_b` place, in meters. 0 = touching or overlapping. Can combine with `limit_per_a`. Omitted = no distance restriction. Allowed values: 0 or greater."),
      limit_per_a: optionalArg(z.number()).describe("Number of nearest `set_b` matches to keep per `set_a` place for the join. Respects `max_distance_m` if supplied. Omitted = all qualifying matches. Allowed values: 1-" + MAX_LIMIT + "."),
      limit: limitArg(20),
      units: unitsArg(),
    },
  },
  async ({ set_a, set_b, join_type, max_distance_m, limit_per_a, limit, units }) => {
    if (max_distance_m != null) checkNonNegative(max_distance_m, "max_distance_m");
    if (limit_per_a != null) checkLimit(limit_per_a, "limit_per_a");
    checkLimit(limit);
    const [A, B] = await allInOrder([resolveJoinSet(set_a, "set_a"), resolveJoinSet(set_b, "set_b")]);
    const idsB = new Set(B.locs.map((l) => l.entity_id));
    // Both sets hold the same places: keep each unordered pair once
    const selfJoin = join_type === "inner" && limit_per_a == null && A.locs.length === B.locs.length && A.locs.every((l) => idsB.has(l.entity_id));

    let matches: { ai: number; bi: number; d: number }[] = [];
    if (A.locs.length && B.locs.length) {
      const near = max_distance_m != null ? `AND a.w && ${bboxAroundSql("b.w", "$3")} AND ST_DWithin(a.g, b.g, $3)` : "";
      const { rows } = await pool.query(
        `WITH A AS (${joinSetSql("$1")}), B AS (${joinSetSql("$2")})
         SELECT a.i::int AS ai, b.i::int AS bi, min(ST_Distance(a.g, b.g)) AS d
         FROM A a JOIN B b ON a.eid <> b.eid ${selfJoin ? `AND a.eid < b.eid COLLATE "C"` : ""} ${near}
         GROUP BY a.i, b.i`,
        [A.locs.map((l) => l.entity_id), B.locs.map((l) => l.entity_id), ...(max_distance_m != null ? [max_distance_m] : [])]
      );
      const aId = (i: number) => A.locs[i - 1].entity_id;
      const bId = (i: number) => B.locs[i - 1].entity_id;
      matches = rows.sort((x, y) => x.d - y.d || byId(aId(x.ai), aId(y.ai)) || byId(bId(x.bi), bId(y.bi)));
    }
    if (limit_per_a != null) {
      const per = new Map<number, number>();
      matches = matches.filter((r) => per.set(r.ai, (per.get(r.ai) ?? 0) + 1).get(r.ai)! <= limit_per_a);
    }
    const matched = new Set(matches.map((r) => r.ai));
    const unmatched = A.locs.filter((_, i) => !matched.has(i + 1));
    const measures = (d: number) => distanceAndBearing(d, null, units);

    const pairs = join_type === "inner" || join_type === "left";
    const rows: unknown[] =
      pairs ? [
        ...matches.map((r) => ({ location_a: A.locs[r.ai - 1], location_b: B.locs[r.bi - 1], ...measures(r.d) })),
        ...(join_type === "left" ? unmatched.map((loc) => ({ location_a: loc, location_b: null })) : []),
      ]
      : join_type === "semi" ? matches.filter((r, i) => matches.findIndex((s) => s.ai === r.ai) === i).map((r) => withMeasures(A.locs[r.ai - 1], measures(r.d)))
      : unmatched;
    const shown = rows.slice(0, limit);
    const count = shown.length;
    const total = rows.length;
    const anti = join_type === "anti";
    const empty = !A.locs.length ? "set_a" : !B.locs.length ? "set_b" : null;
    const guidance = sentences(
      empty ? `${empty} is empty. Pass entity_ids returned by range_search, knn_search or knn_search_network.`
      : count === 0 ? (anti ? "Every set_a place has a match within max_distance_m." : "No matches. Increase max_distance_m.")
      : total > count && (anti ? `Showing ${count} of ${total}. Raise limit or narrow set_a.` : `Showing the nearest ${count} of ${total}. Raise limit, reduce max_distance_m, or set limit_per_a.`)
    );
    return prunedResult({ count, total_found: total, [pairs ? "pairs" : "locations"]: shown, guidance });
  }
);

// get_user_location: the device position, asked of the client via elicitation.
server.registerTool(
  "get_user_location",
  {
    description:
      "Get the user's device location. Use it when a request is relative to where the user is " +
      "(e.g. 'near me', 'from here'), not when the request names a place.",
    inputSchema: {},
  },
  async () => {
    if (!server.server.getClientCapabilities()?.elicitation) {
      throw new Error("User location unavailable: client cannot provide it. Ask the user for an address or place.");
    }
    const res = await server.server.elicitInput({
      message: "Share your current location",
      requestedSchema: {
        type: "object",
        properties: {
          lat: { type: "number", description: "Latitude in WGS84 decimal degrees." },
          lon: { type: "number", description: "Longitude in WGS84 decimal degrees." },
          accuracy_m: { type: "number", description: "Estimated position uncertainty in meters." },
        },
        required: ["lat", "lon"],
      },
    });
    if (res.action !== "accept" || !res.content) {
      throw new Error("User location unavailable: user declined. Ask the user for an address or place.");
    }
    return { content: [{ type: "text", text: JSON.stringify(res.content) }] };
  }
);

// ---------------------------------------------------------------------------
// Startup
// ---------------------------------------------------------------------------

// Warms the DB caches and both services, routing between two loaded points so any extract works.
async function prewarm(): Promise<void> {
  for (const table of Object.values(TABLE_FOR)) await pool.query(`SELECT COUNT(*) FROM ${table}`);
  const { rows } = await pool.query("SELECT ST_Y(way) AS lat, ST_X(way) AS lon FROM planet_osm_point ORDER BY osm_id LIMIT 2");
  try {
    await fetch(`${PHOTON_URL}/api?q=warmup&limit=1`);
    if (rows.length === 2) await ghRoute(rows[0], rows[1], "car");
  } catch (err) {
    process.stderr.write(`[prewarm] Photon/GraphHopper not reachable yet: ${err}\n`);
  }
}

// MCP stdio shutdown: the client closes stdin (or sends SIGTERM); close the server and DB pool, then exit.
async function shutdown(): Promise<void> {
  try {
    await server.close();
    await pool.end();
  } finally {
    process.exit(0);
  }
}

async function main(): Promise<void> {
  await ensureTables();
  await loadTableColumns();
  await prewarm();
  await server.connect(new StdioServerTransport());
  // The SDK's stdio transport does not watch for stdin closing.
  process.stdin.once("end", shutdown);
  process.once("SIGTERM", shutdown);
  process.once("SIGINT", shutdown);
}

main().catch((err) => {
  process.stderr.write(`Server startup error: ${err}\n`);
  process.exit(1);
});
