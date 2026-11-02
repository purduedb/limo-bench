"""Independent SQL oracle and response checkers shared by every layer's tests.

Read-only (except two tmp_geom test writes) and never reuses server.ts SQL: an entity is all rows stored under its id, set membership is exact
ST_DWithin/ST_Intersects on the stored geometry, join distances are geography distances between the stored shapes. Every query is
bounded by a radius around one anchor, so cost does not grow with the database.
"""
import math
import os
import re
import psycopg
import pytest

TOL = 0.02  # m; the tool rounds coordinates to 7 dp before measuring
CAP = 100
TABLES = {"point": "planet_osm_point", "line": "planet_osm_line", "polygon": "planet_osm_polygon"}
CATEGORY_KEYS = ("amenity", "shop", "tourism", "leisure")
RADIUS_M = 1500  # neighbourhood all fixtures are drawn from


def connect():
    return psycopg.connect(
        dbname=os.getenv("DB_NAME", "osm"), user=os.getenv("DB_USER") or None,
        host=os.getenv("DB_HOST", "localhost"), port=os.getenv("DB_PORT", "5432"),
        options="-c default_transaction_read_only=on", autocommit=True,
    )


def ref_sql(ref, p="r"):
    """Geometry SQL for a reference: entity_id (all its rows), tmp id, {lat, lon}, or ("envelope", lon1, lat1, lon2, lat2); params named p0, p1..."""
    if isinstance(ref, str) and ref.startswith("tmp:"):
        return f"(SELECT geom FROM tmp_geom WHERE id = %({p}0)s)", {f"{p}0": ref}
    if isinstance(ref, dict):
        return f"ST_SetSRID(ST_MakePoint(%({p}0)s, %({p}1)s), 4326)", {f"{p}0": ref["lon"], f"{p}1": ref["lat"]}
    if isinstance(ref, tuple):
        _, x1, y1, x2, y2 = ref
        return (f"ST_MakeEnvelope(least(%({p}0)s, %({p}2)s), least(%({p}1)s, %({p}3)s), greatest(%({p}0)s, %({p}2)s), "
                f"greatest(%({p}1)s, %({p}3)s), 4326)"), {f"{p}0": x1, f"{p}1": y1, f"{p}2": x2, f"{p}3": y2}
    kind, n = ref.split(":")
    return f"(SELECT ST_Collect(way) FROM {TABLES[kind]} WHERE osm_id = %({p}0)s::bigint)", {f"{p}0": n}


def tag_sql(key_param, key, columns):
    """A row's value for an OSM key: its real column if the table has one, else the hstore tag."""
    return f'COALESCE(t."{key.replace(chr(34), chr(34) * 2)}"::text, t.tags -> {key_param})' if key in columns else f"(t.tags -> {key_param})"


def value_regex(v, contains=False):
    """A filter value as a regex: * is the only wildcard, everything else literal; anchored unless contains."""
    body = re.escape(v).replace(r"\*", ".*")
    return body if contains else f"^{body}$"


# Filters are ASTs: ("tag", key, op, values) | ("and" | "or", kids) | ("not", kid). A {key: value | [values]} map is shorthand.
DEFAULT_EXCLUDE = {"highway": "*", "natural": "*", "barrier": "*", "landuse": "*",
                   "amenity": ["bench", "waste_basket", "bicycle_parking", "parking_space", "parking_entrance"]}  # from the design doc


def T(key, *vals, op="="):
    return ("tag", key, op, list(vals))


def AND(*kids):
    return ("and", list(kids))


def OR(*kids):
    return ("or", list(kids))


def NOT(kid):
    return ("not", kid)


def as_ast(f, join="and"):
    """None for a blank filter, an AST as is, a map as its tests joined by `join` (include: and, exclude: or)."""
    if not f:
        return None
    if isinstance(f, tuple):
        return f
    kids = [T(k, *(v if isinstance(v, list) else [v])) for k, v in f.items()]
    return kids[0] if len(kids) == 1 else (join, kids)


def ast_keys(n):
    if n is None:
        return set()
    return {n[1]} if n[0] == "tag" else ast_keys(n[1]) if n[0] == "not" else set().union(*map(ast_keys, n[1]))


def default_exclude(include=None):
    """The design doc's DEFAULT_EXCLUDE as an or of tests, minus the keys named in include."""
    named = ast_keys(as_ast(include))
    return as_ast({k: v for k, v in DEFAULT_EXCLUDE.items() if k not in named}, "or")


def render(f, join="and"):
    """A filter as an overpass_wizard string: and binds tighter than or, quotes only where a value needs them."""
    n = as_ast(f, join)
    if n is None:
        return ""
    if n[0] == "tag":
        q = lambda v: v if v and not re.search(r'[\s()"|\\]', v) else '"' + v.replace("\\", "\\\\").replace('"', '\\"') + '"'
        return f"{n[1]}{n[2]}{'|'.join(q(v) for v in n[3])}"
    if n[0] == "not":
        return "not " + (render(n[1]) if n[1][0] == "tag" else f"({render(n[1])})")
    return f" {n[0]} ".join(f"({render(k)})" if n[0] == "and" and k[0] == "or" else render(k) for k in n[1])


def excl(f):
    """The exclude argument that drops exactly the places f matches (any-key for a map); none when f is blank."""
    return render(f, "or") or "none"


def node_sql(n, columns, params):
    if n[0] in ("and", "or"):
        return "(" + f" {n[0].upper()} ".join(node_sql(k, columns, params) for k in n[1]) + ")"
    if n[0] == "not":
        return f"NOT {node_sql(n[1], columns, params)}"
    _, key, op, vals = n
    i = len(params)
    params[f"f{i}"], params[f"f{i + 1}"] = key, [value_regex(v, op == "~") for v in vals]
    match = f"COALESCE({tag_sql(f'%(f{i})s', key, columns)} ~* ANY(%(f{i + 1})s), false)"
    return f"NOT {match}" if op == "!=" else match


def filter_sql(include, exclude, columns):
    """Row filter for one table: its real column if it has one, else the hstore tag; values case-insensitive; a missing tag matches no test."""
    params, out = {}, ""
    for flt, neg in ((as_ast(include), False), (as_ast(exclude, "or"), True)):
        if flt:
            out += f" AND {'NOT ' if neg else ''}{node_sql(flt, columns, params)}"
    return out, params


class Oracle:
    def __init__(self):
        self.db = connect()
        self._reps = {}
        self.columns = {
            k: {r[0] for r in self.q("SELECT attname FROM pg_attribute WHERE attrelid = %s::regclass AND attnum > 0 AND NOT attisdropped", [tbl])} - {"tags", "way"}
            for k, tbl in TABLES.items()
        }

    def q(self, sql, params=None):
        with self.db.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()

    def near(self, ref, radius_m, include=None, exclude=None, origin=None):
        """[(entity_id, distance_m)] of entities within radius_m of ref's geometry (0 = intersecting), nearest first, ties by smallest area then id;
        a multi-row entity once, at its nearest row, with its summed area.

        origin ({lat, lon}): measure distance and order from it instead of from ref (an isochrone reference)."""
        rsql, params = ref_sql(ref)
        osql, oparams = ref_sql(origin, "o") if origin else (rsql, {})
        params.update(oparams)
        params["rad"] = float(radius_m)
        pred = "ST_Intersects(an.a, t.way)" if radius_m == 0 else "ST_DWithin(an.a::geography, t.way::geography, an.r)"
        # planar prefilter that is a superset of the geodesic radius (lon degrees shrink with latitude)
        box = "ST_Expand(an.a, an.r / (110000 * cos(radians(least(85, greatest(abs(ST_YMin(an.a)), abs(ST_YMax(an.a))) + 1)))))"
        parts = []
        for k, tbl in TABLES.items():
            fsql, fp = filter_sql(include, exclude, self.columns[k])
            params.update(fp)
            parts.append(f"SELECT '{k}:' || osm_id AS id, ST_Distance(an.o::geography, t.way::geography) AS d, "
                         f"CASE WHEN ST_GeometryType(t.way) LIKE '%%Polygon%%' THEN ST_Area(t.way::geography) ELSE 0 END AS ar "
                         f"FROM {tbl} t, an WHERE t.way && {box}{fsql} AND {pred}")
        sql = (f"WITH an AS MATERIALIZED (SELECT {rsql} AS a, {osql} AS o, %(rad)s AS r) "
               f"SELECT id, min(d) FROM ({' UNION ALL '.join(parts)}) u GROUP BY id ORDER BY min(d), sum(ar), id COLLATE \"C\"")
        return [(i, d) for i, d in self.q(sql, params)]

    def knn(self, ref, k, include=None, exclude=None):
        """Exact geodesic nearest first, at least k + 1 entities when available; planar order only seeds a distance bound."""
        rsql, params = ref_sql(ref)
        parts = []
        for kind, tbl in TABLES.items():
            fsql, fp = filter_sql(include, exclude, self.columns[kind])
            params.update(fp)
            parts.append(f"(SELECT '{kind}:' || osm_id, ST_Distance((SELECT a FROM an)::geography, t.way::geography) FROM {tbl} t "
                         f"WHERE true{fsql} ORDER BY t.way <-> (SELECT a FROM an) LIMIT {20 * (int(k) + 1)})")
        nearest = {}
        for i, d in self.q(f"WITH an AS MATERIALIZED (SELECT {rsql} AS a) {' UNION ALL '.join(parts)}", params):
            nearest[i] = min(d, nearest.get(i, d))
        ds = sorted(nearest.values())
        # any k + 1 entities bound the distance of the true (k + 1)-th nearest
        return self.near(ref, ds[k] + 1 if len(ds) > k else 4e7, include, exclude)

    def reps(self, ids):
        """entity_id -> (lon, lat) of the ST_PointOnSurface of all its rows."""
        need = [i for i in set(ids) if i not in self._reps]
        for kind, tbl in TABLES.items():
            ns = [i.split(":")[1] for i in need if i.startswith(kind + ":")]
            if ns:
                for i, x, y in self.q(f"SELECT %s || osm_id, ST_X(p), ST_Y(p) FROM (SELECT osm_id, ST_PointOnSurface(ST_Collect(way)) AS p FROM {tbl} "
                                      "WHERE osm_id = ANY(%s::bigint[]) GROUP BY osm_id) s", [kind + ":", ns]):
                    self._reps[i] = (x, y)
        return {i: self._reps[i] for i in ids}

    def dist(self, A, B):
        """{(a, b): shortest geodesic metres between the two stored shapes (0 = touching), the nearest rows of a multi-row id}."""
        if not A or not B:
            return {}
        geoms = " UNION ALL ".join(f"SELECT i.id, t.way FROM ids i JOIN {tbl} t ON i.id LIKE '{k}:%%' AND t.osm_id = split_part(i.id, ':', 2)::bigint"
                                   for k, tbl in TABLES.items())
        rows = self.q(
            f"WITH ids AS (SELECT DISTINCT unnest(%(all)s::text[]) AS id), g AS ({geoms}) "
            "SELECT a.id, b.id, min(ST_Distance(a.way::geography, b.way::geography)) FROM g a JOIN g b ON a.id = ANY(%(a)s) AND b.id = ANY(%(b)s) GROUP BY a.id, b.id",
            {"all": list(A) + list(B), "a": list(A), "b": list(B)},
        )
        return {(a, b): d for a, b, d in rows}

    def join(self, A, B, join_type="inner", max_distance_m=None, k=None):
        """Expected spatial_join result over deduped id lists A, B (pairs allowed within TOL at k ties)."""
        A, B = list(dict.fromkeys(A)), list(dict.fromkeys(B))
        D = self.dist(A, B)
        self_join = join_type == "inner" and k is None and set(A) == set(B)
        key = (lambda a, b: tuple(sorted((a, b)))) if self_join else (lambda a, b: (a, b))
        allowed, required, per_a, matched, cands = {}, set(), {}, [], {}
        for a in A:
            c = sorted((D[a, b], b) for b in B if b != a and (max_distance_m is None or D[a, b] <= max_distance_m))
            cands[a] = c
            if k is not None and c:
                kth = c[min(k, len(c)) - 1][0]
                allowed.update({key(a, b): x for x, b in c if x <= kth + TOL})
                required.update(key(a, b) for x, b in c[:k] if x < kth - TOL)
                c = c[:k]
            else:
                allowed.update({key(a, b): x for x, b in c})
                required.update(key(a, b) for _, b in c)
            per_a[a] = len(c)
            matched += [x for x, b in c if not self_join or key(a, b) == (a, b)]
        return {
            "A": A, "B": B, "join_type": join_type, "k": k, "key": key, "self_join": self_join,
            "allowed": allowed, "required": required, "per_a": per_a, "matched": sorted(matched),
            "unmatched": [a for a in A if not cands[a]],
            "semi": sorted((c[0][0], a) for a, c in cands.items() if c),
        }

    def measures(self, geom_sql, params):
        """(area_m2, perimeter_m) of a geometry expression; None where the geometry has no such measure."""
        return self.q(
            f"SELECT CASE WHEN ST_GeometryType(g) LIKE '%%Polygon%%' THEN ST_Area(g::geography) END, "
            f"CASE WHEN ST_GeometryType(g) LIKE '%%Polygon%%' THEN ST_Perimeter(g::geography) "
            f"WHEN ST_GeometryType(g) LIKE '%%LineString%%' THEN ST_Length(g::geography) END FROM (SELECT {geom_sql} AS g) u", params)[0]

    def entity_measures(self, ref):
        """(area_m2, perimeter_m) of a stored or tmp entity's geometry."""
        return self.measures(*ref_sql(ref))

    def _derived_sql(self, op, refs):
        parts, params = [], {}
        for n, r in enumerate(refs):
            sql, p = ref_sql(r, f"g{n}_")
            parts.append(sql)
            params.update(p)
        arr = f"ARRAY[{', '.join(parts)}]"
        inter = parts[0]
        for part in parts[1:]:
            inter = f"ST_Intersection({inter}, {part})"
        geom = {
            "bbox": f"ST_Envelope(ST_Collect({arr}))", "convex_hull": f"ST_ConvexHull(ST_Collect({arr}))",
            "min_bounding_circle": f"ST_MinimumBoundingCircle(ST_Collect({arr}))", "union": f"ST_Union({arr})",
            "intersection": f"ST_CollectionExtract({inter})", "boundary": f"ST_Boundary(ST_CollectionExtract(ST_Union({arr}), 3))",
        }[op]
        return geom, params, arr

    def derived(self, op, refs):
        """(area_m2, perimeter_m) of a make_geom result computed here."""
        geom, params, _ = self._derived_sql(op, refs)
        return self.measures(geom, params)

    def circle_center(self, refs):
        """{lat, lon} of the center of the smallest circle enclosing refs."""
        _, params, arr = self._derived_sql("min_bounding_circle", refs)
        x, y = self.q(f"SELECT ST_X(c), ST_Y(c) FROM (SELECT (ST_MinimumBoundingRadius(ST_Collect({arr}))).center AS c) s", params)[0]
        return {"lat": y, "lon": x}

    def distance(self, p, q):
        """Geodesic metres between two {lat, lon} points."""
        return self.q("SELECT ST_Distance(ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography)",
                      [p["lon"], p["lat"], q["lon"], q["lat"]])[0][0]

    def azimuth(self, p, q):
        """Geodesic initial bearing in degrees (0 = north, clockwise) from {lat, lon} p to q."""
        return self.q("SELECT degrees(ST_Azimuth(ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography))",
                      [p["lon"], p["lat"], q["lon"], q["lat"]])[0][0]

    def bearing_to_entity(self, ref, entity_id):
        """Geodesic azimuth from ref's geometry (entity, tmp id or {lat, lon}) to an entity, between their mutually closest points."""
        (a, pa), (b, pb) = ref_sql(ref, "o"), ref_sql(entity_id, "i")
        return self.q(f"SELECT degrees(ST_Azimuth(ST_ClosestPoint(g, h), ST_ClosestPoint(h, g))) FROM (SELECT ({a})::geography g, ({b})::geography h) s",
                      {**pa, **pb})[0][0]

    def distance_to_entity(self, p, entity_id):
        """Geodesic metres from a {lat, lon} point to an entity's geometry (0 when inside)."""
        sql, params = ref_sql(entity_id)
        return self.q(f"SELECT ST_Distance(ST_SetSRID(ST_MakePoint(%(x)s, %(y)s), 4326)::geography, ({sql})::geography)",
                      {**params, "x": p["lon"], "y": p["lat"]})[0][0]

    def age_tmp(self, tmp_id, minutes):
        """Move a tmp entity's creation time back (a write; its own connection) to simulate expiry."""
        with psycopg.connect(dbname=os.getenv("DB_NAME", "osm"), user=os.getenv("DB_USER") or None, host=os.getenv("DB_HOST", "localhost"),
                             port=os.getenv("DB_PORT", "5432"), autocommit=True) as db:
            db.execute("UPDATE tmp_geom SET created_at = created_at - make_interval(mins => %s) WHERE id = %s", [minutes, tmp_id])

    def clear_street_cache(self):
        """Drop merged-street tmp entities (a write; its own connection) so a geocode builds them with the server under test."""
        with psycopg.connect(dbname=os.getenv("DB_NAME", "osm"), user=os.getenv("DB_USER") or None, host=os.getenv("DB_HOST", "localhost"),
                             port=os.getenv("DB_PORT", "5432"), autocommit=True) as db:
            db.execute("DELETE FROM tmp_geom WHERE kind = 'street'")

    def street_components(self, name, gap_m=100.0):
        """[(ids, total_length_m, tag_values)] of the same-named highway lines, chained within gap_m (pairwise ST_DWithin, union-find)."""
        rows = self.q(
            "SELECT osm_id, ST_Length(way::geography), jsonb_strip_nulls(to_jsonb(t) - 'tags' - 'way' - 'osm_id' - 'name' - 'way_area' - 'z_order' "
            "|| hstore_to_jsonb(COALESCE(tags, ''::hstore))) FROM planet_osm_line t WHERE name = %s AND highway IS NOT NULL ORDER BY osm_id", [name])
        parent = {r[0]: r[0] for r in rows}
        find = lambda i: i if parent[i] == i else find(parent[i])
        for a, b in self.q(
            "SELECT a.osm_id, b.osm_id FROM planet_osm_line a JOIN planet_osm_line b ON a.osm_id < b.osm_id AND b.name = a.name AND b.highway IS NOT NULL "
            "AND ST_DWithin(a.way::geography, b.way::geography, %s) WHERE a.name = %s AND a.highway IS NOT NULL", [gap_m, name]):
            parent[find(a)] = find(b)
        groups = {}
        for i, length, tags in rows:
            g = groups.setdefault(find(i), {"ids": [], "length": 0.0, "tags": {}})
            g["ids"].append(i)
            g["length"] += length
            for k, v in tags.items():
                if not k.startswith("addr:"):
                    g["tags"].setdefault(k, set()).add(str(v))
        return [(g["ids"], g["length"], {k: sorted(v) for k, v in g["tags"].items()}) for g in groups.values()]

    def route_facts(self, tmp_id, origin, dest):
        """(length_m, metres from line start to origin, from line end to dest, from anchor to line, from anchor to ST_PointOnSurface of the line) of a tmp route."""
        pt = "ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography"
        return self.q(
            f"SELECT ST_Length(geom::geography), ST_Distance(ST_StartPoint(geom)::geography, {pt}), ST_Distance(ST_EndPoint(geom)::geography, {pt}), "
            f"ST_Distance(geom::geography, ST_SetSRID(ST_MakePoint(anchor_lon, anchor_lat), 4326)::geography), "
            f"ST_Distance(ST_PointOnSurface(geom)::geography, ST_SetSRID(ST_MakePoint(anchor_lon, anchor_lat), 4326)::geography) FROM tmp_geom WHERE id = %s",
            [origin["lon"], origin["lat"], dest["lon"], dest["lat"], tmp_id])[0]

    def covers(self, outer, inner):
        """Whether geometry ref outer covers geometry ref inner (entity, tmp id or {lat, lon})."""
        (a, pa), (b, pb) = ref_sql(outer, "o"), ref_sql(inner, "i")
        return self.q(f"SELECT ST_Covers({a}, {b})", {**pa, **pb})[0][0]

    def covers_route_end(self, outer, route_id):
        """Whether geometry ref outer covers the end point of a tmp route."""
        sql, p = ref_sql(outer, "o")
        return self.q(f"SELECT ST_Covers({sql}, ST_EndPoint((SELECT geom FROM tmp_geom WHERE id = %(rid)s)))", {**p, "rid": route_id})[0][0]

    def within_radius(self, center, radius_m, ref):
        """Whether the geometry ref lies entirely inside the geodesic disc of radius_m around a {lat, lon}."""
        sql, p = ref_sql(ref, "i")
        return self.q(f"SELECT ST_Covers(ST_Buffer(ST_SetSRID(ST_MakePoint(%(cx)s, %(cy)s), 4326)::geography, %(rad)s)::geometry, {sql})",
                      {**p, "cx": center["lon"], "cy": center["lat"], "rad": float(radius_m)})[0][0]

    def gap_threshold(self, dists, target, gap=0.05):
        """A threshold near target with no distance within gap of it, so rounding cannot flip a pair."""
        ds = sorted(dists)
        for step in range(400):
            t = target + (1 if step % 2 else -1) * (step // 2) * 0.25
            if t > 0 and not any(abs(d - t) < gap for d in ds):
                return t
        raise AssertionError(f"no unambiguous threshold near {target}")


def check_location(loc, eid, reps, errs, tag, with_distance=False):
    """Location entity_id and lat/lon against the oracle's representative point; pair/semi locations carry no distance."""
    if not loc or loc.get("entity_id") != eid:
        errs.append(f"{tag}: entity_id {loc and loc.get('entity_id')} != {eid}")
        return
    if not with_distance and "geodesic_distance_m" in loc:
        errs.append(f"{tag}: location leaks geodesic_distance_m")
    x, y = reps[eid]
    if abs(loc["lat"] - y) > 2e-7 or abs(loc["lon"] - x) > 2e-7:
        errs.append(f"{tag}: lat/lon of {eid} is not its representative point")


def check_join(oracle, res, orc, limit):
    """All errors of a spatial_join response against an Oracle.join expectation (empty list = correct)."""
    errs, jt = [], orc["join_type"]
    shape = "locations" if jt in ("semi", "anti") else "pairs"
    guidance, rows = res.get("guidance"), res.get(shape, [])
    total = res.get("total_found")
    anti = jt == "anti"
    empty = "set_a" if not orc["A"] else "set_b" if not orc["B"] else None
    want = (f"{empty} is empty. Pass entity_ids returned by range_search, knn_search or knn_search_network." if empty
            else ("Every set_a place has a match within max_distance_m." if anti else "No matches. Increase max_distance_m.") if not rows
            else (f"Showing {len(rows)} of {total}. Raise limit or narrow set_a." if anti
                  else f"Showing the nearest {len(rows)} of {total}. Raise limit, reduce max_distance_m, or set limit_per_a.") if total > len(rows) else None)
    if guidance != want:
        errs.append(f"guidance {guidance!r} != {want!r}")
    if set(res) != {"count", "total_found", shape} | ({"guidance"} if guidance else set()):
        errs.append(f"top-level keys {sorted(res)}")
    if res.get("count") != len(rows):
        errs.append(f"count {res.get('count')} != {len(rows)} rows")
    reps = oracle.reps(orc["A"] + [i for pair in orc["allowed"] for i in pair])
    if shape == "pairs":
        n_match = len(orc["matched"])
        total = n_match + (len(orc["unmatched"]) if jt == "left" else 0)
        if res.get("total_found") != total:
            errs.append(f"total_found {res.get('total_found')} != {total}")
        if len(rows) != min(limit, total):
            errs.append(f"{len(rows)} rows != min(limit, total) {min(limit, total)}")
        M = [p for p in rows if p.get("location_b")]
        U = [p for p in rows if not p.get("location_b")]
        if rows[:len(M)] != M:
            errs.append("an unmatched row precedes a matched row")
        if len(M) != min(limit, n_match):
            errs.append(f"{len(M)} matched rows != {min(limit, n_match)}")
        if U and jt != "left":
            errs.append("null location_b outside a left join")
        seen, prev = set(), -1.0
        for i, p in enumerate(M):
            a, b, d = p["location_a"]["entity_id"], p["location_b"]["entity_id"], p["geodesic_distance_m"]
            kk = orc["key"](a, b)
            if kk in seen:
                errs.append(f"duplicate pair {kk}")
            if orc["self_join"] and (a, b) != kk:
                errs.append(f"self-join pair {a} -> {b} does not put the smaller entity_id first")
            seen.add(kk)
            if kk not in orc["allowed"]:
                errs.append(f"pair {a} -> {b} not expected")
                continue
            if abs(d - orc["allowed"][kk]) > TOL or i >= n_match or abs(d - orc["matched"][i]) > TOL:
                errs.append(f"distance {a} -> {b} {d} vs oracle {orc['allowed'][kk]:.3f} at rank {i}")
            if d < prev:
                errs.append("pairs not sorted by distance")
            elif i and d == prev and (a, b) < tuple(M[i - 1][k]["entity_id"] for k in ("location_a", "location_b")):
                errs.append(f"tied pairs not ordered by location_a, then location_b entity_id at rank {i}")
            prev = d
            check_location(p["location_a"], a, reps, errs, "location_a")
            check_location(p["location_b"], b, reps, errs, "location_b")
        last = M[-1]["geodesic_distance_m"] if M else -1
        missing = [kk for kk in orc["required"] if kk not in seen and (len(M) == n_match or orc["allowed"][kk] < last - TOL)]
        if missing:
            errs.append(f"{len(missing)} expected pairs missing, e.g. {missing[0]}")
        if orc["k"] is not None and len(M) == n_match:
            per = {}
            for p in M:
                per[p["location_a"]["entity_id"]] = per.get(p["location_a"]["entity_id"], 0) + 1
            if any(per.get(a, 0) != n for a, n in orc["per_a"].items()):
                errs.append("matches per set_a entity differ from min(k, candidates)")
        for p in U:
            a = p["location_a"]["entity_id"]
            if a not in orc["unmatched"] or p.get("geodesic_distance_m") is not None:
                errs.append(f"bad unmatched row for {a}")
            check_location(p["location_a"], a, reps, errs, "location_a")
        if [p["location_a"]["entity_id"] for p in U] != [a for a in orc["unmatched"]][:len(U)]:
            errs.append("unmatched rows are not the set_a places without a match, in set_a order")
    else:
        exp = {a: d for d, a in orc["semi"]} if jt == "semi" else {a: None for a in orc["unmatched"]}
        if res.get("total_found") != len(exp):
            errs.append(f"total_found {res.get('total_found')} != {len(exp)}")
        if len(rows) != min(limit, len(exp)):
            errs.append(f"{len(rows)} locations != {min(limit, len(exp))}")
        ids = [r["entity_id"] for r in rows]
        if len(set(ids)) != len(ids) or not set(ids) <= set(exp):
            errs.append(f"locations {ids[:5]} are not distinct members of the expected set")
        if jt == "anti" and ids != orc["unmatched"][:len(ids)]:
            errs.append("anti locations are not in set_a order")
        prev = -1.0
        for r in rows:
            if r["entity_id"] in exp:
                check_location(r, r["entity_id"], reps, errs, "location", with_distance=jt == "semi")
            if jt == "anti":
                if "geodesic_distance_m" in r:
                    errs.append("anti row carries geodesic_distance_m")
            elif r["entity_id"] in exp:
                nm = r.get("geodesic_distance_m")
                if nm is None or abs(nm - exp[r["entity_id"]]) > TOL or nm < prev:
                    errs.append(f"geodesic_distance_m {nm} of {r['entity_id']} vs oracle nearest match {exp[r['entity_id']]:.3f} (or unsorted)")
                prev = nm if nm is not None else prev
        if jt == "semi" and rows:
            last = rows[-1].get("geodesic_distance_m") or 0
            if any(d < last - TOL and a not in ids for a, d in exp.items()):
                errs.append("a nearer semi match is missing from the returned prefix")
    return errs


def assert_join(oracle, res, orc, limit):
    """A spatial_join response equals an Oracle.join expectation."""
    errs = check_join(oracle, res, orc, limit)
    assert not errs, "\n".join(errs[:6])


def check_search(oracle, res, rows, n, total=None, guided=False):
    """Errors of a range_search (total given) or knn_search response against oracle rows [(id, d)]: the n nearest, ties at the cut free."""
    errs, locs = [], res.get("locations", [])
    keys = {"count", "locations"} | ({"total_found"} if total is not None else set()) | ({"guidance"} if guided else set())
    if set(res) != keys:
        errs.append(f"top-level keys {sorted(res)} != {sorted(keys)}")
    ids = [x["entity_id"] for x in locs]
    if not res.get("count") == len(ids) == len(set(ids)) == n:
        errs.append(f"count {res.get('count')}, {len(ids)} locations ({len(set(ids))} distinct), expected {n}")
    if total is not None and res.get("total_found") != total:
        errs.append(f"total_found {res.get('total_found')} vs total {total}")
    want, kth = dict(rows), (rows[n - 1][1] if n else 0)
    extra = [i for i in ids if want.get(i, math.inf) > kth + TOL]
    missed = [i for i, d in rows if d < kth - TOL and i not in ids]
    if extra or missed:
        errs.append(f"not the {n} nearest: extra {extra[:3]}, missing {missed[:3]}")
    reps = oracle.reps(ids)
    for x in locs:
        d, eid = x.get("geodesic_distance_m"), x["entity_id"]
        if eid in want and (d is None or abs(d - want[eid]) > TOL):
            errs.append(f"distance of {eid} {d} vs oracle {want[eid]}")
        check_location(x, eid, reps, errs, "location", with_distance=True)
    ds = [x.get("geodesic_distance_m", 0) for x in locs]
    if ds != sorted(ds):
        errs.append("locations not sorted by distance")
    touching = [x["entity_id"] for x in locs if x.get("geodesic_distance_m") == 0]
    if touching != [i for i, d in rows if d == 0][:len(touching)]:
        errs.append("places at distance 0 not ordered by smallest area then id")
    return errs


def filter_guided(req, count, limit):
    """Whether the request's include/exclude put a note in guidance: the filter echo (include given, or exclude given and not none)
    or the default-exclude note (exclude omitted, null or blank, fewer results than limit). req = the sent arguments; None = no filters, exclude none."""
    req = {"exclude": "none"} if req is None else req
    inc, exc = (req.get("include") or "").strip(), (req.get("exclude") or "").strip()
    return bool(inc) or exc.lower() not in ("", "none") or (exc == "" and count < limit)


def assert_range(oracle, res, rows, limit=20, req=None, noted=False):
    """A range_search response holds the nearest `limit` of the oracle's rows with the full total reported (req: the sent arguments, see filter_guided; noted: a reference note)."""
    n = min(limit, len(rows))
    errs = check_search(oracle, res, rows, n, total=len(rows), guided=noted or n == 0 or len(rows) > n or filter_guided(req, n, limit))
    assert not errs, "\n".join(errs[:6])


def assert_knn(oracle, res, ref_, k, include=None, exclude=None, req=None, noted=False):
    """A knn_search response holds the oracle's exact k nearest (fewer when fewer match); include/exclude are the oracle's filters (exclude None = none),
    req the sent arguments (see filter_guided); noted: a reference note."""
    rows = oracle.knn(ref_, k, include, exclude)
    errs = check_search(oracle, res, rows, min(k, len(rows)), guided=noted or len(rows) < k or filter_guided(req, min(k, len(rows)), k))
    assert not errs, "\n".join(errs[:6])


def assert_nearest(got, rows, n):
    """got is the n nearest of rows [(id, d)]; ties within TOL at the cut may go either way."""
    n = min(n, len(rows))
    kth = rows[n - 1][1] if n else 0
    assert len(got) == n == len(set(got)), f"{len(got)} ids ({len(set(got))} distinct), expected {n}"
    d = dict(rows)
    extra = [i for i in got if d.get(i, math.inf) > kth + TOL]
    missed = [i for i, x in rows if x < kth - TOL and i not in got]
    assert not extra and not missed, f"not the {n} nearest: extra {extra[:3]}, missing {missed[:3]}"


def mid_gap_radius(rows, n):
    """(radius, m): a radius holding exactly the m <= n nearest of rows, placed mid-gap so rounding cannot flip one."""
    for m in range(min(n, len(rows) - 1), 0, -1):
        lo, hi = rows[m - 1][1], rows[m][1]
        if hi - lo > max(1.0, 1e-3 * hi):
            return (lo + hi) / 2, m
    pytest.skip("no radius with a clear gap near the anchor")


def range_case(oracle, ref, include=None, exclude=None, target_m=RADIUS_M, cap=CAP):
    """(range_search args, expected ids) with radius_m mid-gap near target_m, shrunk until under cap entities; include/exclude (filters, exclude None = none) as strings."""
    while True:
        rows = oracle.near(ref, target_m * 1.2, include, exclude)
        n = sum(d <= target_m for _, d in rows)
        if n < cap or target_m < 10:
            break
        target_m /= 2
    if not 0 < n < cap:
        pytest.skip(f"{n} entities match near the anchor, need 1..{cap - 1}")
    r, m = mid_gap_radius(rows, n)
    return {"reference": ref, "radius_m": r, "include": render(include), "exclude": excl(exclude)}, [i for i, _ in rows[:m]]
