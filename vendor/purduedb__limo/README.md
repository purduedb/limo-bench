# limo - spatial MCP server

A Model Context Protocol (MCP) server exposing location/spatial tools over OpenStreetMap data.
Geocoding is self-hosted [Photon](https://photon.komoot.io/); routing/isochrones are self-hosted
[GraphHopper](https://www.graphhopper.com/); everything else (range/KNN search, derived
geometry, spatial joins) queries Postgres/PostGIS directly against `planet_osm_*` tables
loaded from an OSM `.pbf` extract via `osm2pgsql`.

Designed to be portable to any city's `.pbf` from [Geofabrik](https://download.geofabrik.de/) -
nothing in `server.ts` is hardcoded to West Lafayette, though `data/west_lafayette.pbf` (see
`data/README.md` for provenance/license) is the sample used by `services/*/Makefile` and
`tests/` by default.

This package is meant to work standalone - clone it on its own and follow the steps below.

## Requirements

- Node.js (v18+) and `npm`
- PostgreSQL with the `postgis`, `hstore` extensions available
- `osm2pgsql` (e.g. `brew install osm2pgsql` on macOS)
- Java 17+ (for Photon/GraphHopper) and Docker or Apptainer (for Photon's one-time Nominatim import step only - see `services/photon/Makefile`)
- Python 3 with `pytest`, `mcp` and `psycopg` installed, only needed to run `tests/`

## Setup

1. Install server dependencies:
   ```bash
   npm ci
   ```

2. Set everything up: database (`planet_osm_point/line/polygon` via `osm2pgsql`), plus
   Photon (geocoding) and GraphHopper (routing/isochrones) built and started. Run once per
   machine, and again (at least `init_db load_db services_build`) whenever you reimport
   different OSM data:
   ```bash
   make            # init_db + load_db + services_build + services_up
   ```
   Override `PBF_FILE`/`DB_NAME`/`DB_USER`/`DB_HOST`/`DB_PORT` as needed, e.g.
   `make PBF_FILE=path/to/other-city.pbf DB_NAME=other_city`. Thereafter, `make services_up` /
   `make services_down` (or `photon_up`/`graphhopper_up` individually) starts/stops just the
   services - the database doesn't need restarting. `make services_clean` wipes both services'
   build output (jar, index, graph-cache) for a from-scratch rebuild.

3. Prepare `.env` (`server.ts` loads it automatically via `dotenv`; also required to run
   `tests/` - see `tests/README.md`):
   ```bash
   sed "s/your_username/$USER/" .env.example > .env
   ```

4. (Optional) Run the server directly (mainly for manual testing):
   ```bash
   npx tsx server.ts
   ```

5. (Optional) Run the test suite:
   ```bash
   python -m pytest tests/ -v -s
   ```

## MCP client configuration

```json
{
  "mcpServers": {
    "limo": {
      "command": "npx",
      "args": ["tsx", "server.ts"],
      "env": {
        "DB_NAME": "osm",
        "DB_USER": "your_username",
        "DB_HOST": "localhost",
        "DB_PORT": "5432",
        "PHOTON_URL": "http://127.0.0.1:2322",
        "GRAPHHOPPER_URL": "http://127.0.0.1:8989"
      }
    }
  }
}
```

## Tools

Tool schemas are advertised by the server (`server.ts`). A place argument is `{entity_id}` (`source_table:osm_id`,
e.g. `polygon:74144839`, or `tmp:route-a1b2c3d4` for a shape another tool created, kept 1 hour) or `{lat, lon}`,
never free text (use `geocode` first). `include`/`exclude` are one expression over native OSM tags, e.g. `amenity=restaurant`
(`key=a|b`, `key=*`, `key~text`, `key!=value`, joined by `and`/`or`/`not` and parentheses).

| Tool | Description |
|---|---|
| `geocode` | Free-text search (name, address, landmark, street) -> ranked candidates with `entity_id`; optional `location_bias` `{lat, lon}` only reorders. |
| `reverse_geocode` | `point` `{lat, lon}` -> closest places, nearest first, with `geodesic_distance_m`, `bearing_deg`, `direction`. |
| `lookup` | `entity_id` or OSM id (n/w/r) -> the place. |
| `bearing` | Compass bearing/direction from one place to another. |
| `route` | Road-network route between two places (GraphHopper) - returns a new `entity_id` for the path. |
| `make_geom` | A shape from the full geometry of places: bbox, convex_hull, min_bounding_circle (lat/lon = center), union, intersection, boundary (outline of their areas) - returns a new `entity_id`; an empty intersection or boundary returns none. |
| `make_isochrone` | Area reachable from an `origin` within `max_duration_min` or `max_distance_m` for a `profile` (GraphHopper) - returns a new `entity_id`; searches around it measure from the origin. |
| `range_search` | Places within `radius_m` of a `reference` place (0 = inside or touching), nearest first; `include`/`exclude` filter expressions with `*` patterns, a default exclude of street noise (`none` keeps all); `total_found`. |
| `knn_search` | The `limit` nearest places to a `reference` place by straight-line distance; same filters as `range_search`. |
| `knn_search_network` | The `limit` nearest places to a `reference` place by route distance for a `profile` (car, bike, foot), exact among the 100 nearest by straight line; same filters as `range_search`. |
| `spatial_join` | Pairs from two lists of `entity_ids` (from `range_search` or `knn_search` results) within `max_distance_m` between shapes; inner, left, semi, anti; `limit_per_a`; at most 100 places per set. |

## Tests

See `tests/README.md`.

## License

Server code: Apache 2.0, see `LICENSE`. Bundled OSM data in `data/`: ODbL, see `data/README.md`.
