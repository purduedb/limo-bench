# Tests

Tests against the live `server.ts` MCP server, talking the real MCP stdio protocol (not the DB directly) - exercises schema validation, dispatch, and response shaping, not just SQL. Vendor-specific: these test this server's own tool names/schemas, not a generic MCP contract.

**Requires**: run `make` from this package's root first (see the top-level README) to load the `osm` Postgres DB these tests need, plus the Photon and GraphHopper services. Also needs the `benchmark` conda env and `npx`/`tsx` available.

Run everything (from this package's root, one level up):
```bash
python -m pytest tests/ -v
```

Run one file or one test:
```bash
python -m pytest tests/test_l1b_range_knn.py -v -s
python -m pytest tests/test_l1b_range_knn.py::test_range_search_radius_membership_is_exact_at_the_edge -v -s
```

## Layers

Each layer tests its own tools and relies on the layers below being verified.

| Layer | File | Tools |
|---|---|---|
| L0 basic blocks | `test_l0_places.py` | `geocode` (POI, building, landmark, park, boundary, house address, street merge, guidance, input checks), `reverse_geocode` (distance, bearing, guidance), `lookup` (incl. multi-row entities, relations stored as lines), address (tags, then Photon, state code)|
| L1a two-place tools | `test_l1a_route_bearing.py` | `route`, `bearing`, `knn_search_network` (exactly the k nearest by route among the 100 straight-line nearest, each routed independently via GraphHopper; origin and measures, left-out and candidate-cap guidance, no-route guidance, units); a GraphHopper outage (stubbed) is a retry error, never no route |
| L1a derived geometry | `test_l1a_derived_geom.py` | `make_geom` (each op vs the oracle, lat/lon, shapes without area, empty results, reference checks, tmp expiry), `make_isochrone` (budget semantics, origin as its lat/lon, default profile, checks) |
| L1b single-reference tools | `test_l1b_range_knn.py` | `range_search`, `knn_search`: membership and nearest-first order per reference kind, a multi-row entity once, include/exclude expression strings (each operator, and/or/not, parentheses, quoting, case, whitespace, canonical echo, parse errors with position), default vs `none` exclude, total_found, guidance, bearing, limits, units |
| L2 join | `test_l2_spatial_join.py` | `spatial_join`: sets are entity_id lists (also the ids of `range_search`, `knn_search`, `knn_search_network` results), pairing, join types (pairs vs locations), empty sets, unknown ids, `limit_per_a`, `max_distance_m` omitted or between shapes (agrees with `range_search`), 100 ids per set, output shape, determinism |
| Composition | `test_chains.py` | geocode, reverse_geocode, route, make_geom, make_isochrone feeding range/knn/route/bearing, and range searches whose ids feed join |
| Client side | `test_user_location.py` | `get_user_location` (elicitation: accept, decline, unsupported client) |

## Shared modules

- `harness.py`: MCP client (`call`, one `server.ts` per test module), `ref()` (free text geocoded to an entity_id first), `place()` (a PlaceRef argument), `rejects`, `assert_converted`.
- `oracle.py`: independent SQL oracle (read-only but for two tmp_geom test writes; an entity is all rows under its id) (`near`, `knn`, `dist`, `azimuth`, `bearing_to_entity`, `join`, route/geometry facts), a filter AST (`T`, `AND`, `OR`, `NOT`, or a `{key: value}` map) with its own SQL translation and `render` to the expression string a tool takes (the server's parser is never reused), and response checkers (`assert_range`, `assert_knn`, `assert_join`).
- `fixtures.py`: entities picked by property from whatever extract is loaded (dense anchor, categories, uniquely named places, route trio, polygons); every query is bounded around the anchor, so cost does not grow with the DB.
- `geo.py`: pure spherical destination point and compass names (distance and bearing come from the PostGIS geography oracle).
- `conftest.py`: session fixtures `oracle` / `fx` and the per-tool latency report.

Nothing is hardcoded to one extract: expected values come from the oracle at test time. A test skips only when the extract lacks a fixture it needs (reason in `-rs`).

## Slow tests

The suite runs in about 50 s. The slowest are the wide-radius `range_search` / `knn_search` cases in `test_l1b_range_knn.py` and the tests that start a second server (`test_real_column_filters_survive_a_non_default_search_path`, the GraphHopper stub tests).

## Mutation check

`LIMO_SERVER_DIR` points the suite at another directory holding `server.ts` (with `node_modules` and `.env`), e.g. a copy with an injected bug:
```bash
LIMO_SERVER_DIR=/path/to/patched python -m pytest tests/ -q
```

A test failure means the tool returned something wrong or crashed - re-run with `-s` to see the raw request/response.
