Makefile targets (see `../Makefile`):

1. `make init_db`:
    - Creates a new Postgres database named `osm` (or `DB_NAME`).
    - Enables the `postgis` and `hstore` extensions.

2. `make load_db`:
    - Loads OpenStreetMap data from a `.pbf` file (default: `../data/west_lafayette.pbf`) into the `osm` database.
    - Creates `planet_osm_*` tables and indexes using `osm2pgsql`.
        * `planet_osm_point`: Point features (POIs, amenities, addresses).
        * `planet_osm_line`: Linestring features (all roads, paths, rivers, etc.).
        * `planet_osm_polygon`: Polygon features (buildings, land use, boundaries).
    - Indexes:
        * `planet_osm_point_way_idx`, `planet_osm_line_way_idx`, `planet_osm_polygon_way_idx`: GiST index on each table's geometry column (`way`), for spatial lookups.
    - Byproduct tables (unused downstream):
        * `planet_osm_roads`: Major roads only (primary/secondary/trunk/motorway), for low-zoom rendering.
        * `osm2pgsql_properties`: osm2pgsql's own import metadata (prefix, output format).
        * `spatial_ref_sys`: PostGIS's standard SRID reference table.
