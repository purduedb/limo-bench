# Data

`west_lafayette.pbf` - a West Lafayette, Indiana subset of Geofabrik's Indiana `.pbf`, plus the Indiana state boundary relation (so Photon can attach the state to addresses and rank the city boundary first when geocoding the city name). The Postgres database, Photon and GraphHopper are all built from `west_lafayette.pbf`; the patch adds only Indiana-wide or distant features and changes no local result. Used to load the `osm` Postgres/PostGIS/pgRouting database the tests and server run against; also a ready-made sample for trying the MCP server against a small, real dataset without downloading and extracting a full state/country extract yourself.

`server.ts` works with any city's `.pbf` from [Geofabrik](https://download.geofabrik.de/) - this one is just a convenient, pre-extracted starting point.

Data license and attribution: see [NOTICE.md](NOTICE.md).
