# LimoBench

Benchmark for street-level spatial awareness of LLMs on OpenStreetMap (OSM) data of West Lafayette, Indiana, with and without spatial tools served by the Limo MCP server (PostGIS, Photon, GraphHopper).

## Contents

```text
data/west_lafayette/v1/   # 70 location queries with ground truth (benchmark_tasks.json),
                          # 12 semantic queries (semantic_tasks_12.json), task viewer
data/motivation/          # Figure 1 probe: 10 geocode + 10 reverse-geocode queries per city
                          # (benchmark_<city>.json hold the queries and ground truth)
vendor/purduedb__limo/    # Limo MCP server (server.ts), DB/Photon/GraphHopper setup, tests, and its own copy of the
                          # West Lafayette OSM extract (data/west_lafayette.pbf)
docker-compose.local.yml  # Local Nominatim/Overpass/OSRM/tiles
src/                      # run_baseline.py, run_mcp.py, evaluate.py (see src/EVALUATION.md)
prompts/unified/          # System prompts (rendered/) and default locations
tests/                    # Offline tests (model calls mocked)
```

Seven operations, ten location queries each: geocoding, reverse geocoding, directionality, routing, nearest neighbors, range queries, spatial joins. Each task has a natural-language `prompt`, metadata, and `answer` (ground truth, read only by the evaluator).

## Setup

```bash
conda env create -f environment.yml && conda activate benchmark
cp .env.example .env     # add one provider API key (openai, openrouter, gemini, anthropic, genai)
make test                # offline
```

The baseline needs no services. For the MCP condition, install PostgreSQL with PostGIS (3.6.4+), then build the OSM database and services:

```bash
make -C vendor/purduedb__limo   # loads OSM into PostGIS, builds Photon and GraphHopper
```

The West Lafayette OSM extract is included twice, at `data/west_lafayette.pbf` (used by `docker-compose.local.yml`) and `vendor/purduedb__limo/data/west_lafayette.pbf` (used by Limo's own setup), so Limo can later live in its own repository. For another region, cut a bbox with `osmium extract --bbox=... <region>.osm.pbf -o <out>.pbf`.

The third-party MCP servers used for comparison in the paper are not included.

Ground truth was produced from the OSM snapshot of the authors' extract; a newer extract may differ slightly. See `vendor/purduedb__limo/README.md` and `data/README.md` there for details.

## Run and evaluate

```bash
make baseline mcp RUN_ID=my-run PROVIDER=openrouter MODEL=openai/gpt-6-luna
make eval RUN_ID=my-run     # writes outputs/my-run/eval.csv
```

Options: `TASK_FILE=...` (another task file, e.g. the semantic queries), `OPERATION=knn`, `LIMIT=3`, `REASONING=low`. Use a unique `RUN_ID` per experiment. Runs call paid model APIs. Results go to `outputs/<RUN_ID>/{baseline,mcp}/run.json`; scoring rules are in `src/EVALUATION.md`.

## License and attribution

The repository root `LICENSE` is Apache-2.0. Map data is derived from OpenStreetMap, (c) OpenStreetMap contributors, available under the [Open Database License](https://opendatacommons.org/licenses/odbl/); attribution is required. See `data/NOTICE.md`. See https://www.openstreetmap.org/copyright.
