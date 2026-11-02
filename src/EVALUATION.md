# Benchmark Evaluation

Run and score both conditions with the same task file:

```bash
make all RUN_ID=my-study PROVIDER=openrouter MODEL=openai/gpt-6-luna
make eval RUN_ID=my-study
```

The evaluator accepts the current run JSON. Ground truth is loaded from the benchmark paths in `config.tasks`, never from the run's transcript or model output. To use a different benchmark file:

```bash
python src/evaluate.py outputs/my-study/mcp/run.json --tasks data/my-benchmark.json --csv outputs/my-study/eval.csv
```

Both runners default to one warm-up of the first selected task (`--warmup-runs 1`). Full records are saved in `warmup_tasks`, with the same structure as `tasks`, but evaluation reads only `tasks`. Use `ARGS="--warmup-runs 0"` with Make to disable warm-up. `run.started_at` is recorded after warm-up and its delay, so run duration excludes warm-up. Sum the two record lists separately when reporting benchmark cost and total API spend.

Task IDs match run entries to benchmark answers. Pending entries and unlabelled benchmark tasks are skipped. Keep the original task files available or supply `--tasks` when moving logs to another machine.

## Repeated runs

Use a unique `RUN_ID` for each condition/configuration. MCP tasks are sequential within a runner; these commands also run repetitions sequentially to avoid competing for local tool services.

```bash
for n in 1 2 3; do
  make mcp RUN_ID=my-study/run$n PROVIDER=openrouter MODEL=openai/gpt-6-luna REASONING=low
  python src/evaluate.py outputs/my-study/run$n/mcp/run.json --csv outputs/my-study/run$n/eval.csv
done
python src/evaluate.py outputs/my-study/run*/mcp/run.json --csv outputs/my-study/eval_all.csv
```

Use `OPERATION=knn`, `LIMIT=3`, or `ARGS="--task-id WL-DIR-000"` to select tasks. Use `REASONING=medium` or `ARGS="--temperature 0.5"` for configuration sweeps.

## Shared metrics

- `correct` scores the final prediction against benchmark ground truth. Tool outputs supply diagnostics, never the expected answer. An empty ground-truth set requires an explicit empty result; a missing answer or execution failure is not a correct empty set.
- `grounded` means the claimed result appears in tool output. `correct_grounded` combines both checks.
- `gt_in_tool_output` distinguishes successful retrieval from answer-selection failures.
- Baseline tool diagnostics are blank, since there are no tools.
- Tool calls fall in exactly one class: failed (`n_tool_errors`: errored, timed out or returned an error), repeated identical (`n_redundant_calls`: succeeded, same tool and arguments as an earlier successful call), or clean (`n_clean_calls`: the rest); they add up to `tool_calls`. Failed takes precedence, and a successful retry of a failed call is clean.
- Latency columns use seconds; token columns include both raw counts and thousands. Model latency includes SDK retries. Cost uses reported usage when available, otherwise the local estimate.
- `format_ok` checks the result shape expected by the task; `schema_ok` checks the shared JSON output contract. With `json_valid` they are conformance diagnostics only.
- `accuracy_score`, `outcome`, `answer_valid` and `correct` (`outcome == correct`) come from `quality_metrics`; the tiers (geocode 50/200 m, route 1%/10%, direction exact/adjacent, set F1) and recovery rules are in `analysis/README.md` (Row 2). The per-operation "scorer pass" notes below are the stricter scorer-level check (name match, route 0.25%) that drives `failure_mode`, `grounded` and `correct_strict`; it is not the CSV `correct` column.
- Unparsed replies: `recover_answer` takes the first complete JSON object of the final reply unless a later object states another result; nested measures or labels count when all carriers agree (`single_result`).
- `--regeocode` adds a forward-geocoding check for reverse answers, using MCP and optional Mapbox fallback. Ordinary scoring makes no live tool calls.

The terminal report groups tasks by operation and prints aggregate accuracy, latency, usage, and operation-specific diagnostics. `--csv` writes task rows and operation averages. Run offline regression checks with `make test`.

## Operation metrics

**geocode / geocode_reverse**
* scorer pass: name matches the target (exact; case, punctuation, parenthetical aliases ignored) or the answer's `entity_id` is the target's (and it states a name), and the answer's lat/lon is within 50 m (numeric; no coordinates fails).
* `failure_mode`: `none`, `no_answer`, `no_name`, `wrong_location`, `wrong_name` (within 5 m, name not matched), `hallucinated` (not a returned row), `wrong_type`, `nearby_wrong_feature`; without tools `wrong_place` replaces the last three.
* `grounded`: resolves to a returned row by `entity_id`, otherwise by name (and within 5 m if coordinates are stated), or by coordinates within 5 m when unnamed.
* `osm_id_match`: the answer's own `entity_id` is the target's (blank if none); `row_is_target`: the row the answer resolves to is the target.

**directionality**
* scorer pass: label equals the ground-truth label (8 points, 45-degree sectors, edges round up, e.g. 22.5 gives NE); case and padding ignored.
* `failure_mode`: `none`, `no_answer`, `invalid_label`, `reversed` (opposite, likely swapped origin/destination), `adjacent_off` (one step), `wrong`.
* `adjacent`: within one step (exact included); `angular_error_deg`: label centre vs true bearing.
* `grounded`: label matches a returned `bearing` result; `gt_in_tool_output`: a returned bearing has the true label.
* `endpoints_found`: A and B in returned rows; `endpoints_used`: `bearing` args reference both; `args_order_ok`: last resolved call has origin=B, destination=A.

**route**
* scorer pass: the answer's distance (`distance_m`, or `distance_km` / `distance_mi` converted) is within `max(0.25%, 1 m)` of the ground-truth `distance_m` (`ROUTE_REL_TOL`, `ROUTE_ABS_TOL_M`); ground truth is GraphHopper on the same endpoints, so the tolerance only absorbs rounding and stays below the smallest gap between travel modes (0.4%). Duration never gates. Unreachable ground truth is correct only for `status: "unanswerable"`.
* `failure_mode`: `none`, `no_answer` (no numeric distance), `unit_error` (off by 1000 or 1609.344, e.g. km as m), `reversed_endpoints` / `wrong_endpoint` (last successful `route` call is B to A, or not the A/B places), `wrong_mode` (call `profile`, default car, is not the ground-truth mode), `off_tolerance` (within 25% of truth, no diagnosed cause), `wrong`.
* `dist_err_m`, `dist_err_pct`; `time_err_min`, `time_err_pct` only when the answer states a travel time (`duration_min`; the answer key's `time_min` is read as the same quantity).
* `mode_ok`: last successful `route` call has the ground-truth profile; `tool_distance_m`, `tool_time_min`: what that call returned.
* `grounded`: the answer's distance matches a `route` output (same tolerance); `gt_in_tool_output`: some `route` output matches the ground truth.
* `endpoints_found`: A and B in returned rows; `endpoints_used`: `route` args reference both; `args_order_ok`: last resolved call has origin=A, destination=B.

**knn**
* scorer pass: the answered set equals the ground-truth k nearest, nothing missing, extra or repeated; order never gates (`exact_order` reports it). An item is a ground-truth place when its name matches (`name_eq`) or its `entity_id` names the place (a name is also required for named targets), and any stated lat/lon is within 50 m; matching is one-to-one, so a repeat is an extra. The answer is `locations`, or a single `location` for k=1.
* Ties: tied ground-truth `distance_m` values are interchangeable in the order metrics; ties across the cutoff are not representable (the task lists only k places; none of the 10 tasks has one, checked in PostGIS).
* Set metrics (order ignored): `n_gt`, `n_answered`, `n_correct_items`, `n_missing`, `n_extra` (includes repeats), `n_duplicates`, `precision`, `recall`, `f1`, `jaccard`. An empty answer scores 0 (both sides empty scores 1). The report adds micro p / r / f1 pooled over all items; the AVG row holds per-task means.
* Order metrics (the answer's own list order): `rank_match` (share of ranks holding the right place), `exact_order`, `kendall_tau` (matched items, tied pairs skipped), `top1_correct` (nearest place first).
* Distances (stated per matched item; `geodesic_distance_*` for euclidean, `distance_*` for network, the other as fallback): `n_dist`, `dist_mae_m`, `dist_mre_pct`. `dist_ok` (network only): all within `max(0.25%, 1 m)`, the route tolerance, since ground truth is the same GraphHopper route. Euclidean ground truth is point to point while the tool measures shape to shape (up to ~30% apart), so it is error only.
* `failure_mode`: `none`, `no_answer`, `wrong_reference`, `wrong_distance_type` (tool or profile), `wrong_filter`, `wrong_k` (limit below k), `hallucinated` (an answered item is not a returned row), `missing_items`, `extra_items`, `wrong` (both); the first that applies in this order, call checks only when a knn call exists.
* Diagnostics of the last successful `knn_search` / `knn_search_network` call (last attempted call if none succeeded): `reference_ok` (entity_id or coordinates within 50 m of the reference), `distance_type_ok` (`knn_search` for euclidean, `knn_search_network` for network), `profile_ok` (network only, default car), `k_ok` (limit equals k; a larger limit is false but does not trigger `wrong_k`), `filter_ok` (include has exactly the ground-truth keys and allows every ground-truth value; value lists, `*` wildcards and case allowed, extra keys fail since keys AND), `n_knn_calls`.
* `gt_in_tool_output`, `tool_set_recall`: all / share of ground-truth places in any returned row; `n_dropped`: in the output but not answered (selection failure).
* `grounded`: every answered item is a returned row (`ungrounded_count` = those that are not).


**range_query**
* scorer pass: the answered places equal the ground-truth places in the region as sets: order ignored, repeats count as extras (`n_duplicates`), nothing missing or extra. Place matching follows the KNN rules; nameless targets also accept an unnamed answer within 5 m.
* Set metrics as in knn: `n_gt`, `n_answered`, `n_correct_items`, `n_missing`, `n_extra`, `precision`, `recall`, `f1`, `jaccard`; `failure_mode`: `none`, `no_answer`, `missing_items`, `extra_items`, `wrong` (both).

**spatial_join**
* scorer pass: inner and left joins compare `pairs`, semi and anti joins compare `locations`, as sets (rules as range_query). An inner pair matches either way round (also for a self join); a left join keeps roles, and an unmatched row has `location_b` null.
* `correct_strict`: also requires each pair's roles (`location_a` from set_a); self joins remain unordered. `correct_any_shape`, `f1_any_shape`: accept equivalent shapes, e.g. a semi join answered as pairs or a partner nested in its place row. `shape_recovered`: the expected list is absent and the other shape carries the rows, so it is scored (semi join as pairs, anti join as partnerless pairs, pair join as places with nested partners).
* Set metrics and `failure_mode` as range_query.
* knn / range / join also report `correct_name`, `f1_name` (stated coordinates ignored; diagnostic) and `grounded` / `ungrounded_count` (every answered place is a returned row, `spatial_join` pair rows included; blank without tools).
