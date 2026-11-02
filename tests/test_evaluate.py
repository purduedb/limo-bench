#!/usr/bin/env python3
"""Hand-built pass/fail cases for every scorer in src/evaluate.py, incl. baseline-style (no-tool) answers. Run: python tests/test_evaluate.py"""
import json
import os
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import evaluate as E

NO_TOOLS = {"calls": [], "rows": [], "outs": [], "tools": False}
north = lambda p, m: {**p, "lat": p["lat"] + m / 111320}  # moved m metres north


def place(name, lat, lon, osm_id=1, table="osm.planet_osm_point"):
    return {"osm_id": osm_id, "source_table": table, "name": name, "lat": lat, "lon": lon}


def test_name_folding():
    assert E.name_eq("Ross\u2013Ade Stadium", "Ross-Ade Stadium") and E.name_eq("Cordova Rec", "C\u00f3rdova Rec")
    assert E.name_eq("Cafe\u0301", "Caf\u00e9") and not E.name_eq("Ross Ade", "Ross Ades")


def test_geocode():
    gt = place("Neil Armstrong Hall of Engineering (ARMS)", 40.4309, -86.9150, table="osm.planet_osm_polygon")
    ok = lambda v: E.score_geocode(v, gt, NO_TOOLS)
    loc = lambda **kw: {"location": {"name": gt["name"], "lat": gt["lat"], "lon": gt["lon"], **kw}}
    assert ok(loc())["correct"]
    assert ok(loc(name="Neil Armstrong Hall of Engineering"))["correct"]  # parenthetical alias
    assert ok(loc(name="Neil Armstrong Hall of Engineering, Purdue University"))["correct"]  # trailing comma qualifier
    assert not ok(loc(name="Purdue University campus, near Neil Armstrong Hall"))["correct"]
    assert not ok(loc(name="Armstrong Hall"))["correct"]  # no partial match
    assert ok({"location": north(loc()["location"], 45)})["correct"] and not ok({"location": north(loc()["location"], 60)})["correct"]
    assert not ok({"location": north(loc()["location"], 1000)})["correct"]  # right name, coordinates a kilometre away
    assert not ok(loc(lat=str(gt["lat"]), lon=str(gt["lon"])))["correct"]  # string coordinates are not numbers
    a = ok(loc(lat=gt["lon"], lon=gt["lat"]))  # lat/lon swapped
    assert not a["correct"] and a["failure_mode"] == "wrong_location"
    a = ok({"location": {"name": gt["name"], "lat": None, "lon": None}})  # name only, no coordinates
    assert not a["correct"] and a["err_m"] is None
    for junk in ({"status": "unanswerable", "notes": "x"}, None, "text", {}):
        a = ok(junk)
        assert not a["correct"] and a["failure_mode"] == "no_answer", junk
    a = ok({"location": {"name": "Some Other Hall", "lat": 40.5, "lon": -86.9}})
    assert not a["correct"] and a["failure_mode"] == "wrong_place"  # no tools: not judged hallucinated


def test_geocode_reverse():
    gt = place("Mackey Arena", 40.4333, -86.9162)
    sc = lambda name: E.score_geocode_reverse({"location": {"name": name, "lat": gt["lat"], "lon": gt["lon"]}}, gt, NO_TOOLS)
    a = sc("Purdue University")  # echoed query coordinates, wrong name
    assert not a["correct"] and a["failure_mode"] == "wrong_name" and a["err_m"] < 1
    assert sc("mackey arena")["correct"]
    a = E.score_geocode_reverse({"location": {"name": None, "lat": gt["lat"], "lon": gt["lon"]}}, gt, NO_TOOLS)
    assert not a["correct"] and a["failure_mode"] == "no_name"


def test_directionality():
    A, B = place("A", 40.0, -86.0, 1), place("B", 40.0, -86.1, 2)
    sc = lambda v, b=90.0: E.score_directionality(v, {"bearing_deg": b, "location_a": A, "location_b": B}, NO_TOOLS)
    assert sc({"direction": "E"})["correct"] and sc({"direction": " e "})["correct"]
    assert not sc({"bearing_deg": 90.0})["correct"] and sc({"bearing_deg": 90.0})["failure_mode"] == "no_answer"  # a bearing is not the asked label
    assert sc({"pairs": [{"location_a": A, "location_b": B, "direction": "E"}]})["correct"]  # direction on the only pair
    assert sc({"pairs": [{"direction": "E"}, {"direction": "e"}]})["correct"]  # carriers agree
    assert not sc({"pairs": [{"direction": "E"}, {"direction": "W"}]})["correct"]  # carriers disagree: no single answer
    assert sc({"direction": "W"})["failure_mode"] == "reversed" and sc({"direction": "NE"})["failure_mode"] == "adjacent_off"
    assert sc({"direction": "East"})["failure_mode"] == "invalid_label"
    assert sc({"direction": "E"})["grounded"] is False  # scorer-level; evaluate() blanks it for baseline


def test_route():
    A, B = place("A", 40.0, -86.0, 1), place("B", 40.0, -86.1, 2)
    G = {"location_a": A, "location_b": B, "mode": "car", "distance_m": 4000.0, "time_min": 6.0}
    sc = lambda v: E.score_route(v, G, NO_TOOLS)
    for v in ({"distance_m": 4009}, {"distance_km": 4.0}, {"distance_mi": 4000 / 1609.344}, {"pairs": [{"location_a": A, "location_b": B, "distance_m": 4000}]}):
        assert sc(v)["correct"], v
    pair = {"location_a": A, "location_b": B, "distance_m": 4000}
    assert sc({"pairs": [pair], "geodesic_distance_m": 3000, "duration_min": 6.0})["correct"]  # asked value only on the single pair, other top-level measures do not hide it
    assert sc({"pairs": [pair, pair]})["correct"] and not sc({"pairs": [pair, {**pair, "distance_m": 3000}]})["correct"]  # disagreeing carriers: no single answer
    assert sc({"pairs": [{**pair, "distance_m": 3000}], "distance_m": 4000.0})["correct"]  # a top-level value wins over the pair's
    assert not sc({"distance_m": 4011})["correct"] and sc({"distance_m": 4011})["failure_mode"] == "off_tolerance"  # 0.25% is 10 m
    assert sc({"distance_m": 4.0})["failure_mode"] == "unit_error"
    assert not sc({"geodesic_distance_m": 4000})["correct"]  # straight line is not the travel distance asked
    assert not sc({"status": "unanswerable"})["correct"] and not sc({"duration_min": 6.0})["correct"]  # time never gates, nor substitutes
    assert sc({"answer": {"distance_m": 4000}})["correct"] and sc({"measure": {"distance_km": 4.0}, "location": B})["correct"]  # one nested carrier: a wrapper
    assert sc({"measure": {"distance_m": 4000}, "pairs": [pair]})["correct"]  # agreeing carriers
    assert not sc({"location": {**A, "distance_m": 4000}, "pairs": [{"distance_m": 3000}]})["correct"]  # disagreeing carriers


def knn_gt(n=3, **kw):
    return {"k": n, "distance_type": "euclidean", "include": {"amenity": "library"}, "reference": place("Ref", 39.9, -86.0, 99),
            "locations": [{**place(f"Lib {i}", 40.0 + i / 100, -86.0, i), "distance_m": 100.0 * (i + 1)} for i in range(n)], **kw}


def test_knn():
    gt = knn_gt()
    ans = lambda items: {"locations": items}
    full = [{k: g[k] for k in ("name", "lat", "lon")} for g in gt["locations"]]
    sc = lambda v: E.score_knn(v, gt, NO_TOOLS)
    a = sc(ans(full))
    assert a["correct"] and a["f1"] == 1 and a["grounded"] is None and a["ungrounded_count"] is None  # no tools: grounding n/a
    assert sc(ans(full[::-1]))["correct"]  # order never gates
    a = sc(ans([north(p, 300) for p in full]))  # right names, coordinates from memory 300 m off
    assert not a["correct"] and a["f1"] == 0 and a["correct_name"] and a["f1_name"] == 1 and a["failure_mode"] == "wrong"
    a = sc(ans(full + [{"name": "Invented Library", "lat": 40.5, "lon": -86.0}]))
    assert not a["correct"] and a["failure_mode"] == "extra_items" and a["precision"] == 0.75
    assert sc(ans([{"name": p["name"]} for p in full]))["correct"]  # documented: unstated coordinates are not checked
    assert not sc(ans(["Lib 0", "Lib 1", "Lib 2"]))["correct"]  # bare strings are not places
    a = sc(ans(full + full[:1]))  # a repeat is an extra
    assert not a["correct"] and a["n_duplicates"] == 1 and a["n_extra"] == 1
    assert not sc({"status": "unanswerable"})["correct"]
    one = knn_gt(1)
    assert E.score_knn({"location": full[0]}, one, NO_TOOLS)["correct"]  # k = 1 as a single location


def test_range_and_count():
    gt = {"locations": [place("P0", 40.0, -86.0, 1), place("P1", 40.001, -86.0, 2)]}
    items = [{"name": "P0", "lat": 40.0, "lon": -86.0}, {"name": "p1", "lat": 40.001, "lon": -86.0}]
    a = E.score_range({"locations": items, "count": 2}, gt, NO_TOOLS)
    assert a["correct"] and a["grounded"] is None
    assert E.answer_format({"locations": items, "count": 2}, gt)["format_ok"]  # count is a measure, may accompany a list
    assert E.answer_format({"count": 2}, gt)["format_mode"] == "wrong_field"
    rows = [{"loc": {"entity_id": f"point:{i + 1}", **{k: v for k, v in g.items() if k in ('name', 'lat', 'lon')}}} for i, g in enumerate(gt["locations"])]
    ctx = {"calls": [], "rows": rows, "outs": [], "tools": True}
    assert E.score_range({"locations": items}, gt, ctx)["grounded"]
    a = E.score_range({"locations": items + [{"name": "X", "lat": 41.0, "lon": -86.0}]}, gt, ctx)
    assert not a["correct"] and not a["grounded"] and a["ungrounded_count"] == 1


def test_precision_scoring():
    """answer scoring "precision": accepted list, correct = at least one item and none outside it; recall is not scored."""
    gt = {"scoring": "precision", "locations": [place("P0", 40.0, -86.0, 1), place("P1", 40.001, -86.0, 2), place("P2", 40.002, -86.0, 3)]}
    it = lambda *i: [{"name": f"P{k}", "lat": 40.0 + k / 1000, "lon": -86.0} for k in i]

    def outcome(items):
        pred = {"locations": items}
        row = E.precision_mode(E.score_range(pred, gt, NO_TOOLS))
        return row["correct"], E.quality_metrics("range_query", pred, row, gt)["outcome"]
    assert outcome(it(0)) == (True, "correct") and outcome(it(2, 0)) == (True, "correct")  # subset of the accepted list
    assert outcome(it(0, 1, 2)) == (True, "correct")
    assert outcome(it(0) + [{"name": "X", "lat": 41.0, "lon": -86.0}]) == (False, "partially_correct")
    assert outcome([{"name": "X", "lat": 41.0, "lon": -86.0}]) == (False, "incorrect")
    assert outcome([]) == (False, "incorrect")  # nothing returned
    assert outcome(it(0, 0)) == (True, "correct")  # a repeat of a matched place is dropped, not an extra
    twice = it(0, 1) + [{"name": "P0", "lat": 40.0003, "lon": -86.0}]  # same place again 33 m away (point and polygon)
    row = E.precision_mode(E.score_range({"locations": twice}, gt, NO_TOOLS))
    assert (row["precision"], row["n_extra"], row["n_duplicates"]) == (1.0, 0, 1)
    assert outcome(it(0) + [{"name": "X", "lat": 41.0, "lon": -86.0}] * 2)[1] == "partially_correct"  # a repeated wrong place stays extra
    assert not E.score_range({"locations": it(0)}, gt, NO_TOOLS)["correct"]  # default scoring still demands the full set
    pred = {"locations": it(0) + [{"name": "X", "lat": 41.0, "lon": -86.0}]}
    assert E.quality_metrics("range_query", pred, E.precision_mode(E.score_range(pred, gt, NO_TOOLS)), gt)["accuracy_score"] == 50.0  # accuracy = precision
    buf = {**gt, "locations": gt["locations"][:1] + [{**g, "accepted": "buffer"} for g in gt["locations"][1:]]}
    with tempfile.TemporaryDirectory() as d:  # pipeline: recall and f1 count core (non-buffer) items only
        tasks, run = Path(d) / "tasks.json", Path(d) / "run.json"
        tasks.write_text(json.dumps([{"task_id": "T1", "operation": "range_query", "answer": buf}]))
        out = lambda items: {"task_id": "T1", "output": {"answer": {"locations": items}, "metrics": {"latency_ms": 1, "cost_usd": 0, "zero_hit": False}}, "trace": {"transcript": [], "events": []}}
        run.write_text(json.dumps({"config": {"condition": "baseline", "model": "m", "tasks": [str(tasks)]}, "tasks": [out(it(0)), out(it(1, 2))]}))
        core, extra = E.evaluate(None, [str(run)])["range_query"]
    assert (core["accuracy_score"], core["recall"], core["f1"], core["outcome"]) == (100.0, 1.0, 1.0, "correct")
    assert (extra["accuracy_score"], extra["recall"], extra["outcome"]) == (100.0, 0.0, "correct")  # buffer-only answer passes, core recall shows it


def test_csv_columns_ignore_task_order():
    """eval.csv columns do not depend on which operation comes first."""
    rows = {"geocode": [{"task_id": "G", "operation": "geocode", "err_m": 1.0}], "knn": [{"task_id": "K", "operation": "knn", "f1": 1.0}]}
    with tempfile.TemporaryDirectory() as d:
        heads = []
        for order in (rows, dict(reversed(rows.items()))):
            E.write_csv(Path(d) / "e.csv", order)
            heads.append((Path(d) / "e.csv").read_text().splitlines()[0])
    assert heads[0] == heads[1]


def test_join():
    P = lambda i: {"name": f"p{i}", "lat": 40.0 + i / 1e3, "lon": -86.0}
    G = lambda i: {**P(i), "osm_id": i, "source_table": "osm.planet_osm_point"}
    inner = {"join_type": "inner", "set_a": {"a": 1}, "set_b": {"b": 1}, "pairs": [{"location_a": G(0), "location_b": G(1)}]}
    assert E.score_join({"pairs": [{"location_a": P(1), "location_b": P(0)}]}, inner, NO_TOOLS)["correct"]  # inner pairs unordered
    assert not E.score_join({"pairs": [{"location_a": P(0), "location_b": P(2)}]}, inner, NO_TOOLS)["correct"]
    semi = {"join_type": "semi", "locations": [G(0)]}
    a = E.score_join({"pairs": [{"location_a": P(0), "location_b": P(5)}, {"location_a": P(0), "location_b": P(6)}]}, semi, NO_TOOLS)
    assert a["correct"] and a["shape_recovered"] and not a["correct_strict"]  # semi join as pairs: its left places, repeats dropped
    anti = {"join_type": "anti", "locations": [G(0)]}
    assert E.score_join({"pairs": [{"location_a": P(0), "location_b": None}]}, anti, NO_TOOLS)["correct"]
    assert not E.score_join({"pairs": [{"location_a": P(0), "location_b": P(5)}]}, anti, NO_TOOLS)["shape_recovered"]  # a partner contradicts an anti join
    a = E.score_join({"locations": [{**P(0), "nearest": P(1)}]}, inner, NO_TOOLS)
    assert a["correct"] and a["shape_recovered"]  # pair join as places with nested partners
    assert not E.score_join({"locations": [P(0)]}, inner, NO_TOOLS)["shape_recovered"]  # no partner: the pairs are missing
    # grounding sees spatial_join's own pair rows
    res = {"tool_calls": [{"tool": "spatial_join", "args": {}, "mcp_calls": [{"result": {"pairs": [
        {"location_a": {"entity_id": "point:0", **P(0)}, "location_b": {"entity_id": "point:1", **P(1)}}]}}]}]}
    ctx = E.build_ctx(res, None)
    assert E.score_join({"pairs": [{"location_a": P(0), "location_b": P(1)}]}, inner, ctx)["grounded"]


def test_missing_answers_do_not_match_empty_truth():
    for scorer, truth, empty in [
        (E.score_range, {"locations": []}, {"locations": []}),
        (E.score_join, {"join_type": "semi", "locations": []}, {"locations": []}),
        (E.score_join, {"join_type": "anti", "locations": []}, {"locations": []}),
        (E.score_join, {"join_type": "inner", "pairs": []}, {"pairs": []}),
        (E.score_join, {"join_type": "left", "pairs": []}, {"pairs": []}),
        (E.score_knn, {**knn_gt(), "locations": []}, {"locations": []}),
    ]:
        assert scorer(empty, truth, NO_TOOLS)["correct"]
        for missing in [None, "text", {}, {"status": "unanswerable"}, {"locations": [1]}]:
            row = scorer(missing, truth, NO_TOOLS)
            assert not row["correct"] and row["failure_mode"] == "no_answer", (scorer.__name__, missing, row)
            assert row["f1"] == 0


def test_outcome_ignores_benign_format():
    """invalid_answer only when no usable required result; fences, echoed or unknown fields keep the semantic outcome."""
    A, B = place("A", 40.0, -86.0, 1), place("B", 40.0, -86.1, 2)
    route = {"location_a": A, "location_b": B, "mode": "car", "distance_m": 4000.0}
    knn, semi, inner = knn_gt(1), {"join_type": "semi", "locations": [A]}, {"join_type": "inner", "pairs": [{"location_a": A, "location_b": B}]}

    def outcome(op, scorer, pred, gt, fenced=False):
        row = {"json_valid": not fenced, **scorer(pred, gt, NO_TOOLS), **E.answer_format(pred, gt), **E.answer_schema(pred)}
        q = E.quality_metrics(op, pred, row, gt)
        return q["outcome"], q["accuracy_score"]
    a = {k: A[k] for k in ("name", "lat", "lon")}
    assert outcome("geocode", E.score_geocode, {"location": a}, A, fenced=True) == ("correct", 100.0)
    assert outcome("geocode", E.score_geocode, {"location": north(a, 201)}, A) == ("incorrect", 0.0)
    assert outcome("geocode", E.score_geocode, {"location": {"name": "A"}}, A)[0] == "invalid_answer"
    d = {"bearing_deg": 90.0, "location_a": A, "location_b": B}
    assert outcome("directionality", E.score_directionality, {"direction": "E", "location": a}, d) == ("correct", 100.0)  # echoed origin
    assert outcome("directionality", E.score_directionality, {"direction": "NE"}, d)[0] == "partially_correct"
    assert outcome("directionality", E.score_directionality, {"direction": "East"}, d)[0] == "invalid_answer"
    for v, want in [({"distance_m": 4040, "answer": "4.04 km"}, "correct"), ({"distance_km": 4.4, "foo": 1}, "partially_correct"),
                    ({"distance_m": 4401}, "incorrect"), ({"distance": 4000}, "invalid_answer"), ("4000 m", "invalid_answer")]:
        assert outcome("route", E.score_route, v, route)[0] == want, v
    assert outcome("knn", E.score_knn, {"locations": [knn["locations"][0]], "reference": a}, knn) == ("correct", 100.0)
    assert outcome("range_query", E.score_range, {"locations": []}, {"locations": []}) == ("correct", 100.0)  # explicit empty set
    q = lambda f1: E.quality_metrics("knn", {"locations": []}, {"f1": f1}, {"locations": []})["outcome"]
    assert (q(1.0), q(0.999), q(1e-9), q(0.0)) == ("correct", "partially_correct", "partially_correct", "incorrect")  # set ops: partial is 0 < F1 < 1
    assert outcome("range_query", E.score_range, {"status": "unanswerable"}, {"locations": []})[0] == "invalid_answer"
    assert outcome("knn", E.score_knn, {"location": a, "status": "unanswerable"}, knn)[0] == "invalid_answer"  # echoed place beside a decline
    assert outcome("spatial_join", E.score_join, {"pairs": [{"location_a": a, "location_b": None}]}, semi) == ("correct", 100.0)  # recovered shape
    assert outcome("spatial_join", E.score_join, {"locations": [a]}, inner)[0] == "invalid_answer"  # no partners: required pairs absent
    assert outcome("route", E.score_route, {"measure": {"distance_m": 4000}}, route) == ("correct", 100.0)
    gone = {**route, "distance_m": None}  # unreachable truth
    assert [outcome("route", E.score_route, v, gone)[0] for v in ({"status": "unanswerable"}, {"distance_m": 4000}, {"notes": "x"})] == ["correct", "incorrect", "invalid_answer"]


def test_recover_answer():
    """Unparsed replies: the first object counts unless a later object gives a different result."""
    r = E.recover_answer
    assert r('{"locations": [{"name": "A"}]},"notes": "x"}') == {"locations": [{"name": "A"}]}  # stray brace before notes
    assert r('{"distance_m": 5}\n\nWait, correcting format:\n{"distance_m": 5, "notes": "y"}') == {"distance_m": 5}  # same result twice
    assert r('{"direction": "N"}\nActually:\n{"direction": "NE"}') is None  # two different answers: ambiguous
    assert r('Let me check {"query": "x"} first') is None and r("") is None and r(None) is None  # no result object
    assert r('{"locations": [{"name": "A"}}, {"name": "B"}]}') is None  # broken first object: fragments are not repaired
    assert r([{"type": "text", "text": "{}"}]) is None  # content blocks, not a string


def test_join_shape_needs_absent_list():
    """A present but malformed expected list is invalid, not recovered from the other shape."""
    a, b = place("A", 40.0, -86.0), place("B", 40.1, -86.1)
    semi = {"join_type": "semi", "locations": [a]}
    pair = {"location_a": a, "location_b": b}
    assert E.score_join({"pairs": [pair]}, semi)["shape_recovered"]
    assert not E.score_join({"locations": [a, "B"], "pairs": [pair]}, semi)["shape_recovered"]


def test_pipeline_sides():
    """Same answer scores the same on both sides; baseline rows carry no tool diagnostics; raw tokens and run_id."""
    gt = place("Mackey Arena", 40.4333, -86.9162)
    ans = {"location": {"name": "Mackey Arena", "lat": 40.4333, "lon": -86.9162}}
    with tempfile.TemporaryDirectory() as d:
        tasks = os.path.join(d, "tasks.json")
        Path(tasks).write_text(json.dumps([{"task_id": "T1", "operation": "geocode", "answer": {"location": gt}}]))
        rows = {}
        for side in ("baseline", "mcp"):
            run = os.path.join(d, side, "geocode", "run2")
            os.makedirs(run)
            trace = {'task_id': 'T1', 'turn_count': 1, 'prompt_tokens': 1234, 'completion_tokens': 56, 'cached_tokens': 1000}
            transcript, events = [], []
            if side == "mcp":
                transcript = [{"role": "assistant", "tool_calls": [{"id": "c1", "function": {"name": "geocode", "arguments": '{"query":"Mackey"}'}}]},
                              {"role": "tool", "content": json.dumps({"locations": [{"entity_id": "point:1", **ans["location"]}]})}]
                events = [{"type": "tool_call", "call_id": "c1", "latency_ms": 0,
                           "request_transcript_index": 0, "result_transcript_index": 1}]
            document = {"config": {"condition": side, "model": "test", "tasks": [tasks]}, "tasks": [{"task_id": "T1",
                        "input": {"operation": "geocode"},
                        "output": {"answer": ans, "metrics": {**trace, "latency_ms": 1.0, "cost_usd": .001, "zero_hit": False}},
                        "trace": {"transcript": transcript, "events": events}}]}
            with open(os.path.join(run, 'run.json'), 'w') as f:
                json.dump(document, f)
            rows[side] = E.evaluate(None, [os.path.join(run, 'run.json')])['geocode'][0]
            csv_path = os.path.join(run, "eval.csv")
            E.write_csv(csv_path, {"geocode": [rows[side]]})
        b, m = rows["baseline"], rows["mcp"]
        assert b["correct"] and m["correct"] and b["side"] == "baseline" and m["side"] == "mcp"
        assert b["run_id"] == m["run_id"] == "run2"
        assert all(b[k] is None for k in E.TOOL_COLS if k in b) and b["tool_calls"] is None and b["grounded"] is None
        assert m["tool_calls"] == 1 and m["grounded"] and m["n_tool_errors"] == 0
        assert b["prompt_tokens"] == 1234 and b["cached_tokens"] == 1000 and b["prompt_tokens_k"] == 1.234


def test_pipeline_all_benchmark_tasks():
    """Both conditions score correct and wrong predictions consistently across the benchmark."""
    source = Path(__file__).parents[1] / 'data/west_lafayette/v1/benchmark_tasks.json'
    tasks = json.loads(source.read_text())
    wrong_place = {"name": "Unrelated test place", "lat": 0, "lon": 0}

    def clean(value):
        if isinstance(value, list):
            return [clean(item) for item in value]
        if not isinstance(value, dict):
            return value
        return {key: clean(item) for key, item in value.items() if key not in {"osm_id", "source_table"}}

    def prediction(task, correct):
        truth = task['answer']
        if task['operation'] == 'directionality':
            return {'direction': truth['direction'] if correct else E.LABELS_8[(E.LABELS_8.index(truth['direction']) + 4) % 8]}
        if task['operation'] == 'route':
            return {'distance_m': truth['distance_m'] if correct else 1e10}
        key = next(key for key in ('location', 'locations', 'pairs') if key in truth)
        if correct:
            return {key: clean(truth[key])}
        value = wrong_place if key == 'location' else [wrong_place] if key == 'locations' else [{'location_a': wrong_place, 'location_b': wrong_place}]
        return {key: value}

    with tempfile.TemporaryDirectory() as directory:
        for correct in (True, False):
            rows_by_side = {}
            for side in ('baseline', 'mcp'):
                entries = []
                for task in tasks:
                    transcript, events = [], []
                    if side == 'mcp':
                        # Correct retrieval must not rescue an incorrect final prediction.
                        transcript = [{'role': 'assistant', 'tool_calls': [{'id': 'call', 'function': {'name': 'fixture_result', 'arguments': '{}'}}]},
                                      {'role': 'tool', 'content': json.dumps(prediction(task, True))}]
                        events = [{'type': 'tool_call', 'call_id': 'call', 'latency_ms': 1,
                                   'request_transcript_index': 0, 'result_transcript_index': 1}]
                    entries.append({'task_id': task['task_id'], 'input': {'prompt': task['prompt'], 'operation': task['operation']},
                        'output': {'answer': prediction(task, correct), 'metrics': {'latency_ms': 2, 'cost_usd': .001, 'zero_hit': False}},
                        'trace': {'transcript': transcript, 'events': events}})
                path = Path(directory) / side / 'run.json'
                path.parent.mkdir(exist_ok=True)
                path.write_text(json.dumps({'config': {'condition': side, 'model': 'test', 'tasks': [str(source)]}, 'tasks': entries}))
                rows = [row for group in E.evaluate(None, [str(path)]).values() for row in group]
                assert len(rows) == len(tasks)
                assert all(row['correct'] is correct for row in rows), [(row['task_id'], row['correct']) for row in rows if row['correct'] is not correct]
                rows_by_side[side] = {row['task_id']: row['correct'] for row in rows}
            assert rows_by_side['baseline'] == rows_by_side['mcp']


def load_tests(loader, tests, pattern):
    """Include scorer scenarios in ordinary unittest discovery."""
    cases = [unittest.FunctionTestCase(E.self_check)]
    cases.extend(unittest.FunctionTestCase(value) for name, value in sorted(globals().items()) if name.startswith("test_"))
    return unittest.TestSuite(cases)


if __name__ == "__main__":
    unittest.main()


def test_call_classes_are_disjoint():
    """Failed first, then successful repeats of an earlier successful call; a retry that succeeds after a failure is clean."""
    ok = {"tool": "t", "args": {"a": 1}, "result": {}}
    bad = {"tool": "t", "args": {"a": 1}, "error": "boom"}
    calls = [bad, bad, ok, ok, ok, {"tool": "u", "args": {}, "result": {}}]  # failed x2 (one is a repeat of a failure), clean, repeat x2, clean
    m = E.common_metrics({"tool_calls": calls, "latency_ms": 1.0, "cost_usd": 0, "zero_hit": False}, {})
    assert (m["n_tool_errors"], m["n_redundant_calls"], m["n_clean_calls"], m["tool_calls"]) == (2, 2, 2, 6)
