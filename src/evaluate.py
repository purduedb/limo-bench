#!/usr/bin/env python3
"""
Score MCP runs against benchmark ground truth: per-operation reports plus optional CSV.

Usage:
    python3 src/evaluate.py outputs/RUN/mcp/run.json [more results...] [--csv out.csv]

Notes:
    - Each run.json contains task inputs, predictions, metrics, and interactions.
    - Ground truth comes from config.tasks, or the explicit --tasks benchmark file.
    - Ground truth comes from saved task inputs (--tasks overrides).
    - New operation: one `score_<op>(pred, gt, ctx, geocoder=None) -> dict` with a `correct` key, registered in SCORERS.
    - ctx holds the tool calls, their outputs and every returned location row, so scorers can ground answers.
"""
import asyncio
import argparse
import csv
import itertools
import json
import math
import os
import re
import statistics as st
import sys
import unicodedata
from collections import Counter
from fnmatch import fnmatchcase
from pathlib import Path

DEFAULT_TASKS = "data/west_lafayette/v1/benchmark_tasks.json"
LABELS_8 = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
SECTOR_DEG = 360 / len(LABELS_8)  # width of one compass label
CORRECT_M = 50      # policy: geocode / reverse geocode pass radius, and how far a named answer may sit from its ground-truth place in knn/range/join
GROUND_M = 5        # nameless answer must sit this close to a returned row
NAMELESS_M = 5      # policy: a nameless ground-truth place matches an unnamed answer this close (no name to check)
REL_TOL = 0.10      # route_time pass tolerance
ROUTE_REL_TOL = 0.0025  # route distance pass tolerance, below the smallest gap between travel modes (0.4%)
ROUTE_ABS_TOL_M = 1     # route distance floor: whole-metre rounding on short routes
NEAR_REL_TOL = 0.25     # wrong answer this close counts as off_tolerance
UNIT_M = {"m": 1, "km": 1000, "mi": 1609.344}
KNN_TOOLS = {"euclidean": "knn_search", "network": "knn_search_network"}  # policy: tool names of the server under test, knn diagnostics only
TIME_KEYS = ("duration_min", "time_min")  # policy: the prompt and tools name travel time duration_min, the answer keys time_min
KNN_DEFAULT_LIMIT = 5   # policy: that server's knn limit when omitted, knn diagnostics only (k_ok, wrong_k)
RESULT_FIELDS = ("location", "locations", "pairs", "direction", "count", "answer")  # policy: the prompt's <output_format> result fields
MEASURE_FIELDS = ("count",)  # policy: the prompt lists `count` among measures, so it may accompany other fields
OUTPUT_FIELDS = RESULT_FIELDS + ("status", "notes")  # policy: every top-level field the prompt allows besides measures
STATUS_VALUES = ("out_of_scope", "unanswerable")  # policy: values the prompt allows for `status`
MEASURE_RE = re.compile(r"[a-z]+(_[a-z0-9]+)*_(m|km|mi|min|h|s|deg|m2|km2)")  # policy: <quantity>_<unit> measure names


# --- generic helpers

def haversine_m(lat1, lon1, lat2, lon2):
    """Great-circle distance in metres."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371000 * math.asin(math.sqrt(a))


def norm(s):
    """Lowercase alphanumerics only, accents folded (NFKD, combining marks dropped)."""
    return "".join(c for c in unicodedata.normalize("NFKD", s if isinstance(s, str) else "").lower() if c.isalnum() and not unicodedata.combining(c))


def name_eq(a, b):
    """Same name ignoring case, punctuation, parenthetical aliases such as "(ARMS)" and trailing comma qualifiers of a (", Purdue University")."""
    strip = lambda s: re.sub(r"\([^)]*\)", "", s) if isinstance(s, str) else s
    nb = norm(strip(b))
    heads = [a.rsplit(",", i)[0] for i in range(a.count(",") + 1)] if isinstance(a, str) else [a]
    return bool(nb) and any(norm(strip(h)) == nb for h in heads)


def circ_diff(a, b):
    """Smallest angle between two bearings (degrees)."""
    d = abs(a - b) % 360
    return min(d, 360 - d)


def fmt_round(x):
    """3 decimals below 1, else 1 decimal (CSV format for seconds and k-tokens)."""
    return round(x, 3) if abs(x) < 1 else round(x, 1)


def fmt_show(x):
    """3 decimals below 1, else 1 decimal (display only)."""
    return f"{x:.3f}" if abs(x) < 1 else f"{x:.1f}"


def secs(ms):
    """Milliseconds to seconds."""
    return ms / 1000 if ms is not None else None


def kilo(n):
    """Count to thousands."""
    return n / 1000 if n is not None else None


def abs_err(pred, gt):
    """Absolute error; None if either value is missing."""
    return abs(pred - gt) if pred is not None and gt is not None else None


def err_stats(pred, gt):
    """(absolute error, percent error); None where either value is missing."""
    err = abs_err(pred, gt)
    return err, 100 * err / gt if err is not None and gt else None


def leaves(x):
    """Yield every non-container value in nested dicts/lists."""
    if isinstance(x, dict):
        x = x.values()
    elif not isinstance(x, list):
        yield x
        return
    for v in x:
        yield from leaves(v)


def numbers(x):
    """Numeric (non-bool) leaves."""
    return (v for v in leaves(x) if isinstance(v, (int, float)) and not isinstance(v, bool))


def grounded_number(v, ctx):
    """Reported number appears (within 1%) in some tool output."""
    if v is None or ctx is None:
        return None
    return any(abs(n - v) <= 0.01 * abs(v) + 1e-9 for n in numbers(ctx["outs"]))


def has_coords(loc):
    """True if loc has numeric lat and lon."""
    return all(isinstance(loc.get(k), (int, float)) and not isinstance(loc.get(k), bool) for k in ("lat", "lon"))


def coord_err_m(gt, loc):
    """Metres from the ground-truth point to a location dict; None without numeric coordinates."""
    return haversine_m(gt["lat"], gt["lon"], loc["lat"], loc["lon"]) if has_coords(loc) else None


# --- tool-output context

def tool_outputs(call):
    """Outputs from recorded MCP subcalls, or a direct tool result."""
    return [m.get('result') for m in call['mcp_calls']] if call.get('mcp_calls') else [call.get('result')]


def build_ctx(result, task, tools=True):
    """calls: tool calls; outs: raw tool results; rows: every returned location (pair members too) with rank/kind/osm_id; tools: False for a no-tool (baseline) run."""
    calls = result.get("tool_calls") or []
    rows, outs = [], []
    for ci, c in enumerate(calls):
        for res in tool_outputs(c):
            outs.append(res)
            locs = res.get("locations") if isinstance(res, dict) else None
            pairs = res.get("pairs") if isinstance(res, dict) and isinstance(res.get("pairs"), list) else []
            locs = (locs if isinstance(locs, list) else []) + [p.get(k) for p in pairs if isinstance(p, dict) for k in ("location_a", "location_b")]
            for rank, loc in enumerate(locs, 1):
                if isinstance(loc, dict):
                    eid = str(loc.get("entity_id", ""))
                    rows.append({"loc": loc, "call": ci, "rank": rank, "n": len(locs),
                                 "kind": eid.split(":")[0], "osm_id": eid.split(":")[-1]})
    return {"calls": calls, "rows": rows, "outs": outs, "task": task, "tools": tools}


def maps_citation_grounding(pred, calls):
    """Match named answer places to Google Maps citations; citations do not verify coordinates or measures."""
    names = []
    for call in calls:
        if call.get("tool") != "google_maps":
            continue
        blocks = call.get("result") or []
        for block in blocks if isinstance(blocks, list) else [blocks]:
            for place in block.get("places", []) if isinstance(block, dict) else []:
                name = re.sub(r"^(?:Review|Photo)s? of ", "", str(place.get("name") or ""))
                name = re.sub(r"\s*- Google Maps$", "", name).strip()
                if name:
                    names.append(name)
    places = answer_items(pred)
    if isinstance(pred, dict) and isinstance(pred.get("pairs"), list):
        places += [pair.get(key) for pair in pred["pairs"] if isinstance(pair, dict)
                   for key in ("location_a", "location_b") if isinstance(pair.get(key), dict)]
    named = [place.get("name") for place in places if isinstance(place.get("name"), str) and place["name"].strip()]
    matched = sum(any(name_eq(name, citation) or name_eq(citation, name) for citation in names) for name in named)
    return {"maps_citation_count": len(names), "maps_citation_matches": matched,
            "maps_citation_grounded": matched == len(named) if named else None}


def answer_loc(pred):
    """Location dict from an answer, {} if absent."""
    loc = pred.get("location", pred) if isinstance(pred, dict) else {}
    return loc if isinstance(loc, dict) else {}


def select_row(loc, rows):
    """Tool row the answer refers to: by entity_id, else name (+ nearest coords), else coords."""
    if loc.get("entity_id"):
        return next((r for r in rows if r["loc"].get("entity_id") == loc["entity_id"]), None)
    cands = [r for r in rows if name_eq(loc.get("name"), r["loc"].get("name"))] if loc.get("name") else rows
    if not has_coords(loc):
        return cands[0] if loc.get("name") and cands else None

    def dist(r):
        """Metres to the answer point; inf without coordinates."""
        d = coord_err_m(loc, r["loc"])
        return math.inf if d is None else d

    best = min(cands, key=dist, default=None)
    return best if best and dist(best) <= GROUND_M else None  # stated coordinates must come from a row


# --- optional re-geocoding (--regeocode)

class ForwardGeocoder:
    """Re-geocodes a claimed name/address (geocode tool, then Mapbox) to check what the model said."""

    def __init__(self, mcp_client=None, mapbox_token=None):
        """Optional MCP client and Mapbox token."""
        self.mcp_client = mcp_client
        self.mapbox_token = mapbox_token
        self._session = None
        self.locations = {}

    async def lookup(self, name, address):
        """Resolved place for a name/address, or None."""
        query = name or address
        if not query:
            return None
        if self.mcp_client is not None:
            try:
                result = await self.mcp_client.call_tool("geocode", {"query": query})
                top = ((result or {}).get("locations") or [{}])[0]
                if top.get("lat") is not None and top.get("lon") is not None:
                    return top
            except Exception:
                pass
        if self.mapbox_token:
            import requests
            self._session = self._session or requests.Session()
            q = ", ".join(x for x in (name, address) if x)  # address disambiguates same-name branches
            try:
                resp = await asyncio.to_thread(self._session.get,
                    f"https://api.mapbox.com/geocoding/v5/mapbox.places/{requests.utils.quote(q)}.json",
                    params={"access_token": self.mapbox_token, "limit": 1}, timeout=10)
                resp.raise_for_status()
                features = resp.json().get("features") or []
                if features:
                    lon, lat = features[0]["geometry"]["coordinates"][:2]
                    return {"name": features[0].get("text"), "address": features[0].get("place_name"),
                            "entity_id": features[0].get("id"), "lat": lat, "lon": lon}
            except Exception:
                pass
        return None

    def geocode(self, name, address):
        return self.locations.get(tuple(value if isinstance(value, str) else None for value in (name, address)))

    async def prepare(self, paths):
        """Resolve reverse-answer claims once before synchronous scoring."""
        self.locations = {}
        for path in paths:
            document = json.loads(Path(path).read_text(encoding="utf-8"))
            for task in document["tasks"]:
                if task["input"].get("operation") != "geocode_reverse" or "output" not in task:
                    continue
                loc = answer_loc(task["output"]["answer"])
                key = tuple(loc.get(key) if isinstance(loc.get(key), str) else None for key in ("name", "address"))
                if key not in self.locations:
                    self.locations[key] = await self.lookup(*key)


# --- per-operation scorers: (pred, gt, ctx, geocoder=None) -> {metric: number|bool|str}

def gt_kind_of(gt):
    """Feature kind (point/line/polygon) from the ground-truth source table."""
    return str(gt.get("source_table", "")).rsplit("_", 1)[-1]


def is_target(entity_id, gt):
    """entity_id ('kind:osm_id') names the ground-truth feature."""
    kind, _, oid = str(entity_id or "").rpartition(":")
    return oid == str(gt["osm_id"]) and kind in ("", gt_kind_of(gt))


def row_metrics(loc, gt, ctx):
    """Selected tool row plus identity and grounding metrics."""
    sel = select_row(loc, ctx["rows"])
    return sel, {
        "name_match": name_eq(loc.get("name"), gt["name"]),
        "osm_id_match": is_target(loc["entity_id"], gt) if loc.get("entity_id") else None,  # answer's own id, if stated
        "row_is_target": bool(sel) and is_target(sel["loc"].get("entity_id"), gt),
        "chosen_rank": sel["rank"] if sel else None,
        "n_candidates": sel["n"] if sel else None,
        "gt_in_tool_output": any(is_target(r["loc"].get("entity_id"), gt) for r in ctx["rows"]),
        "grounded": sel is not None,
    }


def score_place(pred, gt, ctx):
    """Shared geocode / reverse scoring: the answer must name the target place and sit within CORRECT_M of it."""
    loc = answer_loc(pred)
    sel, m = row_metrics(loc, gt, ctx)
    err = coord_err_m(gt, loc)
    named = isinstance(loc.get("name"), str) and bool(loc["name"].strip())
    correct = bool((m["name_match"] or (named and m["osm_id_match"])) and err is not None and err <= CORRECT_M)
    if correct:
        mode = "none"
    elif not named:
        mode = "no_name" if has_coords(loc) else "no_answer"
    elif m["name_match"] or m["osm_id_match"]:
        mode = "wrong_location"
    elif err is not None and err <= GROUND_M:
        mode = "wrong_name"
    elif not ctx.get("tools", True):
        mode = "wrong_place"  # no tool rows to tell an invented place from a real wrong one
    elif sel is None:
        mode = "hallucinated"
    else:
        mode = "wrong_type" if sel["kind"] != gt_kind_of(gt) else "nearby_wrong_feature"
    out = {"address_match": norm(loc["address"]) == norm(gt.get("address")) if loc.get("address") else None,
           "err_m": err, **m, "failure_mode": mode, "correct": correct}
    return loc, out


def score_geocode(pred, gt, ctx, geocoder=None):
    """Geocode: shared place scoring plus distance bands."""
    _, out = score_place(pred, gt, ctx)
    err = out["err_m"]
    return {**out, **{f"within_{t}m": err is not None and err <= t for t in (10, CORRECT_M, 100)}}


def score_geocode_reverse(pred, gt, ctx, geocoder=None):
    """Reverse geocode: shared place scoring, optional re-geocode error."""
    loc, out = score_place(pred, gt, ctx)
    if geocoder is not None:
        resolved = geocoder.geocode(loc.get("name"), loc.get("address"))
        out.update({f"regeocoded_{k}": resolved.get(k) if resolved else None for k in ("name", "address", "entity_id", "lat", "lon")})
        out["err_m_regeocode"] = coord_err_m(gt, resolved) if resolved else None
    return out


def compass_label(bearing):
    """Compass label of a bearing; floor(x/sector+0.5) rounds the half-sector edges up like the tool's Math.round (Python round() is banker's)."""
    return LABELS_8[math.floor(bearing / SECTOR_DEG + 0.5) % len(LABELS_8)]


def nested_dicts(x):
    """Every dict nested below x (x itself excluded)."""
    for v in (x.values() if isinstance(x, dict) else x if isinstance(x, list) else ()):
        if isinstance(v, dict):
            yield v
        yield from nested_dicts(v)


def single_result(pred, value):
    """The answer, or else a dict nested in it that carries the asked value (value(x) not None): a pair, as in the answer key's
    location_a/location_b/measure shape, or a wrapper such as `measure` or `answer`; carriers that disagree leave no single answer."""
    if not isinstance(pred, dict) or value(pred) is not None:
        return pred
    found = [d for d in nested_dicts(pred) if value(d) is not None]
    return found[0] if found and all(value(d) == value(found[0]) for d in found) else pred


def parse_direction(pred):
    """Upper-cased direction string from {"direction": x} (else agreeing nested carriers, see single_result) or a bare string; None when absent or blank."""
    label = lambda p: (p["direction"].strip().upper() or None) if isinstance(p.get("direction"), str) else None
    return label(single_result(pred, label)) if isinstance(pred, dict) else (pred.strip().upper() or None) if isinstance(pred, str) else None


def bearing_records(x):
    """Every dict carrying bearing_deg or direction anywhere in tool output."""
    if isinstance(x, dict):
        if "bearing_deg" in x or "direction" in x:
            yield x
        x = x.values()
    elif not isinstance(x, list):
        return
    for v in x:
        yield from bearing_records(v)


def ref_is_place(ref, place):
    """A PlaceRef (entity_id or lat/lon within CORRECT_M) names the ground-truth place."""
    if not isinstance(ref, dict):
        return False
    return bool((ref.get("entity_id") and is_target(ref["entity_id"], place)) or (has_coords(ref) and coord_err_m(place, ref) <= CORRECT_M))


def endpoint_of(ref, gt):
    """'A' or 'B' when a PlaceRef names that ground-truth place, else None."""
    return next((k for k in ("A", "B") if ref_is_place(ref, gt[f"location_{k.lower()}"])), None)


def endpoint_pairs(ctx, gt, tool):
    """(origin, destination) endpoint letters, None when unresolved, for each call of a two-place tool."""
    return [(endpoint_of((c.get("args") or {}).get("origin"), gt), endpoint_of((c.get("args") or {}).get("destination"), gt))
            for c in ctx["calls"] if c.get("tool") == tool and isinstance(c.get("args"), dict)]


def endpoints_found(ctx, gt):
    """Both ground-truth endpoints appear in returned rows."""
    return all(any(is_target(r["loc"].get("entity_id"), gt[k]) for r in ctx["rows"]) for k in ("location_a", "location_b"))


def score_directionality(pred, gt, ctx, geocoder=None):
    """8-point label match from the final answer; tool calls and outputs only feed diagnostics."""
    label, gt_label = parse_direction(pred), compass_label(gt["bearing_deg"])
    valid = label in LABELS_8
    step = (LABELS_8.index(label) - LABELS_8.index(gt_label)) % len(LABELS_8) if valid else None  # label steps from truth
    err = circ_diff(LABELS_8.index(label) * SECTOR_DEG, gt["bearing_deg"]) if valid else None
    recs = list(bearing_records(ctx["outs"]))
    tool_bearings = [d["bearing_deg"] for d in recs if isinstance(d.get("bearing_deg"), (int, float)) and not isinstance(d["bearing_deg"], bool)]
    tool_labels = {compass_label(b) for b in tool_bearings} | {str(d["direction"]).strip().upper() for d in recs if d.get("direction")}
    pairs = endpoint_pairs(ctx, gt, "bearing")
    resolved = [p for p in pairs if None not in p]
    exact = valid and step == 0
    adjacent = valid and min(step, len(LABELS_8) - step) <= 1
    if label is None:
        mode = "no_answer"
    elif not valid:
        mode = "invalid_label"
    elif exact:
        mode = "none"
    elif step == len(LABELS_8) // 2:
        mode = "reversed"
    elif adjacent:
        mode = "adjacent_off"
    else:
        mode = "wrong"
    return {
        "pred_label": label, "gt_label": gt_label, "exact": exact, "adjacent": adjacent, "angular_error_deg": err,
        "endpoints_found": endpoints_found(ctx, gt),
        "endpoints_used": {"A", "B"} <= {e for p in pairs for e in p},
        "args_order_ok": resolved[-1] == ("B", "A") if resolved else None,  # origin=B, target=A in the last resolved bearing call
        "grounded": valid and label in tool_labels,
        "gt_in_tool_output": gt_label in {compass_label(b) for b in tool_bearings},
        "failure_mode": mode, "correct": exact,
    }


def route_distance_m(pred, key="distance"):
    """Answer distance in metres from its <key>_m / _km / _mi entry (first found), else None."""
    for u, f in UNIT_M.items():
        v = pred.get(f"{key}_{u}") if isinstance(pred, dict) else None
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return v * f
    return None


def route_close(a, b):
    """Distances agree within ROUTE_REL_TOL, floored at ROUTE_ABS_TOL_M."""
    return abs(a - b) <= max(ROUTE_REL_TOL * abs(b), ROUTE_ABS_TOL_M)


def score_route(pred, gt, ctx, geocoder=None):
    """Route distance from the final answer; route calls and outputs only feed diagnostics."""
    pred = single_result(pred, route_distance_m)
    pred_m, gt_m = route_distance_m(pred), gt.get("distance_m")
    err, err_pct = err_stats(pred_m, gt_m)
    correct = (pred_m is None and isinstance(pred, dict) and pred.get("status") == "unanswerable") if gt_m is None \
        else pred_m is not None and route_close(pred_m, gt_m)
    calls = [(c["args"], next(iter(tool_outputs(c)), None))
             for c in ctx["calls"] if c.get("tool") == "route" and isinstance(c.get("args"), dict)]
    pairs = endpoint_pairs(ctx, gt, "route")  # aligned with calls
    resolved = [p for p in pairs if None not in p]
    ok = [i for i, (_, r) in enumerate(calls) if isinstance(r, dict) and isinstance(r.get("distance_m"), (int, float))]
    used = (ok or range(len(calls)))[-1] if calls else None  # last successful call, else last call
    pair = pairs[used] if calls else None
    mode_ok = (calls[used][0].get("profile") or "car") == gt["mode"] if calls else None  # tool default is car
    out = calls[used][1] if ok else {}
    outs = [calls[i][1] for i in ok]
    pred_t = travel_min(pred)
    t_err, t_pct = err_stats(pred_t, travel_min(gt))
    if correct:
        mode = "none"
    elif pred_m is None:
        mode = "no_answer"
    elif gt_m is None:
        mode = "wrong"
    elif any(route_close(pred_m * f, gt_m) for f in (1000, 1 / 1000, 1609.344, 1 / 1609.344)):
        mode = "unit_error"
    elif pair == ("B", "A"):
        mode = "reversed_endpoints"
    elif pair is not None and pair != ("A", "B"):
        mode = "wrong_endpoint"
    elif mode_ok is False:
        mode = "wrong_mode"
    else:
        mode = "off_tolerance" if err_pct <= 100 * NEAR_REL_TOL else "wrong"
    return {
        "pred_distance_m": pred_m, "gt_distance_m": gt_m, "dist_err_m": err, "dist_err_pct": err_pct,
        "time_err_min": t_err, "time_err_pct": t_pct,
        "mode_ok": mode_ok,
        "endpoints_found": endpoints_found(ctx, gt),
        "endpoints_used": {"A", "B"} <= {e for p in pairs for e in p},
        "args_order_ok": resolved[-1] == ("A", "B") if resolved else None,  # origin=A, destination=B in the last resolved route call
        "tool_distance_m": out.get("distance_m"), "tool_time_min": out.get("duration_min"),
        "grounded": pred_m is not None and any(route_close(pred_m, r["distance_m"]) for r in outs) if gt_m is not None else None,
        "gt_in_tool_output": any(route_close(r["distance_m"], gt_m) for r in outs) if gt_m is not None else None,
        "failure_mode": mode, "correct": correct,
    }


def answer_items(pred):
    """Answered places as dicts: the `locations` list, else a single `location`."""
    if not isinstance(pred, dict):
        return []
    items = pred["locations"] if isinstance(pred.get("locations"), list) else [pred.get("location")]
    return [x for x in items if isinstance(x, dict)]


def has_result(pred, key):
    """An explicit result list distinguishes an empty answer from no answer."""
    if not isinstance(pred, dict):
        return False
    value = pred.get(key)
    return (isinstance(value, list) and all(isinstance(item, dict) for item in value)
            or key == "locations" and isinstance(pred.get("location"), dict))


def same_place(item, gt):
    """Answered item names the ground-truth place (name, or entity_id plus a name) and any stated coordinates sit within CORRECT_M.
    A nameless ground-truth place matches by entity_id, or an unnamed item within NAMELESS_M."""
    named = isinstance(item.get("name"), str) and bool(item["name"].strip())
    err = coord_err_m(gt, item)
    nameless = not (isinstance(gt.get("name"), str) and gt["name"].strip())
    ident = (name_eq(item.get("name"), gt.get("name")) or bool((named or nameless) and item.get("entity_id") and is_target(item["entity_id"], gt))
             or (nameless and not named and err is not None and err <= NAMELESS_M))
    stated = item.get("lat") is not None or item.get("lon") is not None
    return ident and (not stated or (err is not None and err <= CORRECT_M))


def match_places(items, gts, same=None):
    """(match, n_dup): match[i] is the ground-truth index answered item i identifies under same (default same_place; one-to-one, closest coordinates first), else None; n_dup counts repeats."""
    same = same or same_place
    match, taken, n_dup = [], set(), 0
    for it in items:
        cands = [j for j, g in enumerate(gts) if same(it, g)]
        j = min((j for j in cands if j not in taken), key=lambda j: coord_err_m(gts[j], it) or 0, default=None)
        taken |= {j} - {None}
        n_dup += bool(cands) and j is None
        match.append(j)
    return match, n_dup


def precision_mode(m, core=None):
    """Tasks with answer scoring "precision": accuracy is precision over the accepted list, correct means at least one item and every answered item accepted.
    recall and f1 are diagnostics; with core (the score against the non-buffer items only) recall counts core items found.
    A repeat of an already matched place (n_duplicates) is dropped, not an extra."""
    n_dup = m["n_duplicates"] if m["precision"] else 0  # precision 0: unanswered, or no matched place to repeat
    m = {**m, "n_extra": m["n_extra"] - n_dup, "precision": m["n_correct_items"] / (m["n_answered"] - n_dup) if n_dup else m["precision"]}
    ok = m["n_answered"] > 0 and m["n_extra"] == 0
    p, r = m["precision"], (core or m)["recall"]
    m = {**m, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0}
    return {**m, "correct": ok, "failure_mode": "none" if ok else "no_answer" if not m["n_answered"] else "wrong" if not m["n_correct_items"] else "extra_items"}


def prf(correct, answered, total):
    """Set precision, recall, F1; an empty side scores 0, both empty scores 1."""
    if not answered and not total:
        return 1.0, 1.0, 1.0
    p, r = correct / answered if answered else 0.0, correct / total if total else 0.0
    return p, r, 2 * p * r / (p + r) if p + r else 0.0


def tie_ranks(gts):
    """Ground-truth rank per place; equal distance_m share the first rank, so tied places are interchangeable."""
    d = [g.get("distance_m") for g in gts]
    return [d.index(x) if x is not None else i for i, x in enumerate(d)]


def kendall_tau(seq):
    """Kendall tau of ranks listed in answer order (1 = same order); tied pairs skipped, None without a pair."""
    pairs = [(a, b) for i, a in enumerate(seq) for b in seq[i + 1:] if a != b]
    return sum((a < b) - (a > b) for a, b in pairs) / len(pairs) if pairs else None


def filter_covers(include, want):
    """Call include (keys AND, value lists OR, * wildcards, case ignored) has exactly the ground-truth keys and allows every ground-truth value."""
    vals = lambda v: [str(x).lower() for x in (v if isinstance(v, list) else [v])]
    return isinstance(include, dict) and set(include) == set(want) and all(
        all(any(fnmatchcase(w, p) for p in vals(include[k])) for w in vals(v)) for k, v in want.items())


def knn_modes(m):
    """First failing cause of a wrong KNN answer, most upstream first; call-level checks apply only when a knn call exists."""
    if not m["n_answered"]:
        return "no_answer"
    for key, mode in (("reference_ok", "wrong_reference"), ("distance_type_ok", "wrong_distance_type"), ("profile_ok", "wrong_distance_type"), ("filter_ok", "wrong_filter")):
        if m[key] is False:
            return mode
    if m["limit"] is not None and m["limit"] < m["n_gt"]:
        return "wrong_k"
    if m["ungrounded_count"]:
        return "hallucinated"
    return "wrong" if m["n_missing"] and m["n_extra"] else "missing_items" if m["n_missing"] else "extra_items"


def score_knn(pred, gt, ctx, geocoder=None):
    """KNN from the final answer: the answered set (order and repeats ignored) must be exactly the ground-truth k nearest; order, knn calls and outputs only feed diagnostics."""
    gts, items = gt["locations"], answer_items(pred)
    match, n_dup = match_places(items, gts)
    sets = set_metrics(match, n_dup, len(items), len(gts), answered=has_result(pred, "locations"))
    hits = [j for j in match if j is not None]
    rank = tie_ranks(gts)
    n_ans, n_gt, n_ok = sets["n_answered"], sets["n_gt"], sets["n_correct_items"]
    net = gt["distance_type"] == "network"
    key = ("distance", "geodesic_distance") if net else ("geodesic_distance", "distance")  # stated measure, preferred first
    stated = [(it, gts[j]) for it, j in zip(items, match) if j is not None]
    errs = [(abs(d - g["distance_m"]), g["distance_m"], d) for it, g in stated
            if (d := next((x for x in (route_distance_m(it, k) for k in key) if x is not None), None)) is not None]
    calls = [(c["tool"], c["args"], next(iter(tool_outputs(c)), None))
             for c in ctx["calls"] if c.get("tool") in KNN_TOOLS.values() and isinstance(c.get("args"), dict)]
    done = [i for i, (_, _, res) in enumerate(calls) if isinstance(res, dict) and not res.get("error") and "locations" in res]
    tool, args, _ = calls[(done or range(len(calls)))[-1]] if calls else (None, {}, None)  # last successful call, else last call
    limit = (args.get("limit") or KNN_DEFAULT_LIMIT) if calls else None
    in_tool = [any(is_target(row["loc"].get("entity_id"), g) for row in ctx["rows"]) for g in gts]
    ungrounded = ungrounded_count(items, ctx)
    m = {
        **{k: v for k, v in sets.items() if k not in ("correct", "failure_mode")}, **by_name(items, gts, same_place, answered=has_result(pred, "locations")),
        "rank_match": sum(j is not None and i < n_gt and rank[j] == rank[i] for i, j in enumerate(match)) / n_gt if n_gt else None,
        "exact_order": has_result(pred, "locations") and n_ok == n_ans == n_gt and [rank[j] for j in hits] == rank,
        "kendall_tau": kendall_tau([rank[j] for j in hits]),
        "top1_correct": bool(match) and match[0] is not None and rank[match[0]] == rank[0] if n_gt else None,
        "n_dist": len(errs), "dist_mae_m": st.mean(e for e, _, _ in errs) if errs else None,
        "dist_mre_pct": st.mean(100 * e / g for e, g, _ in errs if g) if errs else None,
        "dist_ok": all(route_close(d, g) for _, g, d in errs) if errs and net else None,  # euclidean ground truth is point to point, the tool's shape to shape
        "reference_ok": ref_is_place(args.get("reference"), gt["reference"]) if calls else None,
        "distance_type_ok": tool == KNN_TOOLS[gt["distance_type"]] if calls else None,
        "profile_ok": (tool == KNN_TOOLS["network"] and (args.get("profile") or "car") == gt["profile"]) if calls and net else None,  # tool default is car
        "k_ok": limit == n_gt if calls else None,
        "filter_ok": filter_covers(args.get("include"), gt["include"]) if calls and isinstance(args.get("include"), dict) else None,  # string filter grammars: unknown
        "n_knn_calls": len(calls),
        "tool_set_recall": sum(in_tool) / n_gt if n_gt else None, "gt_in_tool_output": all(in_tool),
        "n_dropped": sum(t and j not in hits for j, t in enumerate(in_tool)),  # in tool output but not answered
        "ungrounded_count": ungrounded, "grounded": bool(n_ans) and not ungrounded if ungrounded is not None else None,
        "limit": limit,
    }
    mode = "none" if sets["correct"] else knn_modes(m)
    return {**{k: v for k, v in m.items() if k != "limit"}, "failure_mode": mode, "correct": sets["correct"]}


def score_set(items, gts, same, answered=True):
    """Set metrics of answered items against ground truth, matched one-to-one under same(item, gt)."""
    return set_metrics(*match_places(items, gts, same), len(items), len(gts), answered=answered)


def set_metrics(match, n_dup, n_items, n_gt, answered=True):
    """Set metrics from match_places output; correct means equal sets, order ignored, no missing, extra or repeated item (a repeat is an extra, also counted in n_duplicates)."""
    n_ok, n_ans = sum(j is not None for j in match), n_items
    p, r, f1 = prf(n_ok, n_ans, n_gt) if answered else (0.0, 0.0, 0.0)
    correct = answered and n_ok == n_ans == n_gt
    mode = "none" if correct else "no_answer" if not n_items else "wrong" if n_ok < n_gt and n_ok < n_ans else "missing_items" if n_ok < n_gt else "extra_items"
    return {"n_gt": n_gt, "n_answered": n_ans, "n_correct_items": n_ok, "n_missing": n_gt - n_ok, "n_extra": n_ans - n_ok, "n_duplicates": n_dup,
            "precision": p, "recall": r, "f1": f1, "jaccard": n_ok / (n_ans + n_gt - n_ok) if n_ans + n_gt - n_ok else float(answered),
            "failure_mode": mode, "correct": correct}


def same_pair(it, gt):
    """Answered pair names both ground-truth places; a ground-truth pair without a partner (left join) needs none."""
    a, b = it.get("location_a"), it.get("location_b")
    return isinstance(a, dict) and same_place(a, gt["location_a"]) and (isinstance(b, dict) and same_place(b, gt["location_b"]) if gt.get("location_b") else not b)


def same_pair_unordered(it, gt):
    """same_pair in either order: (u, v) and (v, u) name the same two places."""
    return same_pair(it, gt) or same_pair({"location_a": it.get("location_b"), "location_b": it.get("location_a")}, gt)


PLACE_KEYS = {"name", "lat", "lon", "address", "entity_id", "osm_id"}  # a place row's own fields, never a nested partner


def is_place(x):
    """Dict that describes a place (a name or coordinates)."""
    return isinstance(x, dict) and (bool(x.get("name")) or has_coords(x))


def answer_rows(pred):
    """Rows of a free-form join answer: `pairs`, else the `locations` or `answer` list."""
    if not isinstance(pred, dict):
        return []
    rows = next((pred[k] for k in ("pairs", "locations", "answer") if isinstance(pred.get(k), list)), [])
    return [r for r in rows if isinstance(r, dict)]


def row_places(row):
    """(own place or None, nested places in key order) of one answer row."""
    nested = [y for k, v in row.items() if k not in PLACE_KEYS for y in (v if isinstance(v, list) else [v]) if is_place(y)]
    return (row if is_place(row) and "location_a" not in row else None), nested


def any_shape_pairs(pred):
    """Pairs from any answer shape: location_a/location_b rows, or a place row with nested partner places (none = no partner)."""
    out = []
    for row in answer_rows(pred):
        if "location_a" in row:
            out.append(row)
            continue
        own, nested = row_places(row)
        a, bs = (own, nested) if own else (nested[0], nested[1:]) if nested else (None, [])
        out += [{"location_a": a, "location_b": b} for b in bs or [None]] if a else []
    return out


def any_shape_places(pred):
    """Places from any answer shape: the row's own place, else a pair's location_a or the row's first nested place; repeats dropped."""
    out = []
    for row in answer_rows(pred):
        own, nested = row_places(row)
        p = row.get("location_a") if "location_a" in row else own or (nested[0] if nested else None)
        if is_place(p) and not any(name_eq(p.get("name"), q.get("name")) for q in out):
            out.append(p)
    return out


def ungrounded_count(places, ctx):
    """Answered places that are not a returned row; None without tools or context."""
    return sum(select_row(p, ctx["rows"]) is None for p in places) if ctx and ctx.get("tools", True) else None


def grounding(places, ctx):
    """grounded (every answered place is a returned row) and ungrounded_count; None without tools."""
    n = ungrounded_count(places, ctx)
    return {"ungrounded_count": n, "grounded": bool(places) and not n if n is not None else None}


def score_range(pred, gt, ctx=None, geocoder=None):
    """Range query from the final answer: the answered places must be exactly the ground-truth places in the region."""
    items = answer_items(pred)
    return {**score_set(items, gt["locations"], same_place, answered=has_result(pred, "locations")), **by_name(items, gt["locations"], same_place, answered=has_result(pred, "locations")), **grounding(items, ctx)}


def score_join(pred, gt, ctx=None, geocoder=None):
    """Spatial join from the final answer: `pairs` (inner, left) or `locations` (semi, anti) must equal the ground-truth set.
    Inner pairs match unordered, left pairs role-ordered. correct_strict: every pair role-ordered (self joins stay unordered).
    correct_any_shape/f1_any_shape: any answer shape (any_shape_pairs, any_shape_places), e.g. a semi join answered as pairs.
    shape_recovered: the expected list is absent, so the other shape is scored when it carries the asked rows (a semi join's paired left places,
    an anti join's partnerless places, a pair join's places with nested partners)."""
    if "pairs" not in gt:
        answered, alt = has_result(pred, "locations"), any_shape_pairs(pred)
        recovered = not answered and not (isinstance(pred, dict) and "locations" in pred) and bool(alt) and (gt.get("join_type") == "semi" or not any(p.get("location_b") for p in alt))
        items = any_shape_places(pred) if recovered else answer_items(pred)
        m = score_set(items, gt["locations"], same_place, answered=answered or recovered)
        return {**m, "correct_strict": m["correct"] and not recovered, "shape_recovered": recovered,
                **any_shape(score_set(any_shape_places(pred), gt["locations"], same_place, answered=has_result(pred, "locations") or has_result(pred, "pairs"))),
                **by_name(items, gt["locations"], same_place, answered=answered or recovered), **grounding(items, ctx)}
    self_join = gt.get("set_a") is not None and gt.get("set_a") == gt.get("set_b")
    same = same_pair_unordered if self_join or gt.get("join_type") == "inner" else same_pair
    pairs = pred.get("pairs") if isinstance(pred, dict) else None
    pairs = [x for x in pairs if isinstance(x, dict)] if isinstance(pairs, list) else []
    strict = score_set(pairs, gt["pairs"], same_pair_unordered if self_join else same_pair, answered=has_result(pred, "pairs"))
    answered, alt = has_result(pred, "pairs"), any_shape_pairs(pred)
    recovered = not answered and not (isinstance(pred, dict) and "pairs" in pred) and any(p.get("location_b") for p in alt)
    pairs = alt if recovered else pairs
    places = [x.get(k) for x in pairs for k in ("location_a", "location_b") if isinstance(x.get(k), dict)]
    return {**score_set(pairs, gt["pairs"], same, answered=answered or recovered), "correct_strict": strict["correct"], "shape_recovered": recovered,
            **any_shape(score_set(any_shape_pairs(pred), gt["pairs"], same, answered=has_result(pred, "locations") or has_result(pred, "pairs"))),
            **by_name(pairs, gt["pairs"], same, answered=answered or recovered), **grounding(places, ctx)}


def any_shape(m):
    """correct and f1 of an any-shape score_set result, renamed."""
    return {"correct_any_shape": m["correct"], "f1_any_shape": m["f1"]}


def no_xy(x):
    """Copy of answered places (or pairs) without lat/lon, so places match by identity alone."""
    if isinstance(x, list):
        return [no_xy(v) for v in x]
    return {k: no_xy(v) for k, v in x.items() if k not in ("lat", "lon")} if isinstance(x, dict) else x


def by_name(items, gts, same, answered=True):
    """correct_name, f1_name: the set score with stated coordinates ignored (which places, not where); diagnostic only."""
    m = score_set(no_xy(items), gts, same, answered=answered)
    return {"correct_name": m["correct"], "f1_name": m["f1"]}


def travel_min(d):
    """Travel time of an answer or an answer key under either name; None when absent."""
    return next((d[k] for k in TIME_KEYS if isinstance(d, dict) and d.get(k) is not None), None)


def score_route_time(pred, gt, ctx=None, geocoder=None):
    """Route time error against ground truth."""
    loc = pred if isinstance(pred, dict) else {}
    pred_t, gt_t = travel_min(loc), travel_min(gt)
    err, pct_err = err_stats(pred_t, gt_t)
    return {
        "grounded": grounded_number(pred_t, ctx),
        "time_err_min": err,
        "time_err_pct": pct_err,
        "correct": pct_err is not None and pct_err <= 100 * REL_TOL,
    }


def result_fields(value):
    """Top-level result fields of an answer; every measure key counts as "<measure>"."""
    return {"<measure>" if k in MEASURE_FIELDS or k not in RESULT_FIELDS else k for k in value if k in RESULT_FIELDS or MEASURE_RE.fullmatch(k)}


def answer_format(pred, key):
    """Output-format conformance against the result fields the answer key implies (its named fields, else a measure), separate from correctness."""
    want = result_fields(key) & set(RESULT_FIELDS) or result_fields(key)  # named fields win over measure-shaped inputs (radius_m, k)
    if not isinstance(pred, dict):
        return {"format_ok": False, "answer_fields": "", "format_mode": "not_an_object"}
    got = result_fields(pred)
    used = got - {"location"} | {"locations"} if "locations" in want and len(key["locations"]) == 1 and "location" in got and "locations" not in got else got  # k = 1: a single place is a one-item set
    as_list = lambda k: pred[k] if isinstance(pred.get(k), list) else []
    pairs = as_list("pairs")
    places = ([pred["location"]] if "location" in pred else []) + as_list("locations") + [x.get("location_a") if isinstance(x, dict) else x for x in pairs] + [
        x["location_b"] for x in pairs if isinstance(x, dict) and x.get("location_b") is not None]
    unnamed = [k for k, v in pred.items() if isinstance(v, (int, float)) and not isinstance(v, bool) and k not in RESULT_FIELDS and not MEASURE_RE.fullmatch(k)]
    if not got:
        mode = "bad_measure_name" if unnamed else "no_result_field"
    elif not want <= used:
        mode = "wrong_field"
    elif used - want - {"<measure>"}:
        mode = "extra_fields"
    elif any(not (isinstance(p, dict) and has_coords(p)) for p in places):  # a left join's missing partner may be null
        mode = "bad_location"
    elif "direction" in pred and parse_direction(pred) not in LABELS_8:
        mode = "bad_direction"
    elif unnamed:
        mode = "bad_measure_name"
    else:
        mode = "ok"
    return {"format_ok": mode == "ok", "answer_fields": ";".join(sorted(got)), "format_mode": mode}


def valid_place(p, nullable=False):
    """A `location` as the prompt defines it: numeric in-range lat/lon, name and address a string or null when present."""
    if p is None:
        return nullable
    ok = isinstance(p, dict) and has_coords(p) and abs(p["lat"]) <= 90 and abs(p["lon"]) <= 180
    return ok and all(p.get(k) is None or isinstance(p[k], str) for k in ("name", "address"))


def answer_schema(pred):
    """Task-independent well-formedness against the prompt's output contract: a JSON object of allowed fields with well-typed values; location and locations may coexist."""
    def mode():
        if not isinstance(pred, dict):
            return "not_an_object"
        if not pred:
            return "empty_object"
        measures = [k for k in pred if k not in OUTPUT_FIELDS and MEASURE_RE.fullmatch(k)]
        if set(pred) - set(OUTPUT_FIELDS) - set(measures):
            return "unknown_field"
        if not any(k in pred for k in RESULT_FIELDS + ("status",)) and not measures:
            return "no_result_field"
        if "location" in pred and not valid_place(pred["location"]):
            return "bad_location"
        if "locations" in pred and not (isinstance(pred["locations"], list) and all(valid_place(p) for p in pred["locations"])):
            return "bad_locations"
        if "pairs" in pred and not (isinstance(pred["pairs"], list) and all(
                isinstance(x, dict) and valid_place(x.get("location_a")) and "location_b" in x and valid_place(x["location_b"], nullable=True) for x in pred["pairs"])):
            return "bad_pairs"
        if "direction" in pred and pred["direction"] not in LABELS_8:
            return "bad_direction"
        if "status" in pred and pred["status"] not in STATUS_VALUES:
            return "bad_status"
        if "count" in pred and not (isinstance(pred["count"], int) and not isinstance(pred["count"], bool)):
            return "bad_count"
        if any(not (isinstance(pred[k], (int, float)) and not isinstance(pred[k], bool) and math.isfinite(pred[k])) for k in measures):
            return "bad_measure"
        if "notes" in pred and not isinstance(pred["notes"], str):
            return "bad_notes"
        return "ok"
    m = mode()
    return {"schema_ok": m == "ok", "schema_mode": m}


def quality_metrics(operation, pred, row, key):
    """Single source of truth for the figure's query accuracy and disjoint outcome class.
    invalid_answer means unrecoverable (no object, or no usable required result); json_valid, schema_ok and format_ok stay conformance diagnostics."""
    valid = isinstance(pred, dict)
    if operation == "geocode":
        err, loc = row.get("err_m"), answer_loc(pred)
        valid = valid and err is not None and abs(loc["lat"]) <= 90 and abs(loc["lon"]) <= 180
        strict, partial = valid and err <= 50, valid and 50 < err <= 200
        score = 100.0 * strict
    elif operation == "geocode_reverse":
        loc, err = answer_loc(pred), row.get("err_m_regeocode")
        identity = any(isinstance(loc.get(k), str) and loc[k].strip() for k in ("name", "address")) or bool(loc.get("entity_id"))
        valid = valid and identity
        strict = valid and (bool(row.get("name_match") or row.get("osm_id_match")) or err is not None and err <= 50)
        partial = valid and not strict and err is not None and err <= 200
        score = 100.0 * strict
    elif operation == "directionality":
        valid = valid and row.get("pred_label") in LABELS_8
        strict = valid and bool(row.get("exact"))
        partial = valid and not strict and bool(row.get("adjacent"))
        score = 100.0 * strict
    elif operation in ("route", "shortest_path"):
        pct, err = row.get("dist_err_pct"), row.get("dist_err_m")
        if "distance_m" in key and key["distance_m"] is None:  # unreachable truth: only `status: unanswerable` is right, a distance is wrong
            valid = valid and (row.get("pred_distance_m") is not None or bool(row.get("correct")))
            strict, partial = valid and bool(row.get("correct")), False
        else:
            valid = valid and pct is not None and err is not None and math.isfinite(err)
            strict = valid and (pct <= 1 or err <= 1)
            partial = valid and not strict and pct <= 10
        score = 100.0 * strict
    elif operation in ("knn", "range_query", "spatial_join"):
        f1 = row.get("precision") if key.get("scoring") == "precision" else row.get("f1")  # precision tasks: accepted list, no recall
        want = "pairs" if "pairs" in key else "locations"
        rows = pred.get(want) if valid else None
        lone = valid and want == "locations" and isinstance(pred.get("location"), dict) and not pred.get("status")  # k = 1 single place; beside a status it is a decline
        valid = valid and f1 is not None and (isinstance(rows, list) and all(isinstance(x, dict) for x in rows) or lone or row.get("shape_recovered") is True)  # empty list is a valid answer
        strict = valid and f1 >= 1
        partial = valid and 0 < f1 < 1
        score = 100.0 * f1 if valid else 0.0
    else:
        valid, strict, partial = valid and bool(row.get("answered")), bool(row.get("correct")), False
        score = 100.0 * (valid and strict)
    outcome = "invalid_answer" if not valid else "correct" if strict else "partially_correct" if partial else "incorrect"
    return {"answer_valid": bool(valid), "accuracy_score": score, "outcome": outcome, "correct": outcome == "correct"}


SCORERS = {
    "geocode": score_geocode,
    "geocode_reverse": score_geocode_reverse,
    "directionality": score_directionality,
    "route": score_route,
    "knn": score_knn,
    "range_query": score_range,
    "spatial_join": score_join,
    "route_time": score_route_time,
    "shortest_path": score_route,
    "shortest_path_time": score_route_time,
}

COMMON = ["task_id", "operation", "run_id", "side", "model", "reasoning", "correct", "accuracy_score", "outcome", "answer_valid",
          "answered", "parsed", "json_valid", "status", "grounded", "correct_grounded",
          "turns", "tool_calls", "latency_s", "total_tool_s", "llm_s", "cost_usd",
          "prompt_tokens_k", "completion_tokens_k", "cached_tokens_k", "prompt_tokens", "completion_tokens", "cached_tokens",
          "tool_names", "tool_latencies_s", "n_tool_errors", "n_redundant_calls", "n_clean_calls", "zero_hit"]
# tool diagnostics: not applicable (blank, never 0 or False) on the no-tool side
TOOL_COLS = {"grounded", "correct_grounded", "tool_calls", "total_tool_s", "tool_names", "tool_latencies_s", "n_tool_errors", "n_redundant_calls", "n_clean_calls",
             "gt_in_tool_output", "row_is_target", "chosen_rank", "n_candidates", "endpoints_found", "endpoints_used", "args_order_ok", "mode_ok",
             "tool_distance_m", "tool_time_min", "reference_ok", "distance_type_ok", "profile_ok", "k_ok", "filter_ok", "n_knn_calls",
             "tool_set_recall", "n_dropped", "ungrounded_count"}


# --- loading and per-task metrics

def load_tasks(path):
    """Task file to {task_id: task}."""
    with open(path) as f:
        raw = json.load(f)
    return {t["task_id"]: t for t in (raw["tasks"] if isinstance(raw, dict) and "tasks" in raw else raw)}


def run_id_of(path):
    """Run directory name for a results path."""
    d = os.path.dirname(os.path.abspath(path))
    return os.path.basename(os.path.dirname(d)) if os.path.basename(d) in {"baseline", "mcp"} else os.path.basename(d)


def call_failed(c):
    """True if a tool call errored or timed out."""
    return bool(c.get("error") or c.get("timeout")
                or any(isinstance(res, dict) and res.get("error") for res in tool_outputs(c)))


def side_of(r, side="auto"):
    """'baseline' (no tools) or 'mcp': the run's recorded condition, else whether it logged tool calls."""
    condition = r.get("condition") or ("mcp" if "tool_calls" in r else "baseline")
    return side if side != "auto" else "mcp" if condition == "gemini_maps" else condition


def common_metrics(r, trace):
    """Operation-independent efficiency and cost metrics."""
    calls = r.get("tool_calls") or []
    keys = [(c["tool"], json.dumps(c.get("args"), sort_keys=True)) for c in calls]
    n_failed, n_repeat, ok = 0, 0, set()  # each call is in exactly one class: failed, repeat (succeeded, identical to an earlier successful call), clean
    for c, key in zip(calls, keys):
        if call_failed(c):
            n_failed += 1
        elif key in ok:
            n_repeat += 1
        else:
            ok.add(key)
    tool_ms = sum(c.get("latency_ms") or 0 for c in calls)
    llm_ms = (r.get("latency_breakdown_ms") or {}).get("llm_call_latency_ms")  # includes SDK retries; excludes tool time and orchestration
    value = r.get("value")
    return {
        "model": r.get("model"), "reasoning": trace.get("reasoning"),
        "answered": value is not None and not (isinstance(value, dict) and value.get("status")),
        "parsed": isinstance(value, dict), "status": value.get("status") if isinstance(value, dict) and isinstance(value.get("status"), str) else None,
        "turns": trace.get("turn_count"), "tool_calls": len(calls),
        "latency_s": secs(r["latency_ms"]), "total_tool_s": secs(tool_ms), "llm_s": secs(llm_ms if llm_ms is not None else r["latency_ms"] - tool_ms if r["latency_ms"] is not None else None),
        "cost_usd": r["cost_usd"],
        "prompt_tokens_k": kilo(trace.get("prompt_tokens")), "completion_tokens_k": kilo(trace.get("completion_tokens")),
        "cached_tokens_k": kilo(trace.get("cached_tokens")),
        **{k: trace.get(k) for k in ("prompt_tokens", "completion_tokens", "cached_tokens")},
        "tool_names": ";".join(t for t, _ in keys),
        "tool_latencies_s": ";".join(str(round(secs(c.get("latency_ms") or 0), 3)) for c in calls),
        "n_tool_errors": n_failed, "n_redundant_calls": n_repeat, "n_clean_calls": len(calls) - n_failed - n_repeat,
        "zero_hit": r["zero_hit"],
        "json_valid": trace.get("json_valid"),
    }


def recover_answer(text):
    """Answer object from a reply the runner could not parse: the first JSON object when it is complete (trailing text allowed),
    unless a later object gives a different result; a broken first object is never repaired from fragments."""
    dec, objs, i = json.JSONDecoder(), [], 0
    text = text if isinstance(text, str) else ""
    while (j := text.find("{", i)) >= 0:
        try:
            obj, i = dec.raw_decode(text, j)
            objs.append(obj)
        except ValueError:
            if not objs:
                return None
            i = j + 1
    result = lambda o: {k: v for k, v in o.items() if k in RESULT_FIELDS or MEASURE_RE.fullmatch(k)}
    first = objs[0] if objs and isinstance(objs[0], dict) and result(objs[0]) else None
    return first if first is not None and all(result(o) in ({}, result(first)) for o in objs if isinstance(o, dict)) else None


def final_content(trace):
    """Text of the last assistant message, or None."""
    return next((m.get("content") for m in reversed(trace.get("transcript") or []) if m.get("role") == "assistant"), None)


def evaluate(tasks_path, results_paths, geocoder=None, side="auto"):
    """Score run interactions against the benchmark task sources."""
    overrides = load_tasks(tasks_path) if tasks_path else None
    by_operation = {}
    for rp in results_paths:
        document = json.loads(Path(rp).read_text(encoding='utf-8'))
        tasks = overrides if overrides is not None else {
            task_id: task for path in document["config"]["tasks"] for task_id, task in load_tasks(path).items()}
        for entry in document['tasks']:
            if "output" not in entry:
                continue
            output = entry["output"]
            metrics = output["metrics"]
            task = tasks[entry["task_id"]]
            answer = task.get("answer")
            if answer is None:
                continue  # Unlabelled tasks can run but cannot be scored.
            operation = task["operation"]
            # Recover grounding records from the transcript rather than a duplicate tool log.
            calls = []
            transcript = entry["trace"]["transcript"]
            for event in entry["trace"]["events"]:
                if event["type"] == "llm_call":
                    steps = (event.get("response") or {}).get("steps") or []
                    results = {s.get("call_id"): s for s in steps if s.get("call_id")}
                    calls.extend({"tool": "google_maps", "args": step.get("arguments") or {},
                                  "result": results.get(step.get("id"), {}).get("result"), "latency_ms": 0}
                                 for step in steps if step.get("type") == "google_maps_call")
                    continue
                if event["type"] != "tool_call":
                    continue
                assistant = transcript[event["request_transcript_index"]]
                call = next(call for call in assistant["tool_calls"] if call["id"] == event["call_id"])
                calls.append({"tool": call["function"]["name"],
                              "args": event["arguments"] if "arguments" in event else json.loads(call["function"]["arguments"] or "{}"),
                              "result": json.loads(transcript[event["result_transcript_index"]]["content"]),
                              "latency_ms": event["latency_ms"], "error": event.get("error"),
                              "timeout": event.get("timeout", False)})
            model = document["config"]["model"]
            if "/" not in model and document["config"].get("provider") == "gemini":
                model = f"google/{model}"
            r = {"task_id": entry["task_id"], "condition": document["config"]["condition"],
                 "model": model, "value": output["answer"] if output["answer"] is not None else recover_answer(final_content(entry["trace"])),
                 "tool_calls": calls, **metrics}
            trace = {**metrics, "reasoning": document["config"].get("reasoning")}
            scorer = SCORERS.get(operation)
            if scorer is None:
                sys.exit(f"no scorer registered for operation {operation!r} - add one to SCORERS in evaluate.py")
            s = side_of(r, side)
            gt, ctx = answer.get("location", answer), build_ctx(r, task, tools=s != "baseline")
            score = scorer(r["value"], gt, ctx, geocoder=geocoder)
            if answer.get("scoring") == "precision":
                core = [g for g in gt.get("locations", []) if g.get("accepted") != "buffer"]
                score = precision_mode(score, scorer(r["value"], {**gt, "locations": core}, ctx, geocoder=geocoder) if len(core) < len(gt.get("locations", [])) else None)
            if any(call.get("tool") == "google_maps" for call in calls):
                score.update(maps_citation_grounding(r["value"], calls))
            row = {"task_id": r["task_id"], "operation": operation, "run_id": run_id_of(rp), "side": s, **common_metrics(r, trace), **score}
            row.update(answer_format(r["value"], answer), **answer_schema(r["value"]))
            row.update(quality_metrics(operation, r["value"], row, answer))
            if row.get("grounded") is not None:
                row["correct_grounded"] = bool(row["correct"] and row["grounded"])
            if s == "baseline":
                row.update(dict.fromkeys(TOOL_COLS & set(row)))
            by_operation.setdefault(operation, []).append(row)
    return by_operation


CSV_OP_ORDER = ("directionality", "geocode", "geocode_reverse", "knn", "range_query", "route", "spatial_join")  # fixed column order for any task order
SHORT_COLS = {"latency_s", "total_tool_s", "llm_s", "prompt_tokens_k", "completion_tokens_k", "cached_tokens_k"}


def summary_row(operation, rows):
    """AVG row: mean for numbers (rate for booleans), mode for strings; per-call lists skipped."""
    out = {"task_id": "AVG", "operation": operation}
    for k in dict.fromkeys(k for r in rows for k in r if k not in out and k != "tool_latencies_s"):
        vals = [r[k] for r in rows if r.get(k) is not None]
        if not vals:
            continue
        out[k] = Counter(vals).most_common(1)[0][0] if isinstance(vals[0], str) else st.mean(vals)
    return out


def write_csv(path, by_operation):
    """One row per task plus an AVG row per operation."""
    rows = [x for op, rs in by_operation.items() for x in (*rs, summary_row(op, rs))]
    ops = sorted(by_operation, key=lambda op: CSV_OP_ORDER.index(op) if op in CSV_OP_ORDER else len(CSV_OP_ORDER))
    extra = list(dict.fromkeys(k for op in ops for r in by_operation[op] for k in r if k not in COMMON))

    def fmt_cell(k, v):
        """Format one CSV cell."""
        if isinstance(v, float):
            return fmt_round(v) if k in SHORT_COLS else float(f"{v:.7g}")
        if k == "tool_latencies_s":
            return ";".join(str(fmt_round(float(x))) for x in v.split(";") if x)
        return v

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, COMMON + extra, restval="")
        w.writeheader()
        for r in rows:
            w.writerow({k: fmt_cell(k, v) for k, v in r.items() if v is not None})


# --- reporting

def wilson(k, n, z=1.96):
    """95% Wilson score interval for k of n."""
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0, c - h), min(1, c + h)


def pct(xs, q):
    """Linear-interpolated quantile."""
    xs = sorted(xs)
    i = (len(xs) - 1) * q
    lo = int(i)
    return xs[lo] + (xs[min(lo + 1, len(xs) - 1)] - xs[lo]) * (i - lo)


def print_dist(rows, key, fmt=None):
    """Print mean, median and p95 of a column."""
    xs = [r[key] for r in rows if r.get(key) is not None]
    f = fmt_show if fmt is None else fmt.format
    if xs:
        print(f"{key:<17}: mean={f(st.mean(xs))}  median={f(st.median(xs))}  p95={f(pct(xs, .95))}")


def print_report(operation, rows):
    """Per-task table and aggregates for one operation."""
    n = len(rows)
    correct = sum(r["correct"] for r in rows)
    costs = [r["cost_usd"] for r in rows if r["cost_usd"] is not None]

    print(f"\n=== {operation} (n={n}) ===")
    print(f"{'task_id':<16}{'correct':<9}{'err_m':<10}{'latency_s':<12}{'cost_usd':<10}{'tools':<7}turns")
    for r in rows:
        err = fmt_show(r["err_m"]) if r.get("err_m") is not None else "-"
        cost = f"{r['cost_usd']:.5f}" if r["cost_usd"] is not None else "n/a"
        print(f"{r['task_id']:<16}{str(r['correct']):<9}{err:<10}{fmt_show(r['latency_s']):<12}{cost:<10}{str(r['tool_calls'] if r['tool_calls'] is not None else '-'):<7}{r['turns']}")

    lo, hi = wilson(correct, n)
    print(f"\naccuracy         : {correct}/{n} = {100*correct/n:.0f}%  (95% Wilson {100*lo:.0f}-{100*hi:.0f}%)")
    for k in dict.fromkeys(k for r in rows for k in r if k not in COMMON):  # operation-specific metrics
        vals = [r[k] for r in rows if r.get(k) is not None]
        if vals and all(isinstance(v, bool) for v in vals):
            print(f"{k:<17}: {sum(vals)}/{len(vals)}")
        elif vals and all(isinstance(v, str) for v in vals):
            print(f"{k:<17}: {dict(Counter(vals))}")
        elif vals and all(isinstance(v, (int, float)) for v in vals):
            print_dist(rows, k, "{:.2f}")
    if "n_correct_items" in rows[0]:  # set metrics pooled over all items
        print("micro p / r / f1 : " + " / ".join(f"{x:.3f}" for x in prf(*(sum(r[k] for r in rows) for k in ("n_correct_items", "n_answered", "n_gt")))))
    for k in ("latency_s", "total_tool_s", "llm_s", "prompt_tokens_k", "completion_tokens_k"):
        print_dist(rows, k)
    print_dist(rows, "turns", "{:.1f}")
    print_dist(rows, "tool_calls", "{:.1f}")
    if len(costs) == n:
        print(f"total cost       : ${sum(costs):.5f}   mean/task=${st.mean(costs):.6f}")
    else:
        print(f"total cost       : n/a ({len(costs)}/{n} tasks have cost data)")
    cached = [r for r in rows if r.get("cached_tokens_k") is not None and r.get("prompt_tokens_k")]
    if cached:
        print(f"cache-hit rate   : {100*sum(r['cached_tokens_k'] for r in cached)/sum(r['prompt_tokens_k'] for r in cached):.1f}% of prompt tokens")
    g = [r["grounded"] for r in rows if r.get("grounded") is not None]
    if g:
        print(f"grounded         : {sum(g)}/{len(g)}")
        print(f"correct+grounded : {sum(r['correct_grounded'] for r in rows if r.get('correct_grounded') is not None)}/{len(g)}")
    if all(r["n_tool_errors"] is not None for r in rows):
        print(f"mean repeated    : {st.mean(r['n_redundant_calls'] for r in rows):.2f}   tool errors: {sum(r['n_tool_errors'] for r in rows)}")
    print(f"zero_hit         : {sum(r['zero_hit'] for r in rows)}/{n}")


def check_directionality():
    """Compass labels, angular error, failure modes and tool grounding."""
    W = 360 / len(LABELS_8)  # sector width; label edges sit at W / 2 past each label's centre
    eps = 0.1  # degrees, a hair off an edge
    deg = lambda label: LABELS_8.index(label) * W
    A = {"osm_id": 1, "source_table": "osm.planet_osm_polygon", "lat": 40.0, "lon": -86.0}
    B = {"osm_id": 2, "source_table": "osm.planet_osm_point", "lat": 40.0, "lon": -86.1}
    ends = ({"entity_id": "point:2"}, {"entity_id": "polygon:1"})  # (origin, destination) as the tool should be called
    by_xy = ({"lat": 40.0, "lon": -86.1}, {"lat": 40.0, "lon": -86.0})
    E = deg("E")
    ok_out = {"bearing_deg": E + eps, "direction": "E"}
    bad_out = {"bearing_deg": deg("W") + eps, "direction": "W"}

    def ctx(*calls, out=None):
        """Tool context: bearing(origin, destination) calls in order, one shared output."""
        return {"rows": [{"loc": {"entity_id": e}} for e in ("polygon:1", "point:2")], "outs": [out] if out else [],
                "calls": [{"tool": "bearing", "args": {"origin": o, "destination": d}} for o, d in calls]}

    def sc(answer, bearing, c=None):
        """Score an answer against a ground-truth bearing."""
        return score_directionality(answer, {"bearing_deg": bearing, "location_a": A, "location_b": B}, c or ctx())

    # labels: half-up at the sector edges, wrapping at north
    for bearing, label in ((0, "N"), (W / 2 - eps, "N"), (W / 2, "NE"), (deg("E") - W / 2, "E"), (360 - W / 2 - eps, "NW"), (360 - W / 2, "N"), (360 - eps, "N")):
        assert compass_label(bearing) == label, bearing

    # correctness: exact, adjacent passes, label formats
    assert sc({"direction": "N"}, eps)["exact"] and sc(" ne ", deg("NE"))["correct"] and sc({"direction": "NE"}, W / 2)["correct"]
    a = sc("N", W / 2)
    assert not a["correct"] and a["failure_mode"] == "adjacent_off"
    a = sc("NE", W / 2 - 2)
    assert not a["exact"] and a["adjacent"] and a["angular_error_deg"] == W / 2 + 2

    # angular error wraps around north
    a = sc("N", 360 - W / 4)
    assert a["exact"] and a["angular_error_deg"] == W / 4
    a = sc("NW", 2)
    assert a["gt_label"] == "N" and a["adjacent"] and a["failure_mode"] == "adjacent_off"
    a = sc("E", W - 1)  # adjacent label while the error exceeds one sector
    assert a["gt_label"] == "NE" and a["adjacent"] and a["failure_mode"] == "adjacent_off"

    # failure modes: wrong, reversed (opposite or one step off opposite)
    for answer, bearing, mode in (("E", W / 4, "wrong"), ("S", 360 - 1, "reversed"), ("SW", W, "reversed"), ("W", W, "wrong"),
                                  ("N", deg("S"), "reversed"), ("SE", deg("NW"), "reversed")):
        assert sc(answer, bearing)["failure_mode"] == mode, (answer, bearing)
    assert not sc("E", W / 4)["adjacent"]

    # unparseable answers
    junk_modes = (("bogus", "invalid_label"), ("NNE", "invalid_label"), ({"direction": "up"}, "invalid_label"), (None, "no_answer"),
                  ("", "no_answer"), ({}, "no_answer"), ({"direction": None}, "no_answer"), ({"status": "error"}, "no_answer"), (7, "no_answer"))
    for junk, mode in junk_modes:
        a = sc(junk, W / 4)
        assert a["failure_mode"] == mode and not a["correct"] and a["angular_error_deg"] is None and not a["grounded"], junk

    # tool grounding: argument order, endpoints, which call counts
    a = sc("E", E, ctx(ends, out=ok_out))
    assert a["args_order_ok"] and a["endpoints_used"] and a["endpoints_found"] and a["grounded"] and a["gt_in_tool_output"]
    a = sc("E", E, ctx(ends[::-1], out=bad_out))  # swapped args, answer guessed
    assert a["args_order_ok"] is False and a["endpoints_used"] and not a["grounded"] and not a["gt_in_tool_output"]
    a = sc("W", E, ctx(ends[::-1], out=bad_out))  # grounded in a wrong tool output
    assert a["failure_mode"] == "reversed" and a["grounded"] and not a["gt_in_tool_output"]
    assert sc("E", E, ctx(ends[::-1], ends, out=ok_out))["args_order_ok"]  # last call wins
    a = sc("E", E, ctx(by_xy, out=ok_out))  # coordinate refs
    assert a["args_order_ok"] and a["endpoints_used"]
    a = sc("E", E, ctx(by_xy[::-1], out=ok_out))
    assert a["args_order_ok"] is False and a["endpoints_used"]
    a = sc("E", E, ctx(ends[:1] * 2, out=ok_out))  # same place twice
    assert a["args_order_ok"] is False and not a["endpoints_used"]
    a = sc("E", E, ctx())  # correct without any tool call
    assert a["args_order_ok"] is None and not a["endpoints_used"] and not a["grounded"] and a["correct"]
    a = sc("NE", deg("NE") - 2, ctx(ends, out={"locations": [{"bearing_deg": deg("NE") - 5, "direction": "NNE"}]}))  # nested rows, label from bearing
    assert a["grounded"]
    a = sc("E", E, {"rows": [], "outs": [{"direction": "N/A"}], "calls": [{"tool": "bearing", "args": "bad"}]})
    assert not a["endpoints_found"] and not a["grounded"]


def check_geocode():
    """Name match, coordinate tolerance and entity_id reading, for forward and reverse geocoding."""
    north = lambda m: {"lat": 40.0 + m / 111320, "lon": -86.0}  # m metres north of the place
    here = north(0)
    gt = {"osm_id": 1, "name": "Central Arena (CA)", **here, "source_table": "osm.planet_osm_polygon"}
    row = {"loc": {"entity_id": "polygon:1", "name": "Central Arena", **here}, "call": 0, "rank": 1, "n": 1, "kind": "polygon", "osm_id": "1"}
    ctx = {"rows": [row], "calls": [], "outs": []}
    for score in (score_geocode, score_geocode_reverse):
        ok = lambda **loc: score({"location": loc}, gt, ctx)["correct"]

        # name match ignores case and parenthetical aliases; coordinates must sit within CORRECT_M
        assert ok(name="Central Arena", **here) and ok(name="central arena (CA)", **north(0.9 * CORRECT_M))
        for name in ("Central Arena Parking", "Central", "North Hall"):  # no prefix or partial matching
            assert not ok(name=name, **here), name
        assert not ok(**here) and not ok(name="Central Arena")
        assert not ok(name="Central Arena", **north(2 * CORRECT_M)) and not ok(name="Central Arena", lat="40.0", lon="-86.0")

        # entity_id is read from the answer itself
        by_id = lambda eid: ok(name="Central Arena Parking", entity_id=eid, **here)
        assert by_id("polygon:1") and by_id("1") and not by_id("point:1") and not by_id("polygon:2")
        assert score({"location": {"name": "North Hall", **here}}, gt, ctx)["failure_mode"] == "wrong_name"


def check_route():
    """Distance tolerance, units, failure modes and tool grounding for route, on one synthetic distance D."""
    D, T = 1000.0, 10.0  # synthetic ground-truth distance (m) and time (min)
    tol = max(ROUTE_REL_TOL * D, ROUTE_ABS_TOL_M)
    near, other = D + 3 * tol, D * (1 + 2 * NEAR_REL_TOL)  # just off tolerance (e.g. another travel mode), clearly wrong
    A = {"osm_id": 1, "source_table": "osm.planet_osm_polygon", "lat": 40.0, "lon": -86.0}
    B = {"osm_id": 2, "source_table": "osm.planet_osm_point", "lat": 40.0, "lon": -86.1}
    G = {"location_a": A, "location_b": B, "mode": "foot", "distance_m": D, "time_min": T}
    ra, rb = {"entity_id": "polygon:1"}, {"entity_id": "point:2"}
    by_xy = ({"lat": 40.0, "lon": -86.0}, {"lat": 40.0, "lon": -86.1})
    out = {"distance_m": D + tol / 10, "duration_min": T * (1 + ROUTE_REL_TOL / 10)}  # the tool's own unrounded result
    m = lambda d: {"distance_m": d}

    def ctx(*calls):
        """Tool context: route(origin, destination, profile) calls, each with its result."""
        return {"rows": [{"loc": {"entity_id": e}} for e in ("polygon:1", "point:2")], "outs": [], "calls": [
            {"tool": "route", "args": {"origin": o, "destination": d, **({"profile": p} if p else {})}, "mcp_calls": [{"result": r}]} for o, d, p, r in calls]}

    good = ctx((ra, rb, "foot", out))
    rs = lambda answer, c=good, gt=G: score_route(answer, gt, c)

    # exact answer: every diagnostic passes
    a = rs({"distance_m": D, "duration_min": T})
    for k in ("correct", "grounded", "gt_in_tool_output", "mode_ok", "args_order_ok", "endpoints_used", "endpoints_found"):
        assert a[k], k
    assert a["failure_mode"] == "none" and a["dist_err_m"] == 0 and a["time_err_pct"] == 0
    assert a["tool_distance_m"] == out["distance_m"] and a["tool_time_min"] == out["duration_min"]

    # tolerance: inside passes in any unit key, just outside is off_tolerance (also with a time), far off is wrong
    for answer in (m(D + 0.9 * tol), m(D - 0.9 * tol), {"distance_km": D / UNIT_M["km"]}, {"distance_mi": D / UNIT_M["mi"], "distance_m": D}):
        assert rs(answer)["correct"], answer
    for answer, mode in ((m(near), "off_tolerance"), (m(D - 3 * tol), "off_tolerance"), (m(other), "wrong")):
        assert not rs(answer)["correct"] and rs(answer)["failure_mode"] == mode, answer
    assert not rs({"distance_m": near, "duration_min": T})["correct"]

    # unit errors: the value in another unit, or a thousand times off; a string is no number
    for wrong in (D / UNIT_M["km"], D * UNIT_M["km"], D / UNIT_M["mi"]):
        assert rs(m(wrong))["failure_mode"] == "unit_error" and not rs(m(wrong))["correct"], wrong
    assert not rs(m(str(D)))["correct"]

    # no usable answer
    for junk in (None, {}, {"status": "unanswerable"}, {"distance_m": None}, 7):
        a = rs(junk)
        assert a["failure_mode"] == "no_answer" and not a["correct"] and a["grounded"] is False and a["dist_err_m"] is None

    # travel mode: right answer from a wrong-mode call, grounded in a wrong output, profile defaults to car
    car = ctx((ra, rb, "car", m(other)))
    a = rs(m(D), car)
    assert a["correct"] and not a["mode_ok"] and not a["grounded"] and a["gt_in_tool_output"] is False
    a = rs(m(other), car)
    assert a["failure_mode"] == "wrong_mode" and a["grounded"] and not a["gt_in_tool_output"] and not a["mode_ok"]
    assert rs(m(other), ctx((ra, rb, None, m(other))))["failure_mode"] == "wrong_mode"

    # endpoints: reversed, repeated, outside the data; a route is symmetric so a reversed call can still be right
    assert rs(m(other), ctx((rb, ra, "foot", m(other))))["failure_mode"] == "reversed_endpoints"
    a = rs(m(D), ctx((rb, ra, "foot", m(D))))
    assert a["correct"] and a["args_order_ok"] is False and a["endpoints_used"]
    for o, d in ((ra, ra), (ra, {"lat": 41, "lon": -86})):
        assert rs(m(other), ctx((o, d, "foot", m(other))))["failure_mode"] == "wrong_endpoint"
    assert rs(m(D), ctx((*by_xy, "foot", out)))["args_order_ok"]  # coordinate refs

    # which call counts: last one wins, a failed retry is ignored, no tools or bad arguments
    a = rs(m(D), ctx((rb, ra, "car", m(other)), (ra, rb, "foot", out)))
    assert a["mode_ok"] and a["args_order_ok"] and a["tool_distance_m"] == out["distance_m"]
    a = rs(m(D), ctx((ra, rb, "foot", out), (ra, rb, "foot", {"error": "no route"})))
    assert a["tool_distance_m"] == out["distance_m"] and a["grounded"]
    a = rs(m(D), ctx())
    assert a["correct"] and not a["grounded"] and a["mode_ok"] is None and a["args_order_ok"] is None and a["tool_distance_m"] is None
    assert rs(m(D), {"rows": [], "outs": [], "calls": [{"tool": "route", "args": "bad"}]})["mode_ok"] is None

    # unanswerable ground truth: only an explicit non-answer is correct
    U = {**G, "distance_m": None}
    assert score_route({"status": "unanswerable"}, U, good)["correct"]
    for answer in (m(D), None):
        assert not score_route(answer, U, good)["correct"]


def check_knn():
    """Set metrics, order, answer parsing, grounding, call diagnostics and network distances for knn."""
    ref = {"osm_id": 99, "source_table": "osm.planet_osm_point", "name": "Ref", "lat": 39.9, "lon": -86.0}
    P = lambda i: {"name": f"P{i}", "lat": 40.0 + 0.01 * i, "lon": -86.0}  # i-th nearest, ~1.1 km apart

    def kgt(n=4, **kw):
        """Ground truth: the n nearest places, 100 m apart in distance_m."""
        return {"k": n, "distance_type": "euclidean", "include": {"amenity": "bar"}, "mode": None, "profile": None, "reference": ref, **kw, "locations": [
            {"osm_id": i, "source_table": "osm.planet_osm_point", **P(i), "distance_m": 100.0 * (i + 1)} for i in range(n)]}

    def kcall(tool="knn_search", ref_=None, **args):
        """One knn tool call with defaults matching the ground truth."""
        return {"tool": tool, "args": {"reference": ref_ or {"entity_id": "point:99"}, "include": {"amenity": "bar"}, "limit": 4, **args}, "mcp_calls": [{"result": {"locations": []}}]}

    kctx = lambda *calls, n=6: {"calls": list(calls), "outs": [], "rows": [{"loc": {"entity_id": f"point:{i}", **P(i)}} for i in range(n)]}  # n returned rows
    ks = lambda items, gt=None, ctx=None: score_knn({"locations": items}, gt or kgt(), ctx or kctx(kcall()))
    full = [P(i) for i in range(4)]

    # perfect answer
    a = ks(full)
    assert a["correct"] and a["precision"] == a["recall"] == a["f1"] == a["jaccard"] == 1 and a["exact_order"] and a["kendall_tau"] == 1 and a["rank_match"] == 1 and a["top1_correct"]
    assert a["failure_mode"] == "none" and a["grounded"] and a["gt_in_tool_output"] and a["k_ok"] and a["filter_ok"] and a["reference_ok"] and a["distance_type_ok"] and a["profile_ok"] is None

    # set metrics: subset, reversed subset, superset, order only, repeat
    a = ks(full[:3])
    assert not a["correct"] and a["precision"] == 1 and a["recall"] == 0.75 and a["n_missing"] == 1 and a["failure_mode"] == "missing_items" and a["n_dropped"] == 1
    a = ks(full[::-1][:3])
    assert a["failure_mode"] == "missing_items" and a["kendall_tau"] == -1 and a["rank_match"] == 0 and not a["top1_correct"]
    a = ks(full + [P(4)])
    assert not a["correct"] and a["precision"] == 0.8 and a["recall"] == 1 and a["failure_mode"] == "extra_items" and a["n_extra"] == 1
    a = ks([full[1], full[0], full[2], full[3]])  # order is diagnostic only
    assert a["correct"] and not a["exact_order"] and a["failure_mode"] == "none" and a["rank_match"] == 0.5 and round(a["kendall_tau"], 3) == 0.667 and not a["top1_correct"]
    a = ks([full[0], full[1], full[1], full[2], full[3]])  # a repeat is an extra
    assert not a["correct"] and a["n_duplicates"] == 1 and a["n_extra"] == 1 and a["failure_mode"] == "extra_items" and a["precision"] == 0.8

    # no answer, empty ground truth, a single location for k=1
    for junk in ({"locations": []}, {"status": "unanswerable"}, None, {}, "x", {"locations": ["P0", 5]}):
        a = score_knn(junk, kgt(), kctx(kcall()))
        assert a["failure_mode"] == "no_answer" and not a["correct"] and a["precision"] == a["recall"] == a["f1"] == 0 and not a["grounded"] and a["n_answered"] == 0
    assert score_knn({"locations": []}, kgt(0), kctx())["correct"] and not score_knn({"locations": full}, kgt(0), kctx())["correct"]
    a = score_knn({"location": P(0)}, kgt(1), kctx(kcall(limit=1)))
    assert a["correct"] and a["k_ok"]

    # ties in distance_m make places interchangeable in the order metrics
    tie = kgt()
    tie["locations"][2]["distance_m"] = tie["locations"][1]["distance_m"]
    a = ks([full[0], full[2], full[1], full[3]], tie)
    assert a["correct"] and a["exact_order"] and a["rank_match"] == 1 and a["kendall_tau"] == 1

    # identifying a place: names, coordinates, entity_id
    a = ks([{"lat": p["lat"], "lon": p["lon"]} for p in full])  # unnamed items
    assert not a["correct"] and a["precision"] == 0 and a["n_extra"] == 4
    a = ks([{"name": p["name"]} for p in full])  # names only
    assert a["correct"] and a["grounded"]
    north = lambda p, m: {**p, "lat": p["lat"] + m / 111320}  # moved m metres north
    a = ks([north(full[0], 2 * CORRECT_M), *full[1:]])  # coordinates off: not a returned row
    assert not a["correct"] and a["failure_mode"] == "hallucinated" and a["n_correct_items"] == 3
    assert ks([north(full[0], 0.9 * CORRECT_M), *full[1:]])["correct"]
    assert not ks([{**full[0], "lat": "40.0"}, *full[1:]])["correct"]  # not numeric
    assert ks([{"name": "Other", "entity_id": "point:3", **P(3)}, *full[:3]])["correct"]  # id with a name identifies the place
    assert ks([{**full[0], "name": "p0 "}, {**full[1], "name": "P1 (Bar)"}, *full[2:]])["correct"]  # name_eq rules

    # grounding: invented place, real but wrong place, no tool calls, rows missing from tool output
    a = ks([*full[:3], {"name": "Invented", "lat": 40.5, "lon": -86.0}])
    assert a["failure_mode"] == "hallucinated" and a["ungrounded_count"] == 1 and not a["grounded"]
    a = ks([*full[:3], P(5)])
    assert a["failure_mode"] == "wrong" and a["ungrounded_count"] == 0
    a = ks(full, ctx=kctx(n=0))
    assert a["correct"] and not a["grounded"] and a["k_ok"] is None and a["filter_ok"] is None and a["n_knn_calls"] == 0
    a = ks(full, ctx=kctx(kcall(), n=2))
    assert a["correct"] and a["tool_set_recall"] == 0.5 and not a["gt_in_tool_output"] and a["ungrounded_count"] == 2

    # call diagnostics: limit, filter, reference, distance type; the last call wins
    a = ks(full[:3], ctx=kctx(kcall(limit=3)))
    assert a["failure_mode"] == "wrong_k" and not a["k_ok"]
    a = ks(full[:3], ctx=kctx(kcall(limit=10)))  # a larger limit is not a failure
    assert a["failure_mode"] == "missing_items" and not a["k_ok"]
    a = ks(full[:3], ctx=kctx(kcall(include={"amenity": "pub"})))
    assert a["failure_mode"] == "wrong_filter" and not a["filter_ok"]
    a = ks(full[:3], ctx=kctx(kcall(ref_={"entity_id": "point:7"})))
    assert a["failure_mode"] == "wrong_reference" and not a["reference_ok"]
    assert ks(full[:3], ctx=kctx(kcall(ref_={"lat": 39.9, "lon": -86.0})))["reference_ok"]
    a = ks(full[:3], ctx=kctx(kcall("knn_search_network", profile="foot")))
    assert a["failure_mode"] == "wrong_distance_type" and not a["distance_type_ok"]
    a = ks(full[:3], ctx=kctx(kcall(), kcall(include={"amenity": "pub"})))
    assert a["failure_mode"] == "wrong_filter" and a["n_knn_calls"] == 2

    # filter_covers: the call's include must have the ground-truth keys and allow its values
    assert filter_covers({"amenity": ["bar", "PUB"]}, {"amenity": "bar"}) and filter_covers({"amenity": "*"}, {"amenity": "bar"}) and filter_covers({"amenity": "*ar*"}, {"amenity": "bar"})
    assert not filter_covers({"amenity": "bar", "shop": "x"}, {"amenity": "bar"}) and not filter_covers({"shop": "bar"}, {"amenity": "bar"})
    assert not filter_covers({"amenity": ["pub"]}, {"amenity": "bar"}) and not filter_covers(None, {"amenity": "bar"})

    # distances and travel profile for network ground truth
    nt = lambda **kw: kgt(distance_type="network", mode="foot", profile="foot", **kw)
    nc = lambda **args: kctx(kcall("knn_search_network", profile="foot", **args))
    with_d = lambda ds, key="distance_m": [{**p, key: d} for p, d in zip(full, ds)]
    gt_d = [g["distance_m"] for g in kgt()["locations"]]
    tol = lambda d: max(ROUTE_REL_TOL * d, ROUTE_ABS_TOL_M)
    a = ks(with_d([d + 0.5 * tol(d) for d in gt_d]), nt(), nc())
    assert a["correct"] and a["dist_ok"] and a["profile_ok"] and a["n_dist"] == len(gt_d)
    off = [d + 2 * tol(d) for d in gt_d]  # outside the route tolerance: reported, never gating
    a = ks(with_d(off), nt(), nc())
    assert a["correct"] and not a["dist_ok"] and a["dist_mae_m"] == st.mean(o - d for o, d in zip(off, gt_d))
    assert ks(with_d([d / UNIT_M["km"] for d in gt_d], "distance_km"), nt(), nc())["dist_ok"]
    a = ks(with_d([1, 2, 3, 4], "geodesic_distance_m"), kgt(), kctx(kcall()))  # euclidean: error only
    assert a["dist_ok"] is None and a["n_dist"] == 4 and a["dist_mae_m"] == 247.5
    a = ks(full, nt(), kctx(kcall("knn_search_network", profile="car")))
    assert a["correct"] and not a["profile_ok"] and a["n_dist"] == 0 and a["dist_ok"] is None
    assert ks(full[:3], nt(), kctx(kcall("knn_search_network", profile="car")))["failure_mode"] == "wrong_distance_type"
    assert not ks(full[:3], nt(), kctx(kcall("knn_search_network")))["profile_ok"]  # profile defaults to car


def check_range_join():
    """Range and join basics on a hand-built ground truth; check_sets covers them by mutation."""
    G = lambda i: {"name": f"p{i}", "lat": 40.0 + i / 1e3, "lon": -86.0, "osm_id": i, "source_table": "osm.planet_osm_point"}
    answer = lambda *ix: {"locations": [{k: G(i)[k] for k in ("name", "lat", "lon")} for i in ix]}
    gt01 = {"locations": [G(0), G(1)]}

    # range: order ignored, wrong / missing / extra / no answer / empty truth / repeats
    assert score_range(answer(0, 1), gt01)["correct"] and score_range(answer(1, 0), gt01)["correct"]
    a = score_range(answer(0, 2), gt01)
    assert not a["correct"] and a["failure_mode"] == "wrong" and a["f1"] == 0.5
    assert score_range(answer(0), gt01)["failure_mode"] == "missing_items" and score_range(answer(0, 1, 2), gt01)["failure_mode"] == "extra_items"
    assert score_range({}, {"locations": [G(0)]})["failure_mode"] == "no_answer" and score_range(answer(), {"locations": []})["correct"]
    assert not score_range(answer(0, 0), gt01)["correct"] and not score_range(answer(0, 0), {"locations": [G(0)]})["correct"] and score_range(answer(0, 0), {"locations": [G(0)]})["n_duplicates"] == 1

    # join pairs: order ignored, the partner must match, a left join row without partner, semi/anti as locations
    pairs = lambda *ab: {"pairs": [{"location_a": G(a), "location_b": G(b) if b is not None else None} for a, b in ab]}
    assert score_join(pairs((0, 1), (2, 3)), pairs((2, 3), (0, 1)))["correct"] and not score_join(pairs((0, 3)), pairs((0, 1)))["correct"]
    assert score_join(pairs((0, None)), pairs((0, None)))["correct"] and not score_join(pairs((0, 1)), pairs((0, None)))["correct"]
    assert score_join(answer(0), {"locations": [G(0)]})["correct"] and not score_join(answer(0), gt01)["correct"]


def check_helpers():
    """Set arithmetic, rank correlation and the Wilson interval."""
    assert prf(3, 4, 4) == (0.75, 0.75, 0.75) and prf(0, 0, 0) == (1, 1, 1) and prf(0, 2, 0)[2] == 0
    assert kendall_tau([0]) is None and kendall_tau([1, 0]) == -1
    assert wilson(10, 10)[0] > 0.7
    maps = [{"tool": "google_maps", "result": [{"places": [{"name": "Review of Central Arena - Google Maps"}]}]}]
    assert maps_citation_grounding({"location": {"name": "Central Arena", "lat": 1, "lon": 2}}, maps)["maps_citation_grounded"]
    assert not maps_citation_grounding({"location": {"name": "North Hall", "lat": 1, "lon": 2}}, maps)["maps_citation_grounded"]
    assert maps_citation_grounding({"direction": "N"}, maps)["maps_citation_grounded"] is None


def self_check():
    """Run every scorer check; each group builds its own fixtures."""
    for check in (check_directionality, check_geocode, check_route, check_knn, check_range_join, check_helpers, check_sets, check_pipeline, check_format, check_schema, check_naming):
        check()


def check_sets(tasks_path=DEFAULT_TASKS):
    """Set scorers (knn, range, join) on every answer key in the task file plus synthetic joins: the ground truth answers itself, variants flip as expected."""
    P = lambda i: {"name": f"synthetic {i}", "lat": 10.0 + i / 100, "lon": 10.0, "osm_id": i, "source_table": "planet_osm_point"}
    pair = lambda a, b: {"location_a": a, "location_b": b}
    synthetic = [
        ("spatial_join", {"join_type": "inner", "set_a": {"x": 1}, "set_b": {"x": 1}, "pairs": [pair(P(0), P(1)), pair(P(0), P(2))]}),  # self join
        ("spatial_join", {"join_type": "left", "set_a": {"x": 1}, "set_b": {"x": 2}, "pairs": [pair(P(0), P(1)), pair(P(2), None)]}),
        ("spatial_join", {"join_type": "semi", "locations": [P(0), P(2)]}),
        ("range_query", {"locations": [P(3), {**P(4), "name": None}]}),  # a nameless place
    ]
    real = [(t["operation"], t["answer"]) for t in load_tasks(tasks_path).values()] if os.path.exists(tasks_path) else []
    outsider = {"name": "not in ground truth", "lat": -10.0, "lon": -10.0}
    north = lambda g, m: {**g, "lat": g["lat"] + m / 111320}  # moved m metres north
    renamed = lambda g: {**g, "name": g["name"] and g["name"].upper() + "."}  # case and punctuation differ
    radius = lambda g: CORRECT_M if g["name"] else NAMELESS_M  # match radius for this place
    for op, gt in synthetic + [(op, a) for op, a in real if op in ("knn", "range_query", "spatial_join")]:
        key = "pairs" if "pairs" in gt else "locations"
        truth = gt[key]
        if not truth:
            continue
        score = lambda items, k=key: SCORERS[op]({k: items}, gt, {"calls": [], "rows": [], "outs": []})
        each = (lambda f: [pair(f(x["location_a"]), x["location_b"] and f(x["location_b"])) for x in truth]) if key == "pairs" else (lambda f: [f(x) for x in truth])
        tag = lambda what: (op, what)

        # the ground truth is the perfect answer; order, name style and small coordinate offsets are tolerated, a repeat is not
        assert score(truth)["correct"], tag("identity")
        assert score(truth[::-1])["correct"], tag("any order")
        repeated = score(truth + truth[:1])
        assert not repeated["correct"] and repeated["n_duplicates"] == 1 and repeated["n_extra"] == 1, tag("a repeat is an extra")
        assert score(each(renamed))["correct"], tag("name style")
        assert score(each(lambda g: north(g, 0.9 * radius(g))))["correct"], tag("within the match radius")
        assert not score(each(lambda g: north(g, 2 * radius(g))))["correct"], tag("beyond the match radius")

        # a missing or an extra item breaks set equality
        dropped = score(truth[1:])
        assert not dropped["correct"] and dropped["n_missing"] == 1 and dropped["n_extra"] == 0, tag("drop one")
        added = score(truth + [pair(outsider, outsider) if key == "pairs" else outsider])
        assert not added["correct"] and added["n_extra"] == 1, tag("add one")
        if op != "spatial_join":
            continue
        assert score(truth)["correct_strict"] and score(truth)["correct_any_shape"], tag("identity, strict and any shape")

        # semi/anti: a set of set_a places; as pairs a semi join is recovered, an anti join with partners is not
        if key == "locations":
            as_pairs = score([pair(x, outsider) for x in truth], "pairs")
            semi = gt["join_type"] == "semi" and bool(truth)
            assert as_pairs["correct"] == semi == as_pairs["shape_recovered"] and not (semi and as_pairs["correct_strict"]) and as_pairs["correct_any_shape"], tag("set as pairs")
            continue

        # inner pairs are unordered, strict keeps roles (a self join has none); left joins stay role-ordered
        swapped = [pair(x["location_b"], x["location_a"]) for x in truth if x["location_b"]]
        if gt["join_type"] == "inner":
            self_join = gt.get("set_a") == gt.get("set_b")
            assert score(swapped)["correct"], tag("(u, v) = (v, u)")
            assert score(swapped)["correct_strict"] == self_join, tag("strict roles")
            both = score(truth + swapped[:1])
            assert not both["correct"] and both["n_duplicates"] == 1, tag("a pair and its swap is a repeat")
        elif swapped:
            assert not score(swapped + [x for x in truth if not x["location_b"]])["correct"], tag("left join role-ordered")

        # pairs nested in their set_a place row (null = no partner): recovered when some row names a partner
        nested = score([{**x["location_a"], "match": x["location_b"]} for x in truth], "locations")
        assert nested["correct"] == any(x["location_b"] for x in truth) and nested["correct_any_shape"] and nested["f1_any_shape"] == 1, tag("nested pairs")

def check_pipeline():
    """evaluate() on a synthetic task file in a temp dir (another city, k above 50, empty truth, every join type) with perfect and mutated answers in run.json."""
    import tempfile
    P = lambda i: {"name": f"place {i}", "lat": -33.9 + i / 1000, "lon": 18.4, "osm_id": i, "source_table": "planet_osm_point"}
    pair = lambda a, b: {"location_a": P(a), "location_b": P(b) if b is not None else None}
    many = [P(i) for i in range(60)]
    answers = {
        "knn-big": ("knn", {"k": 60, "distance_type": "euclidean", "include": {"shop": "x"}, "reference": P(999), "locations": many}),
        "range-empty": ("range_query", {"locations": []}),
        "join-inner": ("spatial_join", {"join_type": "inner", "set_a": {"x": 1}, "set_b": {"x": 2}, "pairs": [pair(0, 1), pair(2, 3)]}),
        "join-left": ("spatial_join", {"join_type": "left", "set_a": {"x": 1}, "set_b": {"x": 2}, "pairs": [pair(0, 1), pair(2, None)]}),
        "join-anti": ("spatial_join", {"join_type": "anti", "set_a": {"x": 1}, "set_b": {"x": 2}, "locations": [P(5)]}),
    }
    key = lambda gt: "pairs" if "pairs" in gt else "locations"
    perfect = {t: {key(gt): gt[key(gt)]} for t, (_, gt) in answers.items()}
    mutated = {t: {key(gt): gt[key(gt)][1:] or [P(7)]} for t, (_, gt) in answers.items()}  # drop one, or add one to an empty truth
    with tempfile.TemporaryDirectory() as d:
        for values, want in ((perfect, True), (mutated, False)):
            results = os.path.join(d, "run.json")
            entries = [{"task_id": t, "input": {"operation": answers[t][0]},
                        "output": {"answer": v, "metrics": {"latency_ms": 1.0, "cost_usd": 0.0, "zero_hit": False}},
                        "trace": {"transcript": [], "events": []}} for t, v in values.items()]
            source = os.path.join(d, "tasks.json")
            Path(source).write_text(json.dumps([{"task_id": t, "operation": op, "answer": gt} for t, (op, gt) in answers.items()]))
            with open(results, "w") as f:
                json.dump({"config": {"condition": "baseline", "model": "test", "tasks": [source]}, "tasks": entries}, f)
            rows = [r for rs in evaluate(None, [results]).values() for r in rs]
            assert len(rows) == len(answers)
            assert all(r["correct"] == want for r in rows), [(r["task_id"], r["correct"]) for r in rows]

def check_format():
    """Output-format conformance: each kind of answer key answered in its own shape, then in mutated shapes."""
    place = {"name": "a place", "lat": 1.0, "lon": 2.0}
    pair = {"location_a": place, "location_b": place}
    keys = {  # answer key -> a conforming answer
        "location": ({"location": place}, {"location": place}),
        "locations": ({"locations": [place], "radius_m": 1.0, "k": 1}, {"locations": [place]}),
        "pairs": ({"pairs": [pair], "max_distance_m": 1.0}, {"pairs": [pair, {"location_a": place, "location_b": None}]}),
        "direction": ({"direction": "N", "bearing_deg": 1.0}, {"direction": "n", "bearing_deg": 1.0}),
        "measure": ({"distance_m": 1.0, "location_a": place}, {"distance_km": 1.0, "duration_min": 1.0}),
    }
    for kind, (key, good) in keys.items():
        mode = lambda pred: answer_format(pred, key)["format_mode"]
        assert mode(good) == "ok" and answer_format(good, key)["format_ok"], kind
        # unusable objects, missing or extra result fields
        for junk, want in (("text", "not_an_object"), (None, "not_an_object"), ([good], "not_an_object"),
                           ({"status": "unanswerable"}, "no_result_field"), ({"result": good}, "no_result_field")):
            assert mode(junk) == want, (kind, junk)
        other = {"answer": "text"}
        assert mode(other) == "wrong_field", kind
        assert mode({**good, **other}) == "extra_fields", kind
        assert mode({**good, "notes": "x", "status": "ok"}) == "ok", kind  # notes and status are not result fields
        assert mode({**good, "total": 3}) == "bad_measure_name", kind  # a number without a <quantity>_<unit> name
    # content checks: places need numeric coordinates, direction an 8-point label
    unplaced = {"name": "a place"}
    assert answer_format({"locations": [unplaced]}, keys["locations"][0])["format_mode"] == "bad_location"
    assert answer_format({"pairs": [{"location_a": place, "location_b": unplaced}]}, keys["pairs"][0])["format_mode"] == "bad_location"
    assert answer_format({"direction": "NNE"}, keys["direction"][0])["format_mode"] == "bad_direction"
    assert answer_format({"distance": 1.0}, keys["measure"][0])["format_mode"] == "bad_measure_name"

def check_naming():
    """Travel time is read under either name, a one-item `locations` key accepts a single `location`, and compass arithmetic follows LABELS_8."""
    # travel time: the prompt says duration_min, an answer key may say time_min
    t = 12.5
    for pred_key, gt_key in itertools.product(TIME_KEYS, repeat=2):
        assert score_route_time({pred_key: t}, {gt_key: t})["correct"], (pred_key, gt_key)
        assert not score_route_time({pred_key: t * (1 + 2 * REL_TOL)}, {gt_key: t})["correct"], (pred_key, gt_key)
    # k = 1: a single place answers a one-item set, not a longer one
    place = {"name": "a place", "lat": 1.0, "lon": 2.0}
    assert answer_format({"location": place}, {"locations": [place]})["format_ok"]
    assert answer_format({"locations": [place]}, {"locations": [place]})["format_ok"]
    assert answer_format({"location": place}, {"locations": [place, place]})["format_mode"] == "wrong_field"
    # compass: every label's centre and its half-sector edges, for whatever LABELS_8 holds
    for i, label in enumerate(LABELS_8):
        assert compass_label(i * SECTOR_DEG) == label and compass_label(i * SECTOR_DEG + SECTOR_DEG / 2 - 1e-6) == label
        assert compass_label(i * SECTOR_DEG + SECTOR_DEG / 2) == LABELS_8[(i + 1) % len(LABELS_8)]


def check_schema():
    """Well-formedness: valid objects of every allowed shape pass (location and locations together too); each broken rule gets its own mode."""
    place = {"name": "a place", "lat": 1.0, "lon": 2.0, "address": None}
    valid = [{"location": place}, {"locations": [place, place]}, {"location": place, "locations": [place]}, {"locations": []},
             {"pairs": [{"location_a": place, "location_b": place, "geodesic_distance_m": 3.0}, {"location_a": place, "location_b": None}]},
             {"direction": LABELS_8[0], "bearing_deg": 1.0}, {"distance_m": 1.0, "duration_min": 2}, {"count": 0}, {"answer": ["any", {"thing": 1}]},
             {"status": STATUS_VALUES[0], "notes": "why"}, {"location": {**place, "tags": {"k": "v"}}, "notes": "an extra attribute is allowed"}]
    assert all(answer_schema(v)["schema_ok"] for v in valid), [v for v in valid if not answer_schema(v)["schema_ok"]]
    mode = lambda v: answer_schema(v)["schema_mode"]
    # not a JSON object at all, or nothing in it
    assert [mode(v) for v in (None, "text", [place], 7)] == ["not_an_object"] * 4 and mode({}) == "empty_object"
    # fields outside the contract, and objects without any result
    assert mode({"result": place}) == "unknown_field" and mode({"location": place, "total": 3}) == "unknown_field"
    assert mode({"notes": "only a note"}) == "no_result_field"
    # values of the wrong type or range
    assert mode({"location": {"name": "x"}}) == "bad_location" and mode({"location": {**place, "lat": "1.0"}}) == "bad_location"
    assert mode({"location": {**place, "lat": 91}}) == "bad_location" and mode({"location": {**place, "name": 3}}) == "bad_location" and mode({"location": [place]}) == "bad_location"
    assert mode({"locations": place}) == "bad_locations" and mode({"locations": [place, {"name": "x"}]}) == "bad_locations"
    assert mode({"pairs": [place]}) == "bad_pairs" and mode({"pairs": [{"location_a": place}]}) == "bad_pairs" and mode({"pairs": {"location_a": place}}) == "bad_pairs"
    assert mode({"direction": "NNE"}) == "bad_direction" and mode({"direction": "north"}) == "bad_direction"
    assert mode({"status": "maybe"}) == "bad_status" and mode({"count": 1.5}) == "bad_count" and mode({"count": True}) == "bad_count"
    assert mode({"distance_m": "5"}) == "bad_measure" and mode({"distance_m": float("nan")}) == "bad_measure" and mode({"distance_m": None}) == "bad_measure"
    assert mode({"location": place, "notes": ["a", "b"]}) == "bad_notes"
    base = {"json_valid": True, "schema_ok": True, "format_ok": True}
    q = lambda op, pred, **metrics: quality_metrics(op, pred, {**base, **metrics}, {"locations": []})
    assert q("geocode", {"location": place}, err_m=50)["outcome"] == "correct"
    assert q("geocode", {"location": place}, err_m=200)["outcome"] == "partially_correct"
    assert q("geocode", {"location": place}, err_m=201)["outcome"] == "incorrect"
    assert q("directionality", {"direction": "NE"}, pred_label="NE", exact=False, adjacent=True)["outcome"] == "partially_correct"
    assert q("route", {"distance_m": 2}, dist_err_pct=10, dist_err_m=2)["outcome"] == "partially_correct"
    assert q("knn", {"locations": []}, f1=.5)["outcome"] == "partially_correct"
    assert q("geocode", {"location": place}, json_valid=False, format_ok=False, schema_ok=False, err_m=0)["outcome"] == "correct"  # conformance never gates
    assert q("geocode", "40.0,-86.0", err_m=None)["outcome"] == "invalid_answer"


async def main():
    """Parse args, score runs, print reports, write CSV."""
    ap = argparse.ArgumentParser()
    ap.add_argument("results", nargs="+", help="run.json path(s).")
    ap.add_argument("--tasks", help="Benchmark task file (default: task sources recorded in run config).")
    ap.add_argument("--csv", help="Write one row per task run (union of columns) to this path.")
    ap.add_argument("--side", choices=("auto", "baseline", "mcp"), default="auto", help="No-tool baseline or MCP run (default: the `condition` recorded in results).")
    ap.add_argument("--regeocode", action="store_true", help="Re-geocode reverse answers via the MCP geocode tool / Mapbox (adds err_m_regeocode).")
    ap.add_argument("--mcp-command", default="npx", help="Command to launch the MCP server for --regeocode.")
    ap.add_argument("--mcp-arg", action="append", help="Repeatable MCP server argument (default: --yes tsx vendor/purduedb__limo/server.ts).")
    ap.add_argument("--mcp-env", action="append", help="Repeatable KEY=VALUE for the MCP server (default: local osm DB).")
    args = ap.parse_args()
    self_check()

    geocoder = mcp_client = None
    if args.regeocode:
        from mcp_client import AsyncMCPClient
        mcp_env = args.mcp_env or ["DB_NAME=osm", f"DB_USER={os.getenv('USER', '')}", "DB_HOST=localhost", "DB_PORT=5432"]
        mcp_args = args.mcp_arg or ["--yes", "tsx", "vendor/purduedb__limo/server.ts"]
        env = {**os.environ, **dict(kv.split("=", 1) for kv in mcp_env)}
        mcp_client = AsyncMCPClient(command=args.mcp_command, args=mcp_args, env=env)
        await mcp_client.start()
        geocoder = ForwardGeocoder(mcp_client=mcp_client, mapbox_token=os.getenv("MAPBOX_ACCESS_TOKEN"))

    try:
        if geocoder is not None:
            await geocoder.prepare(args.results)
        by_operation = evaluate(args.tasks, args.results, geocoder=geocoder, side=args.side)
        for operation, rows in by_operation.items():
            print_report(operation, rows)
        if args.csv:
            write_csv(args.csv, by_operation)
            print(f"\ncsv -> {args.csv}")
    finally:
        if mcp_client is not None:
            await mcp_client.close()


if __name__ == "__main__":
    asyncio.run(main())
