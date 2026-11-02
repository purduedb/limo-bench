"""L2 `spatial_join`: pairing, join types, limit_per_a, max_distance_m, set caps, output shape and determinism.

Sets are lists of entity_ids as range_search / knn_search / knn_search_network return them (L1b, assumed verified); a few
contract tests feed real search results in, everything else checks the join itself against an independent SQL oracle
(oracle.py) over whatever extract is loaded.
Run: python -m pytest tests/test_l2_spatial_join.py -v
"""
import json
import random
import re

import pytest

from fixtures import deg_box, far_references
from harness import assert_converted, call, error_of, place, rejects, session  # noqa: F401
from oracle import CAP, RADIUS_M, TOL, assert_join, assert_knn, assert_range, range_case, render

LIMIT = 50  # the max limit
UNKNOWN_ID = "point:9223372036854775807"  # grammatical, never an OSM id
SET_HINT = "Expected an entity_id (point:123, polygon:-456)."


def sj(session, **args):
    return call(session, "spatial_join", args)


def members(session, ids):
    """The first LIMIT entity ids of a set in set order (anti join against an empty set), plus the raw response."""
    r = sj(session, set_a=ids, set_b=[], join_type="anti", limit=LIMIT)
    return [loc["entity_id"] for loc in r["locations"]], r


def ids_of(res):
    return [loc["entity_id"] for loc in res["locations"]]


@pytest.fixture(scope="module")
def set_kinds(session, oracle, fx):
    """kind -> ids of category A as a search tool returns them (checked against the oracle), plus a plain list."""
    rows = oracle.near(fx.anchor_ll, RADIUS_M, fx.cat_a)
    args, _ = range_case(oracle, fx.anchor_ll, fx.cat_a, target_m=rows[min(30, len(rows)) - 1][1], cap=LIMIT)
    res = call(session, "range_search", {**args, "reference": place(fx.anchor_ll), "limit": LIMIT})
    assert_range(oracle, res, oracle.near(fx.anchor_ll, args["radius_m"], fx.cat_a), LIMIT, req=args)
    kinds = {"range_latlon": ids_of(res)}
    box = deg_box(fx.anchor_ll, args["radius_m"])
    tmp = call(session, "make_geom", {"op": "bbox", "references": [{"lon": box[1], "lat": box[2]}, {"lon": box[3], "lat": box[4]}]})["entity_id"]
    req = {"include": render(fx.cat_a), "exclude": "none"}
    res = call(session, "range_search", {"reference": place(tmp), "limit": LIMIT, **req})
    assert_range(oracle, res, oracle.near(box, 0, fx.cat_a), LIMIT, req=req)
    kinds["range_bbox_tmp"] = ids_of(res)
    rows = oracle.knn(fx.anchor["id"], 25, fx.cat_a)
    k = next((m for m in range(min(25, len(rows) - 1), 0, -1) if rows[m][1] - rows[m - 1][1] > 2 * TOL), None)
    res = call(session, "knn_search", {"reference": place(fx.anchor["id"]), "limit": k, **req})
    assert_knn(oracle, res, fx.anchor["id"], k, fx.cat_a, req=req)
    kinds["knn_entity"] = ids_of(res)
    ids = fx.a_ids[:30]
    random.Random(0).shuffle(ids)
    kinds["list"] = ids
    assert all(len(v) >= 5 for v in kinds.values()), {k: len(v) for k, v in kinds.items()}
    return kinds


# ---- correctness vs the oracle: pairwise (L16 orthogonal array) over kind x overlap x join_type x distance x k ----

GF4_MUL = ((0, 0, 0, 0), (0, 1, 2, 3), (0, 2, 3, 1), (0, 3, 1, 2))
KINDS = ("range_latlon", "range_bbox_tmp", "knn_entity", "list")
OVERLAPS = ("identical", "subset", "partial", "disjoint")
JOIN_TYPES = ("inner", "left", "semi", "anti")
DISTANCES = (None, "p25", "p75", "tiny")
KS = (None, 1, 3, LIMIT)
MATRIX = [
    (KINDS[a], OVERLAPS[b], JOIN_TYPES[a ^ b], DISTANCES[a ^ GF4_MUL[2][b]], KS[a ^ GF4_MUL[3][b]])
    for a in range(4) for b in range(4)
]


def overlap_b(a_ids, b_ids, overlap):
    if overlap == "identical":
        return a_ids[::-1]
    if overlap == "subset":
        return a_ids + b_ids[:10]
    if overlap == "partial":
        h = len(a_ids) // 2
        return a_ids[:h] + b_ids[:len(a_ids) - h]
    return b_ids


@pytest.mark.parametrize("kind,overlap,join_type,dist,k", MATRIX, ids=["-".join(map(str, r)) for r in MATRIX])
def test_join_matches_oracle(session, oracle, fx, set_kinds, kind, overlap, join_type, dist, k):
    """Every set kind, set overlap, join type, threshold and limit_per_a level (pairwise) matches the oracle exactly."""
    a_ids = set_kinds[kind]
    b_ids = overlap_b(a_ids, fx.b_ids, overlap)
    ds = sorted(d for (a, b), d in oracle.dist(a_ids, b_ids).items() if a != b)
    d = None if dist is None else oracle.gap_threshold(ds, 0.5 if dist == "tiny" else ds[len(ds) * (1 if dist == "p25" else 3) // 4])
    args = {"set_a": a_ids, "set_b": b_ids, "join_type": join_type, "limit": LIMIT}
    args.update({k_: v for k_, v in (("max_distance_m", d), ("limit_per_a", k)) if v is not None})
    assert_join(oracle, sj(session, **args), oracle.join(a_ids, b_ids, join_type, d, k), LIMIT)


def test_mixed_entity_kinds_join_matches_oracle(session, oracle, fx):
    """Point, line, polygon and relation-polygon entities in one list join by their shapes."""
    polygon = next(i for i in fx.a_ids + fx.b_ids + [i for i, _ in fx.dense] if i.startswith("polygon:"))
    a = list(dict.fromkeys([fx.anchor["id"], fx.line, polygon] + ([fx.relation_polygon] if fx.relation_polygon else [])))
    ds = sorted(oracle.dist(a, fx.b_ids).values())
    d = oracle.gap_threshold(ds, ds[len(ds) // 3])
    res = sj(session, set_a=a, set_b=fx.b_ids, join_type="left", max_distance_m=d, limit=LIMIT)
    assert_join(oracle, res, oracle.join(a, fx.b_ids, "left", d), LIMIT)


@pytest.mark.parametrize("join_type", ["inner", "left", "semi", "anti"])
def test_empty_sets_say_which_is_empty(session, oracle, fx, join_type):
    """An empty set_a gives an empty result of the right shape, an empty set_b leaves every set_a place unmatched; the guidance names the empty set (set_a first)."""
    a = fx.a_ids[:5]
    for set_a, set_b in (([], fx.b_ids), (a, []), ([], [])):
        assert_join(oracle, sj(session, set_a=set_a, set_b=set_b, join_type=join_type), oracle.join(set_a, set_b, join_type), 20)


def test_self_join_needs_identical_id_sets(session, oracle, fx):
    """Only identical id sets in an inner join without limit_per_a dedupe as a self-join, each pair with the smaller entity_id
    (byte order, also across id signs) first; equal sizes or duplicate ids do not fake one."""
    a = fx.a_ids[:6]
    mixed = [fx.anchor["id"], fx.line, fx.way_polygon, *fx.overlap_polygons]
    for set_a, set_b in ((a, a[:3] + fx.b_ids[:3]), ([a[0], a[0], a[1]], a[:3]), (a, a[::-1]), (mixed, mixed[::-1])):
        assert_join(oracle, sj(session, set_a=set_a, set_b=set_b, limit=LIMIT), oracle.join(set_a, set_b), LIMIT)
    assert_join(oracle, sj(session, set_a=a, set_b=a[::-1], limit_per_a=2, limit=LIMIT), oracle.join(a, a[::-1], k=2), LIMIT)


# ---- sets from search results ----

@pytest.fixture(scope="module")
def contract_cases(session, oracle, fx):
    """name -> (search tool, its arguments): searches whose entity_ids feed a join set; exclude none."""
    (key, _), = fx.cat_a.items()
    exc = {key: fx.value_list[key][1]}
    args, _ = range_case(oracle, fx.anchor_ll, fx.value_list, exc, cap=LIMIT)
    box = deg_box(fx.anchor_ll, args["radius_m"])
    tmp = call(session, "make_geom", {"op": "bbox", "references": [{"lon": box[1], "lat": box[2]}, {"lon": box[3], "lat": box[4]}]})["entity_id"]
    cat = {"include": render(fx.cat_a), "exclude": "none"}
    cases = {
        "range_filtered": ("range_search", args),
        "range_tmp_reference": ("range_search", {"reference": tmp, **cat}),
        "knn_filtered": ("knn_search", {"reference": fx.anchor["id"], "limit": 10, "include": render(fx.value_list), "exclude": render(exc, "or")}),
        "knn_network": ("knn_search_network", {"reference": fx.anchor["id"], "limit": 10, "profile": "foot", **cat}),
    }
    for n, far in enumerate(far_references(fx)):
        cases[f"knn_far_reference_{n}"] = ("knn_search", {"reference": far, "limit": 8, **cat})
    return cases


@pytest.fixture(params=["range_filtered", "range_tmp_reference", "knn_filtered", "knn_network", "knn_far_reference_0", "knn_far_reference_1"])
def search_ids(request, session, contract_cases):
    """The entity_ids a search tool returns, in its order."""
    tool, args = contract_cases[request.param]
    r = call(session, tool, {**args, "reference": place(args["reference"]), **({"limit": LIMIT} if tool == "range_search" else {})})
    assert r["count"] >= 2, request.param
    return ids_of(r)


def test_search_results_are_a_join_set_a(session, oracle, fx, search_ids):
    """The entity_ids of a range_search, knn_search or knn_search_network result, passed as set_a, join like the oracle says for every join type."""
    for jt in ("inner", "left", "semi", "anti"):
        assert_join(oracle, sj(session, set_a=search_ids, set_b=fx.b_ids, join_type=jt, limit=LIMIT), oracle.join(search_ids, fx.b_ids, jt), LIMIT)


def test_search_results_are_a_join_set_b(session, fx, search_ids):
    """As set_b, every member and nothing else pairs with a far-away set_a place."""
    far = fx.dense[-1][0]
    r = sj(session, set_a=[far], set_b=search_ids, limit=LIMIT)
    want = set(search_ids) - {far}
    assert {p["location_b"]["entity_id"] for p in r["pairs"]} == want and r["count"] == r["total_found"] == len(want)


# ---- the 100 cap per set ----

def test_set_cap_boundary_100_vs_101(session, fx):
    """100 listed ids are used whole, in list order; a 101st distinct id is an error naming the set and the cap; duplicates count once."""
    ids = [i for i, _ in fx.dense[:CAP + 1]]
    got, r = members(session, ids[:CAP])
    assert got == ids[:LIMIT] and r["total_found"] == CAP
    assert members(session, ids[:CAP] + [ids[0], ids[5]])[1]["total_found"] == CAP
    for name, other in (("set_a", "set_b"), ("set_b", "set_a")):
        text = error_of(session, "spatial_join", **{name: ids, other: [ids[0]]})
        assert f"{name} holds {CAP + 1} places; at most {CAP}. Split the list or narrow the search." in text, text
    assert f"set_a holds {len(fx.dense)} places; at most {CAP}." in error_of(session, "spatial_join", set_a=[i for i, _ in fx.dense], set_b=[])


# ---- distances ----

def test_max_distance_is_inclusive_at_the_exact_distance(session, oracle, fx):
    """A point pair at exactly max_distance_m is kept and dropped 1e-9 relative below it."""
    pa = [i for i in fx.a_ids if i.startswith("point:")][:6]
    pb = [i for i in fx.b_ids if i.startswith("point:")][:6]
    pairs = sorted(oracle.dist(pa, pb).items(), key=lambda kv: kv[1])
    for (a, b), d in pairs[:: max(1, len(pairs) // 5)][:5]:
        assert sj(session, set_a=[a], set_b=[b], max_distance_m=d)["count"] == 1, (a, b, d)
        assert sj(session, set_a=[a], set_b=[b], max_distance_m=d * (1 - 1e-9))["count"] == 0, (a, b, d)


def test_distance_is_between_shapes(session, fx):
    """A point inside a polygon, far from the polygon's own lat/lon, is at distance 0 and matches at max_distance_m 0."""
    if not fx.polygon_with_inner_point:
        pytest.skip("no polygon with an inner point far from its representative point")
    poly, pt = fx.polygon_with_inner_point
    r = sj(session, set_a=[poly], set_b=[pt], max_distance_m=0)
    assert r["count"] == 1 and r["pairs"][0]["geodesic_distance_m"] == 0


def test_semi_join_agrees_with_range_search(session, fx, ab_threshold):
    """A set_a place has a match within max_distance_m exactly when range_search with that radius finds a set_b place around it."""
    a, b, d = SUB_A(fx), SUB_B(fx), ab_threshold
    semi = {x["entity_id"] for x in full(session, a, b, max_distance_m=d, join_type="semi")["locations"]}
    for eid in a:
        near = call(session, "range_search", {"reference": place(eid), "radius_m": d, "include": render(fx.cat_b), "exclude": "none", "limit": LIMIT})
        assert (eid in semi) == bool({x["entity_id"] for x in near["locations"]} & (set(b) - {eid})), eid


# ---- input validation ----

def test_explicit_nulls_equal_omitted_arguments(session, fx):
    """null for every optional argument behaves exactly like omitting it."""
    omitted = sj(session, set_a=fx.a_ids, set_b=fx.b_ids)
    nulls = sj(session, set_a=fx.a_ids, set_b=fx.b_ids, join_type=None, max_distance_m=None, limit_per_a=None, limit=None, units=None)
    assert nulls == omitted and omitted["count"] == min(20, omitted["total_found"])


def test_sets_are_id_lists_only(session, fx):
    """A set is an array of entity_id strings; the old inline search specs and non-strings are rejected by the schema."""
    rejects(session, "Expected array, received object at set_a", "spatial_join", set_a={"entity_ids": fx.a_ids[:2]}, set_b=[])
    rejects(session, "Expected array, received object at set_b", "spatial_join", set_a=[], set_b={"range_search": {"reference": place(fx.anchor_ll)}})
    rejects(session, "Expected string, received number", "spatial_join", set_a=[1, 2], set_b=[])


def test_unknown_or_malformed_entity_ids_are_rejected_naming_them(session, fx):
    """A listed id that does not exist or is outside the table:int64 grammar fails loudly instead of being dropped, in either set."""
    bads = (UNKNOWN_ID, "point:abc", "point:9223372036854775808", "node:1", "point:1:x")
    for name, other in (("set_a", "set_b"), ("set_b", "set_a")):
        for bad in bads:
            for ids in ([bad], [fx.anchor["id"], bad]):
                text = error_of(session, "spatial_join", **{name: ids, other: []})
                assert f'Unrecognized entity_id "{bad}" in {name}. {SET_HINT}' in text, text
    assert "tmp:* entity expires 1 hour after creation" in error_of(session, "spatial_join", set_a=["tmp:route-00000000"], set_b=[])
    assert "in set_a." in error_of(session, "spatial_join", set_a=[UNKNOWN_ID], set_b=["point:abc"])  # set_a is checked first


def test_zero_padded_entity_id_in_a_list_is_deduped(session, fx):
    """point:0N names the same entity as point:N, so a list holding both has it once."""
    kind, n = fx.anchor["id"].split(":")
    assert members(session, [fx.anchor["id"], f"{kind}:0{n}"])[0] == [fx.anchor["id"]]


@pytest.mark.parametrize("arg,bad,message", [
    ("max_distance_m", -1, "max_distance_m must be a number >= 0"),
    ("limit_per_a", 0, "limit_per_a must be an integer from 1 to 50"),
    ("limit_per_a", 1.5, "limit_per_a must be an integer from 1 to 50"),
    ("limit_per_a", 51, "limit_per_a must be an integer from 1 to 50"),
    ("limit", 0, "limit must be an integer from 1 to 50"),
    ("limit", 51, "limit must be an integer from 1 to 50"),
])
def test_numeric_arguments_are_bounded(session, fx, arg, bad, message):
    """Negative distances and out-of-range or fractional limits are rejected naming the argument and the allowed range."""
    rejects(session, re.escape(message), "spatial_join", set_a=fx.a_ids[:2], set_b=fx.b_ids[:2], **{arg: bad})


@pytest.mark.parametrize("join_type", ["inner", "left", "semi", "anti"])
def test_limit_per_a_and_omitted_max_distance(session, oracle, fx, join_type):
    """With max_distance_m omitted every set_a place matches every other set_b place; limit_per_a keeps each place's nearest few."""
    a, b = fx.a_ids[:8], fx.b_ids[:8]
    for k in (None, 1, 3, LIMIT):
        assert_join(oracle, sj(session, set_a=a, set_b=b, join_type=join_type, limit_per_a=k, limit=LIMIT), oracle.join(a, b, join_type, None, k), LIMIT)


# ---- output contract ----

def test_limit_truncates_with_full_total_and_defaults_to_20(session, oracle, fx):
    """Pairs and locations stop at limit (20 by default) while total_found reports the full result and guidance says so."""
    a, b = fx.a_ids, fx.b_ids
    r = sj(session, set_a=a, set_b=b)
    assert (r["count"], r["total_found"]) == (20, len(oracle.join(a, b)["matched"]))
    assert r["guidance"] == f"Showing the nearest 20 of {r['total_found']}. Raise limit, reduce max_distance_m, or set limit_per_a."
    r = sj(session, set_a=a, set_b=b, join_type="semi", limit=2)
    assert (r["count"], r["total_found"]) == (2, len(oracle.join(a, b, "semi")["semi"]))


@pytest.mark.parametrize("unit", ["m", "km", "mi"])
def test_units_add_converted_distances_and_keep_metres(session, fx, unit):
    """units adds geodesic_distance_<unit> beside the metres value of every pair and semi location; anti locations carry no distance."""
    r = sj(session, set_a=fx.a_ids[:1], set_b=fx.b_ids, units=unit, limit=LIMIT)
    semi = sj(session, set_a=fx.a_ids[:3], set_b=fx.b_ids, join_type="semi", units=unit)
    assert r["pairs"] and semi["locations"]
    for x in r["pairs"] + semi["locations"]:
        assert_converted(x, "geodesic_distance", unit)
    anti = sj(session, set_a=fx.a_ids[:3], set_b=[], join_type="anti", units=unit)
    assert anti["locations"] and not any(k.startswith("geodesic_distance") for x in anti["locations"] for k in x)


def test_left_join_unmatched_row_has_null_location_b_and_no_distance(session, fx):
    """An unmatched set_a row is a pair with location_b null and no distance."""
    r = sj(session, set_a=fx.a_ids[:1], set_b=fx.b_ids[:1], join_type="left", max_distance_m=1e-3)
    assert r["count"] == 1 and r["pairs"] == [{"location_a": r["pairs"][0]["location_a"], "location_b": None}]
    assert r["pairs"][0]["location_a"]["entity_id"] == fx.a_ids[0]


# ---- metamorphic relations (small sets, so every result fits in one call) ----

SUB_A = lambda fx: fx.a_ids[:6]  # noqa: E731
SUB_B = lambda fx: fx.b_ids[:6]  # noqa: E731


@pytest.fixture(scope="module")
def ab_threshold(oracle, fx):
    ds = sorted(oracle.dist(SUB_A(fx), SUB_B(fx)).values())
    return oracle.gap_threshold(ds, ds[len(ds) // 3])


def full(session, set_a, set_b, **args):
    r = sj(session, set_a=set_a, set_b=set_b, limit=LIMIT, **args)
    assert r["count"] == r["total_found"], "result does not fit in one call"
    return r


def pair_ids(r):
    return {(p["location_a"]["entity_id"], p["location_b"]["entity_id"]) for p in r["pairs"] if p["location_b"]}


def test_join_types_partition_set_a(session, fx, ab_threshold):
    """semi is the distinct left side of inner, semi and anti partition set_a, left is inner plus the anti rows."""
    a, b, d = SUB_A(fx), SUB_B(fx), ab_threshold
    inner, left = full(session, a, b, max_distance_m=d), full(session, a, b, max_distance_m=d, join_type="left")
    semi = [x["entity_id"] for x in full(session, a, b, max_distance_m=d, join_type="semi")["locations"]]
    anti = [x["entity_id"] for x in full(session, a, b, max_distance_m=d, join_type="anti")["locations"]]
    assert semi and anti
    assert set(semi) == {x for x, _ in pair_ids(inner)} and len(semi) == len(set(semi))
    assert not set(semi) & set(anti) and sorted(semi + anti) == sorted(a)
    assert pair_ids(left) == pair_ids(inner)
    assert [p["location_a"]["entity_id"] for p in left["pairs"] if not p["location_b"]] == anti


# ---- determinism ----

def test_tied_distances_give_one_answer_for_any_input_order(session, fx):
    """Ties at equal distance resolve the same way for every input order and every repeat."""
    if not fx.tie:
        pytest.skip("no two point entities share coordinates near the anchor")
    a = [next(i for i in fx.a_ids if i not in fx.tie)]
    norm = lambda r: json.dumps(r.get("pairs", r.get("locations")), sort_keys=True)  # noqa: E731
    for args in ({"limit_per_a": 1}, {"join_type": "semi"}, {}):
        outs = [norm(sj(session, set_a=a, set_b=b, **args)) for b in (fx.tie, fx.tie[::-1], fx.tie)]
        assert outs[0] == outs[1] == outs[2], args
