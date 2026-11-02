"""L1b single-reference tools: `range_search` and `knn_search`.

What is specific to them: membership and nearest-first order (smallest area, then id, at equal distance), the include/exclude
filter grammar the three search tools share, the default exclude, total_found, guidance and bearing. Place validation, distance and azimuth
formulas are tested with lookup, route and bearing (L0, L1a). Expectations come from the SQL oracle over the loaded extract.
Run: python -m pytest tests/test_l1b_range_knn.py -v
"""
import pytest

from fixtures import far_references
from geo import compass_name
from harness import assert_converted, call, error_of, place, rejects, session, session_with  # noqa: F401
from oracle import AND, NOT, OR, RADIUS_M, T, assert_knn, assert_range, default_exclude, range_case, render

LIMIT = 50  # the tools' max limit
REF_KINDS = ("point_entity", "line_entity", "polygon_entity", "relation_polygon", "latlon")
DEFAULT_NOTE = "Default exclude applied (roads, trees, fences, landuse, street furniture); pass exclude none to keep them."
FORM = ('Form: key=value, key=a|b, key=*, key~text, key!=value; join with and, or, not and ( ); '
        'quote values with spaces or special characters, e.g. name="Main St".')
LOCATION_KEYS = ["entity_id", "name", "address", "lat", "lon", "tags", "perimeter_m", "area_m2", "geodesic_distance_m",
                 "bearing_deg", "direction", "merged_from", "merged_count"]


def search(session, tool, reference, **args):
    """range_search / knn_search with the reference as a place; exclude none unless given."""
    return call(session, tool, {"reference": place(reference), "exclude": "none", **args})


def range_search(session, reference, **args):
    return search(session, "range_search", reference, **args)


def knn_search(session, reference, **args):
    return search(session, "knn_search", reference, **args)


def reference_of(fx, kind):
    if kind == "relation_polygon" and not fx.relation_polygon:
        pytest.skip("no relation polygon with 2..199 entities near the anchor")
    return {"point_entity": fx.anchor["id"], "line_entity": fx.line, "polygon_entity": fx.way_polygon,
            "relation_polygon": fx.relation_polygon, "latlon": fx.anchor_ll}[kind]


# ---- range_search: membership, order, total_found ----

@pytest.mark.parametrize("kind", REF_KINDS)
@pytest.mark.parametrize("radius", ["zero", "positive"])
def test_range_search_matches_oracle_for_every_reference_kind(session, oracle, fx, kind, radius):
    """Every reference kind finds the oracle's places nearest first (area, then id at distance 0); radius 0 is intersection."""
    reference = reference_of(fx, kind)
    args = range_case(oracle, reference, target_m=120, cap=400)[0] if radius == "positive" else {"reference": reference, "radius_m": 0}
    assert_range(oracle, range_search(session, limit=LIMIT, **args), oracle.near(reference, args["radius_m"]), LIMIT)


def test_range_search_radius_membership_is_exact_at_the_edge(session, oracle, fx):
    """A place just inside radius_m is found and just outside is not, at several bearings."""
    rows = [r for r in oracle.near(fx.anchor["id"], RADIUS_M, fx.cat_a)[:LIMIT] if r[1] > 100]
    for eid, d in rows[:: max(1, len(rows) // 3)][:3]:
        found = lambda f: [x["entity_id"] for x in range_search(session, fx.anchor["id"], radius_m=d * f, include=render(fx.cat_a), limit=LIMIT)["locations"]]
        assert eid in found(1 + 3e-5), (eid, d)
        assert eid not in found(1 - 2e-4), (eid, d)


@pytest.mark.parametrize("limit", [None, 3, LIMIT], ids=["default_20", "3", "max"])
def test_range_search_limit_keeps_the_nearest_and_reports_the_full_total(session, oracle, fx, limit):
    """limit (20 by default) keeps the nearest results while total_found counts all matches."""
    args, _ = range_case(oracle, fx.anchor_ll, target_m=RADIUS_M, cap=400)
    rows = oracle.near(fx.anchor_ll, args["radius_m"])
    assert len(rows) > LIMIT
    assert_range(oracle, range_search(session, **args, **({} if limit is None else {"limit": limit})), rows, limit or 20)


@pytest.mark.parametrize("unit", ["m", "km", "mi"])
def test_units_add_converted_distances_and_keep_metres(session, fx, unit):
    """units adds geodesic_distance_<unit> to every location of range_search and knn_search; m adds nothing."""
    for res in (range_search(session, fx.anchor_ll, radius_m=300, limit=10, units=unit), knn_search(session, fx.anchor_ll, limit=10, units=unit)):
        assert res["locations"]
        for loc in res["locations"]:
            assert_converted(loc, "geodesic_distance", unit)


# ---- filters ----

@pytest.mark.parametrize("case", ["column_key", "hstore_key", "value_list", "wildcard_any_case", "presence", "exclude_presence",
                                  "exclude_keeps_untagged", "include_exclude_same_key", "keys_and", "exclude_any_key"])
def test_filter_membership_matches_oracle(session, oracle, fx, case):
    """include/exclude select exactly the oracle's places: column and hstore keys, value lists OR, * patterns ignoring case,
    "*" as presence, include keys AND, a place matching any one exclude key dropped, NULL-safe excludes."""
    (key, val), = fx.cat_a.items()
    inc, exc = fx.cat_a, None
    if case == "column_key" and not all(key in cols for cols in oracle.columns.values()):
        pytest.skip(f"{key} is not a real column on every table")
    elif case == "hstore_key":
        if not fx.hstore_category:
            pytest.skip("no hstore-only tag with 5..150 entities near the anchor")
        inc = fx.hstore_category
    elif case == "value_list":
        inc = fx.value_list
    elif case == "wildcard_any_case":
        inc = {key: [val[:2].upper() + "*", "*" + val[-3:].upper()]}
    elif case in ("presence", "exclude_presence", "exclude_keeps_untagged", "keys_and", "exclude_any_key"):
        if not fx.partial_tag:
            pytest.skip("no tag carried by only some category members")
        (pkey, _), = fx.partial_tag.items()
        inc, exc = {
            "presence": ({**fx.cat_a, pkey: "*"}, None),
            "exclude_presence": (fx.cat_a, {pkey: "*"}),
            "exclude_keeps_untagged": (fx.cat_a, fx.partial_tag),
            "keys_and": ({**fx.cat_a, **fx.partial_tag}, None),
            "exclude_any_key": (fx.value_list, {key: fx.value_list[key][1], **fx.partial_tag}),
        }[case]
    elif case == "include_exclude_same_key":
        inc, exc = fx.value_list, {key: fx.value_list[key][1]}
    args, _ = range_case(oracle, fx.anchor_ll, inc, exc, cap=400)
    rows = oracle.near(fx.anchor_ll, args["radius_m"], inc, exc)
    if case == "wildcard_any_case":
        assert set(oracle.near(fx.anchor_ll, args["radius_m"], fx.cat_a)) <= set(rows)
    assert_range(oracle, range_search(session, limit=LIMIT, **args), rows, LIMIT, req=args)


def grammar_cases(fx):
    """name -> (include AST, exclude AST or None, include text or None = rendered): one case per operator, combinator and spelling."""
    if not (fx.cat_c and fx.partial_tag):
        pytest.skip("no second category key or partially carried tag near the anchor")
    (ka, va), = fx.cat_a.items()
    (kp, vp), = fx.partial_tag.items()
    A, C, P = T(ka, va), T(*next(iter(fx.cat_c.items()))), T(kp, vp)
    both, other = T(ka, *fx.value_list[ka]), T(ka, fx.value_list[ka][1])
    r = render
    return {
        "equals": (A, None, None),
        "any_of_values": (both, None, None),
        "tag_present": (T(ka, "*"), None, None),
        "contains": (T(ka, va[1:], op="~"), None, None),
        "contains_wildcard": (T(ka, va[:2] + "*" + va[-2:], op="~"), None, None),
        "not_equals": (AND(A, T(kp, vp, op="!=")), None, None),
        "not_equals_list": (AND(both, T(kp, vp, "no-such-value", op="!=")), None, None),
        "tag_absent": (AND(A, T(kp, "*", op="!=")), None, None),
        "or_across_keys": (OR(A, C), None, None),
        "and_binds_tighter_than_or": (OR(A, AND(C, P)), None, f"{r(A)} or {r(C)} and {r(P)}"),
        "parentheses": (AND(OR(A, C), P), None, f"({r(A)} or {r(C)}) and {r(P)}"),
        "not": (AND(A, NOT(P)), None, None),
        "not_group": (AND(A, NOT(OR(P, other))), None, None),
        "double_not": (A, None, f"not not {r(A)}"),
        "keywords_any_case": (OR(A, AND(C, NOT(P))), None, f"{r(A)} OR {r(C)} And NoT {r(P)}"),
        "whitespace": (OR(A, C), None, f"  {r(A)}\t or\n  {r(C)} "),
        "exclude_expression": (both, OR(other, P), None),
        "exclude_not": (A, NOT(P), None),
    }


GRAMMAR = ["equals", "any_of_values", "tag_present", "contains", "contains_wildcard", "not_equals", "not_equals_list", "tag_absent", "or_across_keys",
           "and_binds_tighter_than_or", "parentheses", "not", "not_group", "double_not", "keywords_any_case", "whitespace", "exclude_expression", "exclude_not"]


@pytest.mark.parametrize("case", GRAMMAR)
def test_filter_grammar_matches_oracle(session, oracle, fx, case):
    """Each operator (= | != ~ *), and / or / not with precedence and parentheses, keyword case and whitespace, in include and exclude,
    selects the places the oracle's filter AST selects."""
    inc, exc, text = grammar_cases(fx)[case]
    args, _ = range_case(oracle, fx.anchor_ll, inc, exc, cap=400)
    args = {**args, "include": text or args["include"]}
    assert_range(oracle, range_search(session, limit=LIMIT, **args), oracle.near(fx.anchor_ll, args["radius_m"], inc, exc), LIMIT, req=args)
    assert_knn(oracle, knn_search(session, fx.anchor_ll, limit=8, **{k: args[k] for k in ("include", "exclude")}), fx.anchor_ll, 8, inc, exc, req=args)


def test_precedence_changes_the_answer(oracle, fx):
    """The and-over-or and parentheses cases are not equal by accident: their expected sets differ."""
    inc, _, _ = grammar_cases(fx)["and_binds_tighter_than_or"]
    A, (_, (C, P)) = inc[1]
    assert oracle.near(fx.anchor_ll, RADIUS_M, inc) != oracle.near(fx.anchor_ll, RADIUS_M, AND(OR(A, C), P))


def test_quoted_value_with_spaces_and_escapes(session, oracle, fx):
    """A quoted value keeps its spaces (and parens, pipes, escaped quotes) as one literal value."""
    if not fx.spaced_name:
        pytest.skip("no uniquely named entity with a space in its name")
    eid, name = fx.spaced_name
    for text in (f'name="{name}"', f'name="{name}"|"no (such) | name"'):
        res = range_search(session, eid, radius_m=0, include=text, limit=LIMIT)
        assert_range(oracle, res, oracle.near(eid, 0, T("name", name)), LIMIT, req={"include": text, "exclude": "none"})
        assert eid in [x["entity_id"] for x in res["locations"]]
    for text in ('name="a \\"b\\" (c) | d"', 'name="x\\\\y"'):
        assert range_search(session, eid, radius_m=RADIUS_M, include=text)["count"] == 0


ECHO = [  # (given, canonical): keyword case, spacing and redundant parentheses normalise, needed parentheses and quotes stay
    ("amenity=cafe  OR shop=coffee", "amenity=cafe or shop=coffee"),
    ("(a=1 or b=2) AND c=3", "(a=1 or b=2) and c=3"),
    ("a=1 and (b=2 or c=3)", "a=1 and (b=2 or c=3)"),
    ("a=1 or (b=2 and c=3)", "a=1 or b=2 and c=3"),
    ("NOT (a=1 and b=2)", "not (a=1 and b=2)"),
    ('name="Main St"|x', 'name="Main St"|x'),
    ("a!=1|2", "a!=1|2"),
    ("k~text", "k~text"),
]


@pytest.mark.parametrize("given,canonical", ECHO)
def test_filter_echo_is_the_canonical_form(session, fx, given, canonical):
    """A search echoes how it read include and exclude, each only when present, joined by '; '."""
    ref = {"reference": place(fx.anchor_ll), "radius_m": 1, "limit": 1}
    g = call(session, "range_search", {**ref, "include": given, "exclude": "none"})["guidance"]
    assert f" Filter read as: include {canonical}." in g, g
    g = call(session, "knn_search", {"reference": place(fx.anchor_ll), "include": "x=1", "exclude": given, "limit": 1})["guidance"]
    assert f"Filter read as: include x=1; exclude {canonical}." in g, g
    g = call(session, "range_search", {**ref, "exclude": given})["guidance"]
    assert f"Filter read as: exclude {canonical}." in g and "include" not in g.split("Filter read as:")[1], g


PARSE_ERRORS = [  # (expression, 1-based position of the offending character, expected, found)
    ("amenity", 8, '"=", "!=" or "~" after key "amenity"', "end of input"),
    ("amenity=", 9, "a value (* = any)", "end of input"),
    ("amenity=a|", 11, "a value (* = any)", "end of input"),
    ("=cafe", 1, 'a tag key, "not" or "("', '"="'),
    ("amenity=cafe and", 17, 'a tag key, "not" or "("', "end of input"),
    ("(amenity=cafe", 14, '")", "and" or "or"', "end of input"),
    ("amenity=cafe shop=coffee", 14, '"and", "or" or end of input', '"s"'),
    ("amenity=cafe)", 13, '"and", "or" or end of input', '")"'),
    ('name="Main', 11, 'a closing "', "end of input"),
    ("a'b=x", 2, '"=", "!=" or "~" after key "a"', '"\'"'),
]


def parse_error(label, expr, pos, expected, found):
    return f"Invalid {label} at position {pos}: expected {expected}, found {found}.\n  {expr}\n  {' ' * (pos - 1)}^\n{FORM}"


@pytest.mark.parametrize("expr,pos,expected,found", PARSE_ERRORS, ids=[e[0] for e in PARSE_ERRORS])
def test_invalid_filter_names_position_expectation_and_form(session, fx, expr, pos, expected, found):
    """A malformed include or exclude fails with the position, what was expected, the expression with a caret and the Form line."""
    ref = place(fx.anchor_ll)
    for label, tool in (("include", "range_search"), ("exclude", "range_search"), ("include", "knn_search"), ("exclude", "knn_search"),
                        ("include", "knn_search_network")):
        text = error_of(session, tool, reference=ref, **{label: expr})
        assert parse_error(label, expr, pos, expected, found) in text, (tool, text)


def test_filters_must_be_strings(session, fx):
    """The old map form is rejected by the schema, not guessed at."""
    for arg in ("include", "exclude"):
        rejects(session, f"Expected string, received object at {arg}", "range_search", reference=place(fx.anchor_ll), **{arg: {"amenity": "cafe"}})


def test_filter_values_are_literals(session, fx):
    """SQL fragments, LIKE specials and regex characters in a value match literally (nothing), never break the query; keys are case-sensitive."""
    (key, val), = fx.cat_a.items()
    assert range_search(session, fx.anchor_ll, radius_m=RADIUS_M, include=render(fx.cat_a))["count"] > 0
    for inc in (f"{key}=\"x' OR '1'='1\"", f"{key}=%", f"{key}={val[:-1]}_", f"{key}=_{val[1:]}", f"{key}=.*", f"{key}=[a-z]*", f"{key.upper()}={val}"):
        assert range_search(session, fx.anchor_ll, radius_m=RADIUS_M, include=inc)["count"] == 0, inc


def test_cross_key_and_hint_when_nothing_matches(session, oracle, fx):
    """An 'and' over different keys that matches nothing says to use 'or' and names the keys."""
    if not fx.cat_c:
        pytest.skip("no second category key near the anchor")
    both = f"{render(fx.cat_a)} and {render(fx.cat_c)}"
    if oracle.near(fx.anchor_ll, RADIUS_M, AND(*[T(k, v) for k, v in {**fx.cat_a, **fx.cat_c}.items()])):
        pytest.skip("a place carries both categories")
    g = range_search(session, fx.anchor_ll, radius_m=RADIUS_M, include=both)["guidance"]
    keys = ", ".join({**fx.cat_a, **fx.cat_c})
    assert f'"and" requires every condition, but include mixes keys ({keys}); use "or" between alternatives.' in g
    assert '"and" requires' not in range_search(session, fx.anchor_ll, radius_m=RADIUS_M, include=f"{render(fx.cat_a)} or {render(fx.cat_c)}").get("guidance", "")


def test_real_column_filters_survive_a_non_default_search_path(oracle, fx, monkeypatch):
    """Column-backed filters still match when the server's search_path lists another schema first."""
    (key, _), = fx.cat_a.items()
    if not all(key in cols for cols in oracle.columns.values()):
        pytest.skip(f"{key} is not a real column on every table")
    args, _ = range_case(oracle, fx.anchor_ll, fx.cat_a, cap=400)
    monkeypatch.setenv("PGOPTIONS", "-c search_path=information_schema,public")
    gen = session_with()
    try:
        assert_range(oracle, range_search(next(gen), limit=LIMIT, **args), oracle.near(fx.anchor_ll, args["radius_m"], fx.cat_a), LIMIT, req=args)
    finally:
        next(gen, None)


@pytest.mark.parametrize("case", ["omitted", "null", "blank", "spaces", "none", "none_upper", "given", "include_key", "include_amenity"])
def test_default_exclude(session, oracle, fx, case):
    """Omitted, null or blank exclude drops DEFAULT_EXCLUDE (minus whole keys named in include) and says so; none (any case) keeps all;
    a given exclude replaces the default."""
    hw, am = T("highway", "*"), T("amenity", "*")
    include, exclude, want_exc = {
        "omitted": (None, None, default_exclude()),
        "null": (None, None, default_exclude()),
        "blank": (None, "", default_exclude()),
        "spaces": (None, "  ", default_exclude()),
        "none": (None, "none", None),
        "none_upper": (None, "NoNe", None),
        "given": (None, "building=*", T("building", "*")),
        "include_key": (hw, None, default_exclude(hw)),
        "include_amenity": (am, None, default_exclude(am)),
    }[case]
    is_default = case in ("omitted", "null", "blank", "spaces", "include_key", "include_amenity")
    args, _ = range_case(oracle, fx.anchor_ll, include, want_exc, target_m=200, cap=LIMIT)
    rows = oracle.near(fx.anchor_ll, args["radius_m"], include, want_exc)
    other = default_exclude() if case.startswith("include_") else None if is_default else default_exclude(include)  # the exclusion this case must not be
    assert rows and rows != oracle.near(fx.anchor_ll, args["radius_m"], include, other)
    sent = {"include": render(include), **({"exclude": None} if case == "null" else {} if exclude is None else {"exclude": exclude})}
    res = call(session, "range_search", {"reference": place(fx.anchor_ll), "radius_m": args["radius_m"], "limit": LIMIT, **sent})
    assert_range(oracle, res, rows, LIMIT, req=sent)
    assert (DEFAULT_NOTE in res.get("guidance", "")) == (is_default and res["count"] < LIMIT)
    knn = call(session, "knn_search", {"reference": place(fx.anchor_ll), **sent})
    assert_knn(oracle, knn, fx.anchor_ll, 5, include, want_exc, req=sent)


def test_multi_row_entity_is_one_place(session, oracle, fx):
    """An entity stored as several rows is found once, at its nearest row, and counted once in total_found."""
    if not fx.multirow_line:
        pytest.skip("no entity stored as several rows near the anchor")
    loc = call(session, "lookup", {"id": fx.multirow_line})["location"]
    inc = T("name", loc["name"])
    radius = 2 * loc["perimeter_m"]
    rows = oracle.near(fx.anchor_ll, radius, inc)
    assert fx.multirow_line in dict(rows)
    req = {"include": render(inc), "exclude": "none"}
    assert_range(oracle, range_search(session, fx.anchor_ll, radius_m=radius, limit=LIMIT, **req), rows, LIMIT, req=req)
    assert_knn(oracle, knn_search(session, fx.anchor_ll, limit=LIMIT, **req), fx.anchor_ll, LIMIT, inc, req=req)


# ---- knn_search ----

@pytest.mark.parametrize("kind", REF_KINDS)
@pytest.mark.parametrize("limit", [5, LIMIT])
def test_knn_search_is_the_exact_geodesic_top_k_for_every_reference_kind(session, oracle, fx, kind, limit):
    """Every reference kind gets its exact nearest places, sorted by distance, then area and id."""
    reference = reference_of(fx, kind)
    assert_knn(oracle, knn_search(session, reference, limit=limit), reference, limit)


def test_knn_search_from_references_far_off_the_data_is_exact(session, oracle, fx):
    """Planar index order is not geodesic order away from the data; the k nearest by metres are still returned, filtered or not."""
    for reference in far_references(fx):
        req = {"include": render(fx.cat_a), "exclude": "none"}
        assert_knn(oracle, knn_search(session, reference, limit=8, **req), reference, 8, fx.cat_a, req=req)
        assert_knn(oracle, knn_search(session, reference, limit=8), reference, 8)


def test_knn_search_filters_match_oracle(session, oracle, fx):
    """include and exclude apply before the nearest are taken."""
    (key, _), = fx.cat_a.items()
    exc = {key: fx.value_list[key][1]}
    req = {"include": render(fx.value_list), "exclude": render(exc, "or")}
    assert_knn(oracle, knn_search(session, fx.anchor["id"], limit=10, **req), fx.anchor["id"], 10, fx.value_list, exc, req=req)


def test_knn_search_limit_beyond_the_matches_returns_all_of_them(session, oracle, fx):
    """With fewer matches than limit, every match is returned."""
    (key, val), = fx.cat_a.items()
    named = [n for (n,) in oracle.q("SELECT name FROM planet_osm_point WHERE osm_id = ANY(%s::bigint[]) AND name IS NOT NULL",
                                    [[i.split(":")[1] for i in fx.a_ids if i.startswith("point:")]])]
    if not named:
        pytest.skip("no named category point")
    inc = {key: val, "name": named[0]}
    req = {"include": render(inc), "exclude": "none"}
    assert 0 < len(oracle.knn(fx.anchor_ll, LIMIT, inc)) < LIMIT
    assert_knn(oracle, knn_search(session, fx.anchor_ll, limit=LIMIT, **req), fx.anchor_ll, LIMIT, inc, req=req)


# ---- output ----

@pytest.mark.parametrize("kind", ["latlon", "polygon_entity"])
def test_bearing_follows_the_shortest_line_from_the_reference(session, oracle, fx, kind):
    """bearing_deg and its 8-point direction point along the shortest line from the reference shape; absent at distance 0; key order."""
    reference = reference_of(fx, kind)
    locs = range_search(session, reference, radius_m=150, limit=LIMIT)["locations"]
    assert any(x["geodesic_distance_m"] > 0 for x in locs) and (kind == "latlon" or any(x["geodesic_distance_m"] == 0 for x in locs))
    for x in locs:
        assert [k for k in x if k in LOCATION_KEYS] == [k for k in LOCATION_KEYS if k in x], list(x)
        if x["geodesic_distance_m"] == 0:
            assert "bearing_deg" not in x and "direction" not in x, x["entity_id"]
        else:
            assert x["bearing_deg"] == pytest.approx(oracle.bearing_to_entity(reference, x["entity_id"]), abs=2e-3), x["entity_id"]
            assert x["direction"] == compass_name(x["bearing_deg"], 8)


def test_guidance_says_how_to_widen_or_narrow(session, oracle, fx):
    """No match, a truncated range and a short knn each say what to change, then how the filter was read; the default exclude is named when it may hide places."""
    (key, _), = fx.cat_a.items()
    nothing = f"{key}=no-such-value"
    echo = f" Filter read as: include {nothing}."
    assert range_search(session, fx.anchor_ll, radius_m=10, include=nothing)["guidance"] == "No matches. Increase radius_m or loosen include/exclude." + echo
    assert knn_search(session, fx.anchor_ll, include=nothing)["guidance"] == "No matches. Loosen include/exclude." + echo
    assert call(session, "knn_search", {"reference": place(fx.anchor_ll), "include": nothing})["guidance"] == f"No matches. Loosen include/exclude.{echo} {DEFAULT_NOTE}"
    r = range_search(session, fx.anchor_ll, radius_m=RADIUS_M, include=render(fx.cat_a), limit=3)
    assert r["guidance"] == (f"Showing the nearest 3 of {r['total_found']}. Raise limit, narrow include/exclude, or reduce radius_m. "
                             f"Filter read as: include {render(fx.cat_a)}.")
    name = render(T("name", fx.named_point[1]))
    r = knn_search(session, fx.anchor_ll, include=name, limit=LIMIT)
    assert 0 < r["count"] < LIMIT and r["guidance"] == f"Only {r['count']} matches. Loosen include/exclude for more. Filter read as: include {name}."


# ---- arguments ----

def test_numeric_arguments_are_bounded(session, fx):
    """A negative radius_m and a limit outside 1..50 or fractional are rejected with the allowed range."""
    for tool, arg, bad, msg in (("range_search", "radius_m", -1, "radius_m must be a number >= 0"), ("range_search", "limit", 0, "limit must be an integer from 1 to 50"),
                                ("knn_search", "limit", LIMIT + 1, "limit must be an integer from 1 to 50"), ("knn_search", "limit", 1.5, "limit must be an integer")):
        rejects(session, msg, tool, reference=place(fx.anchor_ll), **{arg: bad})


def test_explicit_nulls_equal_omitted_arguments(session, fx):
    """null for any optional argument behaves exactly like omitting it (exclude null = the default exclude)."""
    ref = {"reference": place(fx.anchor_ll)}
    omitted = call(session, "range_search", ref), call(session, "knn_search", ref)
    nulls = (call(session, "range_search", {**ref, "radius_m": None, "include": None, "exclude": None, "limit": None, "units": None}),
             call(session, "knn_search", {**ref, "include": None, "exclude": None, "limit": None, "units": None}))
    assert nulls == omitted and omitted[1]["count"] == 5
