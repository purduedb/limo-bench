"""Composition: the multi-tool chains an LLM issues, each layer's output feeding the next as its entity_id or coordinates.

Only the hand-off is checked here (against the SQL oracle); each tool's own mechanics are tested in its layer.
Run: python -m pytest tests/test_chains.py -v
"""
import pytest

from harness import call, place, ref, session  # noqa: F401
from oracle import CAP, assert_join, assert_knn, assert_range, range_case, render


LIMIT = 50  # the search tools' max limit


def search(session, tool, args, **extra):
    """range_search / knn_search on range_case args: reference as a place, include and exclude as given."""
    return call(session, tool, {"reference": place(args["reference"]), "include": args["include"], "exclude": args["exclude"], **extra})


def geocode_top(session, query):
    return call(session, "geocode", {"query": query, "limit": 1})["locations"][0]


def test_geocode_then_range_search(session, oracle, fx):
    """'What is near X': the entity_id geocode returns is a reference range_search resolves to the oracle's exact set."""
    eid, name = fx.named_building
    here = ref(session, name)
    assert here == eid
    args, _ = range_case(oracle, here, target_m=150, cap=400)
    res = search(session, "range_search", args, radius_m=args["radius_m"], limit=LIMIT)
    assert_range(oracle, res, oracle.near(here, args["radius_m"]), LIMIT, req=args)


def test_geocode_then_route_then_lookup(session, oracle, fx):
    """'How do I get from X to Y': two geocoded places route between their own points and the route can be looked up."""
    a, b = geocode_top(session, fx.named_point[1]), geocode_top(session, fx.named_building[1])
    r = call(session, "route", {"origin": place(a["entity_id"]), "destination": place(b["entity_id"]), "profile": "foot"})
    length, start_gap, end_gap, *_ = oracle.route_facts(r["entity_id"], a, b)
    assert r["distance_m"] == pytest.approx(length, rel=5e-3) and r["distance_m"] >= oracle.distance(a, b) - 1
    assert start_gap < 100 and end_gap < 100
    loc = call(session, "lookup", {"id": r["entity_id"]})["location"]
    assert loc["entity_id"] == r["entity_id"] and loc["perimeter_m"] == pytest.approx(length, abs=2e-3)


def test_geocode_then_knn_search_network(session, fx):
    """'Nearest X by walking': the network knn from a geocoded place reports the route distance to its result."""
    here = geocode_top(session, fx.named_point[1])
    r = call(session, "knn_search_network", {"reference": place(here["entity_id"]), "profile": "foot", "limit": 1, "include": render(fx.cat_a)})
    best = r["locations"][0]
    direct = call(session, "route", {"origin": place(here["entity_id"]), "destination": {"lat": best["lat"], "lon": best["lon"]}, "profile": "foot"})
    assert r["count"] == 1 and best["distance_m"] == pytest.approx(direct["distance_m"], abs=1e-2)


def test_reverse_geocode_then_bearing(session, oracle, fx):
    """'Which way is the nearest place': bearing from a point to the place reverse_geocode returned follows the formula."""
    nearest = call(session, "reverse_geocode", {"point": fx.anchor_ll, "limit": 1})["locations"][0]
    r = call(session, "bearing", {"origin": fx.anchor_ll, "destination": place(nearest["entity_id"])})
    want = oracle.azimuth(fx.anchor_ll, {"lat": nearest["lat"], "lon": nearest["lon"]})
    assert r["bearing_deg"] == pytest.approx(want, abs=2e-3)


def search_ids(session, oracle, reference, include, origin=None):
    """The entity_ids of range_search around reference (radius 0, exclude none) for a category, checked against the oracle (origin: an isochrone's)."""
    req = {"include": render(include), "exclude": "none"}
    res = call(session, "range_search", {"reference": place(reference), "limit": LIMIT, **req})
    assert_range(oracle, res, oracle.near(reference, 0, include, origin=origin), LIMIT, req=req, noted=bool(origin))
    return [loc["entity_id"] for loc in res["locations"]]


def join_of_searches(session, oracle, reference, a, b, origin=None):
    """'Pair A with B inside reference': both sets from range_search, joined at a distance from their own quartile; equals the oracle's join."""
    a_ids, b_ids = (search_ids(session, oracle, reference, include, origin) for include in (a, b))
    ds = sorted(oracle.dist(a_ids, b_ids).values())
    d = oracle.gap_threshold(ds, ds[len(ds) // 4])
    res = call(session, "spatial_join", {"set_a": a_ids, "set_b": b_ids, "max_distance_m": d, "limit": LIMIT})
    assert_join(oracle, res, oracle.join(a_ids, b_ids, "inner", d), LIMIT)


def test_geocode_then_spatial_join_inside_an_area(session, oracle, fx):
    """'Pair A with B inside area X': the geocoded area polygon bounds both searches, whose entity_ids join like the oracle says."""
    area = fx.area
    found = call(session, "geocode", {"query": area["name"], "limit": 10})["locations"]
    assert area["id"] in [loc["entity_id"] for loc in found]
    join_of_searches(session, oracle, area["id"], area["a"], area["b"])


# ---- derived geometries as references ----

def route_between(session, fx):
    (a, _), (b, _), _ = fx.route_trio
    return call(session, "route", {"origin": place(a), "destination": place(b), "profile": "car"})["entity_id"]


def bbox_of(session, fx):
    (_, a), (_, b), _ = fx.route_trio
    return call(session, "make_geom", {"op": "bbox", "references": [a, b]})["entity_id"]


def isochrone_around(session, fx):
    return call(session, "make_isochrone", {"origin": place(fx.anchor["id"]), "profile": "foot", "max_duration_min": 10})["entity_id"]


MAKERS = {"route": route_between, "bbox": bbox_of, "isochrone": isochrone_around}


@pytest.mark.parametrize("maker", MAKERS)
def test_derived_entity_is_a_reference_for_lookup_range_and_knn(session, oracle, fx, maker):
    """A route, bbox or isochrone entity looks up with its own measures and is searched by its true geometry;
    an isochrone is measured, ranked and given bearings from its origin."""
    tmp = MAKERS[maker](session, fx)
    loc = call(session, "lookup", {"id": tmp})["location"]
    area, perimeter = oracle.entity_measures(tmp)
    assert loc["entity_id"] == tmp and loc.get("area_m2") == pytest.approx(area, abs=2e-3) and loc.get("perimeter_m") == pytest.approx(perimeter, abs=2e-3)
    origin = {"lat": loc["lat"], "lon": loc["lon"]} if maker == "isochrone" else None
    args, _ = range_case(oracle, tmp, fx.cat_a, target_m=200, cap=400)
    for radius in (0, args["radius_m"]):
        res = search(session, "range_search", args, radius_m=radius, limit=LIMIT)
        assert_range(oracle, res, oracle.near(tmp, radius, fx.cat_a, origin=origin), LIMIT, req=args, noted=bool(origin))
        for x in res["locations"] if origin else ():
            if x["geodesic_distance_m"] > 0:
                assert x["bearing_deg"] == pytest.approx(oracle.bearing_to_entity(origin, x["entity_id"]), abs=2e-3), x["entity_id"]
    assert_knn(oracle, search(session, "knn_search", args), origin or tmp, 5, fx.cat_a, req=args, noted=bool(origin))


def test_isochrone_sets_feed_a_spatial_join(session, oracle, fx):
    """'Pair A with B, both within a 10-minute walk': an isochrone bounds both searches, whose entity_ids join like the oracle says."""
    iso = isochrone_around(session, fx)
    a_ids, b_ids = ([i for i, _ in oracle.near(iso, 0, include)] for include in (fx.cat_a, fx.cat_b))
    if not (0 < len(a_ids) < CAP and 0 < len(b_ids) < CAP):
        pytest.skip("the 10-minute walk holds no 1..99 entities of each category")
    loc = call(session, "lookup", {"id": iso})["location"]
    join_of_searches(session, oracle, iso, fx.cat_a, fx.cat_b, origin={"lat": loc["lat"], "lon": loc["lon"]})


def test_isochrones_intersect_into_the_area_reachable_from_both(session, oracle, fx):
    """'Reachable within a 10-minute drive from both A and B': make_geom intersects the two isochrones' areas, as SQL does."""
    (a, _), (b, _), _ = fx.route_trio
    isos = [call(session, "make_isochrone", {"origin": place(p), "profile": "car", "max_duration_min": 10})["entity_id"] for p in (a, b)]
    r = call(session, "make_geom", {"op": "intersection", "references": [place(i) for i in isos]})
    area, _ = oracle.derived("intersection", isos)
    assert area > 0 and r["area_m2"] == pytest.approx(area, abs=2e-3)


def test_derived_entity_as_a_route_end_uses_its_lat_lon(session, oracle, fx):
    """bearing from an isochrone starts at its origin and from a min_bounding_circle at its center."""
    (_, a), (_, b), (_, c) = fx.route_trio
    iso = call(session, "make_isochrone", {"origin": a, "profile": "foot", "max_distance_m": 500})["entity_id"]
    circle = call(session, "make_geom", {"op": "min_bounding_circle", "references": [a, b]})
    for tmp, start in ((iso, a), (circle["entity_id"], {"lat": circle["lat"], "lon": circle["lon"]})):
        r = call(session, "bearing", {"origin": place(tmp), "destination": c})
        assert r["bearing_deg"] == pytest.approx(oracle.azimuth(start, c), abs=2e-3)


# Tools reading a PlaceRef's lat/lon, with their place arguments; range_search, knn_search and spatial_join sets read its shape.
LAT_LON_TOOLS = {"bearing": ("origin", "destination"), "route": ("origin", "destination"), "knn_search_network": ("reference",), "make_isochrone": ("origin",)}
TOOL_ARGS = {
    "bearing": {}, "route": {"profile": "foot"}, "make_isochrone": {"profile": "foot", "max_distance_m": 300},
    "knn_search_network": {"profile": "foot", "limit": 1}, "range_search": {"radius_m": 0}, "knn_search": {"limit": 1},
}


@pytest.mark.parametrize("tool", TOOL_ARGS)
def test_place_reference_note(session, fx, tool):
    """A lat/lon tool says it used a point on a line or polygon, or an isochrone's origin; range_search and knn_search say an isochrone
    reference is measured from its origin; a point, or a line or polygon read by its shape, adds no note."""
    iso = isochrone_around(session, fx)
    lat_lon = tool in LAT_LON_TOOLS
    args = dict(TOOL_ARGS[tool], **({"origin": fx.route_trio[1][1], "destination": fx.route_trio[1][1]} if tool in ("bearing", "route") else {}))
    if "search" in tool:
        args |= {"include": render(fx.cat_a), "exclude": "none"}
    for arg in LAT_LON_TOOLS.get(tool, ("reference",)):
        label = arg.capitalize()
        area = f"{label} is a line or polygon, so a point on it was used." if lat_lon else None
        notes = {
            fx.anchor["id"]: None, fx.line: area, fx.way_polygon: area,
            iso: f"{label} is an isochrone, so its origin was used." if lat_lon
            else "Reference is an isochrone, so distances, bearings and order are measured from its origin.",
        }
        for eid, note in notes.items():
            g = call(session, tool, {**args, arg: place(eid)}).get("guidance", "")
            got = [s for s in (" is a line or polygon, ", " is an isochrone, ") if s in g]
            assert (note in g and len(got) == 1) if note else not got, (arg, eid, g)
