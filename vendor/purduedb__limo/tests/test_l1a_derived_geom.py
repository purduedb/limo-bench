"""L1a derived geometry: `make_geom` over entities and coordinates, `make_isochrone` from an origin.

Each result is a new tmp entity checked against SQL computed here from the stored geometries.
Run: python -m pytest tests/test_l1a_derived_geom.py -v
"""
import pytest

from harness import call, place, rejects, session  # noqa: F401

PROFILES = ("car", "bike", "foot")
TOL_M = 0.02  # the tool rounds coordinates to 7 dp
OPS = ("bbox", "convex_hull", "min_bounding_circle", "union", "intersection", "boundary")


def make_geom(session, op, *places):
    return call(session, "make_geom", {"op": op, "references": [place(p) for p in places]})


def make_isochrone(session, origin, **args):
    return call(session, "make_isochrone", {"origin": place(origin), **args})


def new_entity(what):
    return f"New entity created for this {what}; use its entity_id for further queries."


def made(op):
    return f"{new_entity(op)} Its lat/lon is {'its center' if op == 'min_bounding_circle' else 'its origin' if op == 'isochrone' else 'a point on it'}."


def inputs(fx, op):
    """References for a make_geom op: entities of every kind and a coordinate, overlapping polygons where the op needs them."""
    relation, way = fx.overlap_polygons
    (_, _), (_, point), _ = fx.route_trio
    return {
        "bbox": [fx.way_polygon, point], "convex_hull": [fx.way_polygon, fx.line, point], "min_bounding_circle": [fx.way_polygon, point],
        "union": [relation, way], "intersection": [relation, way], "boundary": [fx.way_polygon, fx.line, point],
    }[op]


# ---- make_geom ----

@pytest.mark.parametrize("op", OPS)
def test_make_geom_matches_the_oracle(session, oracle, fx, op):
    """Every op returns a tmp entity with the op's area and perimeter computed in SQL, its lat/lon on the shape
    (a circle's at its center), keys in order; a boundary is the outline of the areas only."""
    refs = inputs(fx, op)
    r = make_geom(session, op, *refs)
    area, perimeter = oracle.derived(op, refs)
    assert r["entity_id"].startswith(f"tmp:{op}-") and r["guidance"] == made(op)
    assert r.get("area_m2") == pytest.approx(area, abs=2e-3) and r.get("perimeter_m") == pytest.approx(perimeter, abs=2e-3)
    assert (op == "boundary") == (area is None)
    assert list(r) == ["entity_id", "lat", "lon", "perimeter_m"] + (["area_m2"] if area else []) + ["guidance"]
    assert (r.get("area_m2"), r.get("perimeter_m")) == pytest.approx(oracle.entity_measures(r["entity_id"]), abs=2e-3)
    at = {"lat": r["lat"], "lon": r["lon"]}
    if op == "min_bounding_circle":
        center = oracle.circle_center(refs)
        assert (at["lat"], at["lon"]) == pytest.approx((center["lat"], center["lon"]), abs=1e-7)
    else:
        assert oracle.distance_to_entity(at, r["entity_id"]) < TOL_M
    assert call(session, "lookup", {"id": r["entity_id"]})["location"]["lat"] == r["lat"]


def test_make_geom_union_covers_every_input_and_intersection_lies_in_all(session, oracle, fx):
    """The union covers each input; the intersection of three references is covered by each, so it folds over all of them."""
    relation, way = fx.overlap_polygons
    union = make_geom(session, "union", relation, way)["entity_id"]
    assert oracle.covers(union, relation) and oracle.covers(union, way)
    pair = make_geom(session, "intersection", relation, way)
    inside = {"lat": pair["lat"], "lon": pair["lon"]}
    three = make_geom(session, "intersection", relation, way, inside)
    assert "area_m2" not in three and oracle.distance(inside, three) < TOL_M


@pytest.mark.parametrize("op", ["bbox", "convex_hull", "min_bounding_circle"])
def test_make_geom_without_area_says_so(session, fx, op):
    """An enclosing op over one point is a point, over points on one parallel a line (a circle still has area)."""
    p = fx.anchor_ll
    r = make_geom(session, op, p)
    assert "area_m2" not in r and "perimeter_m" not in r and (r["lat"], r["lon"]) == pytest.approx((p["lat"], p["lon"]), abs=1e-7)
    assert r["guidance"] == f"{made(op)} References span no area, so the {op} is a point."
    r = make_geom(session, op, p, {"lat": p["lat"], "lon": p["lon"] + 0.01})
    assert (op == "min_bounding_circle") == ("area_m2" in r)
    if op != "min_bounding_circle":
        assert r["perimeter_m"] > 0 and r["guidance"].endswith(f"References span no area, so the {op} is a line.")


def test_make_geom_empty_shape_creates_no_entity(session, fx):
    """Disjoint references have no intersection, and references without an area no boundary: entity_id null, guidance says why."""
    (_, a), (_, b), _ = fx.route_trio
    assert make_geom(session, "intersection", a, b) == {"entity_id": None, "lat": None, "lon": None, "guidance": "References do not overlap, so the intersection is empty; no entity was created."}
    assert make_geom(session, "boundary", fx.line, a) == {"entity_id": None, "lat": None, "lon": None, "guidance": "References have no area, so the boundary is empty; no entity was created."}


def test_make_geom_checks_its_references(session, fx):
    """Too few or too many references, and an unknown reference named by its index, fail with a fix-it message."""
    relation, way = fx.overlap_polygons
    p = fx.anchor_ll
    rejects(session, "references holds 0 places; give 1 to 100", "make_geom", op="bbox", references=[])
    rejects(session, "references holds 101 places; give 1 to 100", "make_geom", op="bbox", references=[p] * 101)
    assert make_geom(session, "bbox", *[{"lat": p["lat"] + i * 1e-5, "lon": p["lon"]} for i in range(100)])["entity_id"].startswith("tmp:bbox-")
    rejects(session, "intersection needs at least 2 references", "make_geom", op="intersection", references=[place(relation)])
    rejects(session, r"Unrecognized entity_id .* in references\[1\]", "make_geom", op="union", references=[place(relation), {"entity_id": fx.named_point[1]}])
    for _ in range(5):
        rejects(session, r"in references\[0\]", "make_geom", op="union", references=[{"entity_id": "point:1"}, {"entity_id": "tmp:route-00000000"}])


def test_expired_tmp_entity_is_unrecognized_with_a_hint(session, oracle, fx):
    """A tmp entity older than 1 hour no longer resolves, and the error says it expired."""
    tmp = make_geom(session, "bbox", fx.anchor_ll, fx.route_trio[1][1])["entity_id"]
    oracle.age_tmp(tmp, 61)
    hint = "A tmp:\\* entity expires 1 hour after creation; call the tool that made it again."
    rejects(session, f"Unrecognized id .*{hint}", "lookup", id=tmp)
    rejects(session, f"in references\\[0\\].*{hint}", "make_geom", op="bbox", references=[place(tmp)])


# ---- make_isochrone ----

@pytest.mark.parametrize("profile", PROFILES)
def test_make_isochrone_distance_budget_is_an_area_around_the_origin(session, oracle, fx, profile):
    """A distance isochrone contains its origin, stays within the straight-line disc of that distance, reports its own measures
    and has the origin as its lat/lon."""
    (_, origin), *_ = fx.route_trio
    r = make_isochrone(session, origin, profile=profile, max_distance_m=1000)
    assert list(r) == ["entity_id", "lat", "lon", "perimeter_m", "area_m2", "guidance"] and r["guidance"] == made("isochrone")
    assert r["entity_id"].startswith("tmp:isochrone-") and r["area_m2"] > 0
    assert oracle.covers(r["entity_id"], origin)
    assert oracle.within_radius(origin, 1100, r["entity_id"])  # network distance is never below straight-line distance
    assert (r["area_m2"], r["perimeter_m"]) == pytest.approx(oracle.entity_measures(r["entity_id"]), abs=2e-3)
    loc = call(session, "lookup", {"id": r["entity_id"]})["location"]
    assert (r["lat"], r["lon"]) == (loc["lat"], loc["lon"]) == pytest.approx((origin["lat"], origin["lon"]), abs=1e-7)


def test_make_isochrone_larger_time_budget_covers_the_smaller(session, oracle, fx):
    """A longer time budget reaches everything a shorter one does, from an entity origin."""
    (origin, _), *_ = fx.route_trio
    small, large = (make_isochrone(session, origin, profile="foot", max_duration_min=m) for m in (5, 15))
    assert oracle.covers(large["entity_id"], small["entity_id"]) and large["area_m2"] > small["area_m2"]


def test_make_isochrone_time_budget_is_travel_time(session, oracle, fx):
    """A foot isochrone reaches the end of a foot route when its budget exceeds the route's duration, and not at half of it."""
    (origin, _), (dest, _), _ = fx.route_trio
    r = call(session, "route", {"origin": place(origin), "destination": place(dest), "profile": "foot"})
    for factor, reached in ((1.1, True), (0.5, False)):
        iso = make_isochrone(session, origin, profile="foot", max_duration_min=factor * r["duration_min"])
        assert oracle.covers_route_end(iso["entity_id"], r["entity_id"]) is reached, factor


def test_make_isochrone_default_profile_is_car(session, fx):
    """Omitting or nulling profile reaches the same area as car."""
    (_, origin), *_ = fx.route_trio
    car = make_isochrone(session, origin, profile="car", max_distance_m=800)
    for args in ({}, {"profile": None}):
        assert make_isochrone(session, origin, max_distance_m=800, **args)["area_m2"] == car["area_m2"]


def test_make_isochrone_area_origin_uses_a_point_on_it(session, oracle, fx):
    """An area origin starts from its lat/lon, a point on it, which is the isochrone's lat/lon."""
    r = make_isochrone(session, fx.way_polygon, profile="foot", max_distance_m=500)
    origin = call(session, "lookup", {"id": fx.way_polygon})["location"]
    loc = call(session, "lookup", {"id": r["entity_id"]})["location"]
    assert (r["lat"], r["lon"]) == (loc["lat"], loc["lon"]) == (origin["lat"], origin["lon"])


def test_make_isochrone_checks_its_budget_and_origin(session, fx):
    """Both budgets or neither, a budget not above 0, and an origin off the road network fail with a fix-it message."""
    (_, origin), *_ = fx.route_trio
    one = "Pass exactly one of max_duration_min or max_distance_m."
    rejects(session, one, "make_isochrone", origin=origin, max_duration_min=10, max_distance_m=1000)
    rejects(session, one, "make_isochrone", origin=origin)
    rejects(session, one, "make_isochrone", origin=origin, max_duration_min=None, max_distance_m=None)
    rejects(session, r"max_duration_min must be a number > 0 \(got 0\)", "make_isochrone", origin=origin, max_duration_min=0)
    rejects(session, r"max_distance_m must be a number > 0 \(got -5\)", "make_isochrone", origin=origin, max_distance_m=-5)
    rejects(session, "No foot isochrone from origin", "make_isochrone", origin={"lat": -60, "lon": 100}, profile="foot", max_distance_m=500)
