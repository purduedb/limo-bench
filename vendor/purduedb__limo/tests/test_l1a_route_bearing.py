"""L1a two-place tools: `route`, `bearing`, `knn_search_network`.

Their places are entity ids or coordinates as geocode and reverse_geocode return them (L0, assumed verified).
Expectations come from independent geodesy (geo.py) and SQL over the stored route geometry.
Run: python -m pytest tests/test_l1a_route_bearing.py -v
"""
import contextlib
import http.server
import itertools
import json
import math
import os
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest

import geo
from fixtures import far_references
from harness import assert_converted, call, place, ref, rejects, session, session_with  # noqa: F401
from oracle import render

PROFILES = ("car", "bike", "foot")
GH_URL = os.environ.get("GRAPHHOPPER_URL", "http://localhost:8989")  # the real GraphHopper, also while a test stubs it
MAX_SPEED_M_PER_MIN = {"car": 2200, "bike": 600, "foot": 150}  # above any real speed, below a unit or coordinate mix-up


def route(session, a, b, **args):
    return call(session, "route", {"origin": place(ref(session, a)), "destination": place(ref(session, b)), **args})


def bearing(session, a, b, **args):
    return call(session, "bearing", {"origin": place(ref(session, a)), "destination": place(ref(session, b)), **args})


# ---- route ----

@pytest.mark.parametrize("profile", PROFILES)
def test_route_entity_is_the_route_line(session, oracle, fx, profile):
    """The route entity is a line from origin to destination whose length is distance_m, anchored at a point on it."""
    (a, al), (b, bl), _ = fx.route_trio
    r = route(session, a, b, profile=profile)
    length, start_gap, end_gap, anchor_gap, anchor_offset = oracle.route_facts(r["entity_id"], al, bl)
    assert r["entity_id"].startswith("tmp:route-") and "perimeter_m" not in r
    assert r["distance_m"] == pytest.approx(length, rel=5e-3)
    assert r["geodesic_distance_m"] == pytest.approx(oracle.distance(al, bl), abs=2e-3)
    assert r["distance_m"] >= r["geodesic_distance_m"] - 1
    assert 0 < r["duration_min"] and r["distance_m"] / r["duration_min"] <= MAX_SPEED_M_PER_MIN[profile]
    assert start_gap < 5 and end_gap < 100  # the router snaps endpoints onto the network
    assert anchor_gap < 1 and anchor_offset < 1
    assert call(session, "lookup", {"id": r["entity_id"]})["location"]["perimeter_m"] == pytest.approx(length, abs=2e-3)


def gh_path(a, b, profile, instructions=False):
    """GraphHopper's own path from a to b ({lat, lon}), queried independently of server.ts; None when it has no route."""
    q = urllib.parse.urlencode([("point", f"{a['lat']},{a['lon']}"), ("point", f"{b['lat']},{b['lon']}"), ("profile", profile),
                                ("points_encoded", "false"), ("instructions", str(instructions).lower())])
    try:
        with urllib.request.urlopen(f"{GH_URL}/route?{q}") as resp:
            return json.load(resp)["paths"][0]
    except urllib.error.HTTPError:
        return None


def gh_summary(a, b, profile):
    """Expected route summary from GraphHopper turn instructions, recomputed independently of server.ts."""
    steps = gh_path(a, b, profile, instructions=True)["instructions"]
    def runs(pairs):
        out = []
        for name, d in pairs:
            if out and out[-1][0] == name:
                out[-1][1] += d
            else:
                out.append([name, d])
        return out
    stretches = runs([s for s in runs((s.get("street_name") or "", s["distance"]) for s in steps) if s[1] >= 20])
    if not stretches:
        return None
    middle = max(stretches[1:-1], key=lambda t: t[1], default=None)
    parts = [stretches[0], middle, stretches[-1] if len(stretches) > 1 else None]
    return " ... ".join(f"{n or 'unnamed path'} ({math.floor(d + 0.5)} m)" for n, d in filter(None, parts))


@pytest.mark.parametrize("profile", PROFILES)
def test_route_summary_matches_graphhopper_instructions(session, fx, profile):
    """summary equals the rule applied to GraphHopper's own instructions, and is well-formed and consistent with distance_m."""
    for (_, al), (_, bl) in itertools.combinations(fx.route_trio, 2):
        r = route(session, al, bl, profile=profile)
        assert r.get("summary") == gh_summary(al, bl, profile)
        if "summary" in r:
            assert re.fullmatch(r"(.+ \(\d+ m\))( \.\.\. .+ \(\d+ m\)){0,2}", r["summary"])
            listed = [int(m) for m in re.findall(r"\((\d+) m\)", r["summary"])]
            assert min(listed) >= 20 and sum(listed) <= r["distance_m"] + 1


@pytest.mark.parametrize("profile", PROFILES)
def test_route_to_a_farther_place_is_longer_and_slower(session, fx, profile):
    """For one profile, a destination at least twice as far takes more distance and time."""
    (a, _), (b, _), (c, _) = fx.route_trio
    short, long = route(session, a, b, profile=profile), route(session, a, c, profile=profile)
    assert long["distance_m"] > short["distance_m"] and long["duration_min"] > short["duration_min"]


def test_route_entity_and_coordinate_places_agree(session, fx):
    """A place given as entity_id or as its coordinates routes the same."""
    (a, al), (b, bl), _ = fx.route_trio
    by_id, by_ll, mixed = route(session, a, b), route(session, al, bl), route(session, a, bl)
    assert by_ll["distance_m"] == pytest.approx(by_id["distance_m"], abs=1)
    assert mixed["distance_m"] == pytest.approx(by_id["distance_m"], abs=1)


def test_route_default_profile_is_car(session, fx):
    """Omitting or nulling profile routes by car."""
    (a, _), (b, _), _ = fx.route_trio
    car = route(session, a, b, profile="car")
    for args in ({}, {"profile": None}):
        assert route(session, a, b, **args)["distance_m"] == car["distance_m"]


@pytest.mark.parametrize("unit", ["m", "km", "mi"])
def test_route_units_add_converted_distance(session, fx, unit):
    """units adds distance_<unit> and geodesic_distance_<unit> beside their metres values."""
    (a, _), (b, _), _ = fx.route_trio
    r = route(session, a, b, units=unit)
    assert_converted(r, "distance", unit)
    assert_converted(r, "geodesic_distance", unit)


def test_route_unroutable_place_is_an_error(session, fx):
    """A destination far outside the road network is an error, not a bogus route."""
    (a, _), _, _ = fx.route_trio
    rejects(session, "No car route", "route", origin=place(a), destination={"lat": -60, "lon": 100})


GH_DOWN = r"^tool \w+ returned isError=true: GraphHopper \(routing\) is unreachable or failing\. Retry the call shortly\.$"


class StubServer(http.server.ThreadingHTTPServer):
    request_queue_size = 64  # the server keeps 20 requests in flight; a short backlog resets connections and Node on macOS then crashes
    daemon_threads = True


class GraphHopperStub(http.server.BaseHTTPRequestHandler):
    """Answers every request with answer(path) = (status, body)."""
    answer = None

    def do_GET(self):
        status, body = type(self).answer(self.path)
        self.send_response(status)
        self.end_headers()
        self.wfile.write(body.encode())

    def log_message(self, *_):
        pass


@contextlib.contextmanager
def graphhopper_stub(monkeypatch, answer):
    """(session, stub): a server.ts session whose GraphHopper is a local stub answering with answer(path)."""
    GraphHopperStub.answer = staticmethod(answer)
    stub = StubServer(("127.0.0.1", 0), GraphHopperStub)
    threading.Thread(target=stub.serve_forever, daemon=True).start()
    monkeypatch.setenv("GRAPHHOPPER_URL", f"http://127.0.0.1:{stub.server_port}")
    gen = session_with()
    try:
        yield next(gen), stub
    finally:
        next(gen, None)
        stub.shutdown()
        stub.server_close()


def test_graphhopper_outage_is_a_retry_error_not_no_route(fx, monkeypatch):
    """A failing (5xx), garbled (non-JSON) or unreachable GraphHopper fails every routing tool with a retry error, never as no route."""
    (a, _), (b, _), _ = fx.route_trio
    calls = {"route": {"origin": place(a), "destination": place(b)}, "make_isochrone": {"origin": place(a), "max_distance_m": 500},
             "knn_search_network": {"reference": place(a)}}
    replies = iter([(503, "busy"), (200, "<html>")])
    reply = next(replies)
    with graphhopper_stub(monkeypatch, lambda path: reply) as (s, stub):
        for _ in range(3):
            for tool, args in calls.items():
                rejects(s, GH_DOWN, tool, **args)
            reply = next(replies, None)
            if reply is None:
                stub.shutdown()
                stub.server_close()


def test_route_units_default_to_metres(session, fx):
    """Omitting or nulling units adds no converted distance."""
    (a, _), (b, _), _ = fx.route_trio
    for args in ({}, {"units": None}):
        assert not {"distance_km", "distance_mi"} & route(session, a, b, **args).keys()


@pytest.mark.parametrize("tool", ["route", "bearing"])
def test_places_must_be_valid_entity_ids_or_coordinates(session, fx, tool):
    """Free text, an unknown id, out-of-range or swapped coordinates, and mixed keys are rejected with a fix-it message."""
    (a, _), _, _ = fx.route_trio
    bad = [
        ({"entity_id": fx.named_point[1]}, "Unrecognized entity_id"),
        ({"entity_id": "point:999999999999"}, "Unrecognized entity_id"),
        ({"entity_id": "point:92233720368547758070"}, "Unrecognized entity_id .* in origin"),
        ({"lat": 95, "lon": 10}, "origin.lat"),
        ({"lat": -120.5, "lon": 40.4}, "look swapped"),
        ({"lat": 95, "lon": 120}, r"origin\.lat must be a number in \[-90, 90\] \(got 95\)\. Pass \{lat, lon\} in decimal degrees\.$"),
        ({"entity_id": a, "lat": 1, "lon": 1}, ".+"),
    ]
    for origin, pattern in bad:
        rejects(session, pattern, tool, origin=origin, destination=fx.anchor_ll)


# ---- bearing ----

def rep(oracle, place):
    """(lat, lon) of an entity's representative point, or of a coordinate place."""
    if isinstance(place, dict):
        return place["lat"], place["lon"]
    x, y = oracle.reps([place])[place]
    return y, x


def test_bearing_matches_the_geodesic_azimuth(session, oracle, fx):
    """bearing_deg and direction agree with the geodesic azimuth for entity and coordinate places (an entity is measured from its lat/lon)."""
    (a, al), (b, bl), (c, cl) = fx.route_trio
    for origin, dest in ((a, b), (a, bl), (al, b), (bl, cl), (c, a)):
        r = bearing(session, origin, dest)
        (olat, olon), (dlat, dlon) = rep(oracle, origin), rep(oracle, dest)
        p, q = {"lat": olat, "lon": olon}, {"lat": dlat, "lon": dlon}
        assert r["bearing_deg"] == pytest.approx(oracle.azimuth(p, q), abs=2e-3)
        assert r["direction"] == geo.compass_name(r["bearing_deg"], 8)
        assert r["geodesic_distance_m"] == pytest.approx(oracle.distance(p, q), abs=2e-3)


def test_bearing_rose_names_switch_at_every_sector_edge(session, oracle, fx):
    """Just either side of every 16-point sector edge, rose 4, 8 and 16 name the sector the bearing falls in."""
    lat, lon = fx.anchor["lat"], fx.anchor["lon"]
    for edge in range(0, 64):
        for off in (-0.3, 0.3):
            dest = dict(zip(("lat", "lon"), geo.destination(lat, lon, edge * 5.625 + off, 1000)))
            aimed = edge * 5.625 + off
            for rose in (4, 8, 16):
                r = bearing(session, fx.anchor_ll, dest, rose=rose)
                assert r["bearing_deg"] == pytest.approx(oracle.azimuth(fx.anchor_ll, dest), abs=2e-3)
                assert r["bearing_deg"] == pytest.approx(aimed % 360, abs=0.2)  # spherical aim vs ellipsoidal bearing, under the 0.3 offset
                assert r["direction"] == geo.compass_name(aimed % 360, rose), (edge, off, rose)


def test_bearing_just_west_of_north_is_0_not_360(session, fx):
    """A bearing that rounds up to 360 is reported as 0."""
    dest = dict(zip(("lat", "lon"), geo.destination(fx.anchor["lat"], fx.anchor["lon"], 360 - 1e-5, 1000)))
    assert bearing(session, fx.anchor_ll, dest)["bearing_deg"] == 0


def test_bearing_default_rose_is_8(session, fx):
    """Omitting or nulling rose names the 8-point compass: a bearing of 45 degrees is NE, not N or E."""
    dest = dict(zip(("lat", "lon"), geo.destination(fx.anchor["lat"], fx.anchor["lon"], 45, 1000)))
    for args in ({}, {"rose": None}):
        assert bearing(session, fx.anchor_ll, dest, **args)["direction"] == "NE"


def test_bearing_same_place_is_null_with_guidance(session, fx):
    """Origin equal to destination has no bearing and zero distance."""
    r = bearing(session, fx.anchor_ll, fx.anchor_ll)
    assert r == {"bearing_deg": None, "direction": None, "geodesic_distance_m": 0, "guidance": "Origin and destination coincide; no bearing."}


# ---- knn_search_network ----

def network_knn(session, reference, **args):
    return call(session, "knn_search_network", {"reference": place(reference), "exclude": "none", **args})


def network_pool(oracle, reference, profile, include=None):
    """Every one of the 100 nearest matches by straight line routed from the reference's lat/lon: [(entity_id, geodesic m, path or None)],
    the 101st nearest's geodesic m (None if fewer) and the origin's lat/lon."""
    origin = dict(zip(("lat", "lon"), rep(oracle, reference)))
    nearest = oracle.knn(origin, 101, include, exclude={})
    reps = oracle.reps([i for i, _ in nearest[:100]])
    with ThreadPoolExecutor(8) as ex:
        paths = list(ex.map(lambda i: gh_path(origin, {"lat": reps[i][1], "lon": reps[i][0]}, profile), [i for i, _ in nearest[:100]]))
    return [(i, d, p) for (i, d), p in zip(nearest, paths)], (nearest[100][1] if len(nearest) > 100 else None), origin


def expected_network_knn(oracle, reference, profile, k, include=None):
    """(entity_id, distance_m, duration_min) of the k nearest by route among the pool, the left-out note and whether the cap note is due."""
    pool, next_d, origin = network_pool(oracle, reference, profile, include)
    routed = sorted(((p["distance"], p["time"] / 60000, i, d, p) for i, d, p in pool if p), key=lambda r: (r[0], r[1], r[2]))[:k]
    left_out = sum(1 for _, d, p in pool if not p and d < routed[-1][3]) if routed else 0
    note = left_out and f"Left out {left_out} {'match' if left_out == 1 else 'matches'} nearer by straight line than the last result: no {profile} route."
    snap_o = oracle.distance(origin, dict(zip(("lon", "lat"), routed[-1][4]["snapped_waypoints"]["coordinates"][0]))) if routed else 0
    capped = next_d is not None and bool(routed) and (len(routed) < k or next_d - snap_o <= routed[-1][0])
    return [(i, dist, t) for dist, t, i, _, _ in routed], note, capped


CAP_NOTE = "Routed only the 100 nearest matches by straight line"


@pytest.mark.parametrize("profile", ["foot", "car"])
def test_knn_search_network_is_the_k_nearest_by_route_among_the_100_nearest(session, oracle, fx, profile):
    """The result is exactly the k nearest by route among the 100 nearest matches by straight line, each route from the reference's lat/lon
    and geodesic_distance_m from that lat/lon; a route may be shorter than the straight line (both ends snap onto the network), and
    left-out unroutable matches and the candidate cap are noted exactly when due."""
    shorter = False
    for reference, k, include in [(fx.anchor["id"], 3, fx.cat_a), (fx.way_polygon, 3, fx.cat_a), (fx.anchor["id"], 1, None), (fx.anchor["id"], 10, None)] + [(f, 5, None) for f in far_references(fx)]:
        r = network_knn(session, reference, profile=profile, limit=k, **({"include": render(include)} if include else {}))
        want, note, capped = expected_network_knn(oracle, reference, profile, k, include)
        assert [(x["entity_id"], x["distance_m"], x["duration_min"]) for x in r["locations"]] == [(i, pytest.approx(d, abs=1e-3), pytest.approx(t, abs=1e-3)) for i, d, t in want], reference
        origin = dict(zip(("lat", "lon"), rep(oracle, reference)))
        for loc in r["locations"]:
            assert loc["geodesic_distance_m"] == pytest.approx(oracle.distance_to_entity(origin, loc["entity_id"]), abs=2e-2)
            shorter |= loc["distance_m"] < loc["geodesic_distance_m"] - 1
        g = r.get("guidance", "")
        assert (note in g) if note else ("Left out" not in g), (reference, g)
        assert (CAP_NOTE in g) == capped, (reference, g)
    if not shorter:
        pytest.skip(f"no {profile} result here is shorter by route than by straight line")


def test_knn_search_network_candidate_cap_is_flagged(session, oracle, fx):
    """With limit 50 the cap note appears exactly when the 101st straight-line candidate, less the origin's snap distance, is not farther than the 50th route."""
    seen = set()
    for reference in [fx.anchor_ll] + far_references(fx):
        r = network_knn(session, reference, profile="car", limit=50)
        _, _, capped = expected_network_knn(oracle, reference, "car", 50)
        assert r["count"] == 50
        got = [(x["distance_m"], x["duration_min"]) for x in r["locations"]]
        assert got == sorted(got), "not sorted by distance_m, then duration_min"
        assert (CAP_NOTE in r.get("guidance", "")) == capped, reference
        seen.add(capped)
    assert seen == {True, False}, seen


def test_knn_search_network_cap_note_when_fewer_routes_than_limit(oracle, fx, monkeypatch):
    """Over 100 matches but fewer routable than limit: the cap note appears even though the 101st match lies beyond the last route."""
    near = oracle.knn(fx.anchor_ll, 101, None, {})
    reach = near[9][1] + 1  # only places about this close get a route

    def answer(path):
        a, b = [dict(zip(("lat", "lon"), map(float, p.split(",")))) for p in urllib.parse.parse_qs(urllib.parse.urlparse(path).query)["point"]]
        if geo.haversine(a, b) > reach:
            return 400, '{"message": "Connection between locations not found"}'
        with urllib.request.urlopen(GH_URL + path) as resp:
            return 200, resp.read().decode()

    with graphhopper_stub(monkeypatch, answer) as (s, _):
        r = network_knn(s, fx.anchor_ll, profile="car", limit=50)
    last = r["locations"][-1]
    snap = oracle.distance(fx.anchor_ll, dict(zip(("lon", "lat"), gh_path(fx.anchor_ll, last, "car")["snapped_waypoints"]["coordinates"][0])))
    assert 0 < r["count"] < 50 and near[100][1] - snap > last["distance_m"], (r["count"], near[100][1], last["distance_m"])
    assert CAP_NOTE in r["guidance"]


def test_knn_search_network_without_any_route_says_so(session, fx):
    """A reference outside the routing area routes to no match: count 0, guidance says why, and no cap note."""
    r = network_knn(session, {"lat": -60, "lon": 100}, profile="foot", limit=5)
    assert r["count"] == 0 and r["guidance"] == "No foot route to any match. Check the reference is inside the data area and on or near a road, or try another profile."


@pytest.mark.parametrize("unit", ["km", "mi"])
def test_knn_search_network_units_convert_both_distances(session, fx, unit):
    """units adds the converted route and straight-line distances after their metre values."""
    for loc in network_knn(session, fx.anchor_ll, limit=2, include=render(fx.cat_a), units=unit)["locations"]:
        assert_converted(loc, "distance", unit)
        assert_converted(loc, "geodesic_distance", unit)
        keys = list(loc)
        assert keys.index("distance_m") < keys.index(f"distance_{unit}") < keys.index("duration_min") < keys.index("geodesic_distance_m") < keys.index(f"geodesic_distance_{unit}")
