"""L0 basic blocks: `geocode`, `reverse_geocode`, `lookup`.

Expectations come from the SQL oracle over the loaded extract; every tool above this layer relies on these.
Run: python -m pytest tests/test_l0_places.py -v
"""
import os
import urllib.parse
import urllib.request

import pytest

from geo import compass_name
from harness import call, rejects, session  # noqa: F401
from oracle import TABLES

MAX_ID = "9223372036854775807"  # grammatical, never an OSM id
PLACE_KINDS = ["named_point", "named_building", "named_landmark", "named_park", "named_city"]


@pytest.fixture(scope="module", autouse=True)
def fresh_streets(oracle):
    oracle.clear_street_cache()


def geocode(session, query, **args):
    return call(session, "geocode", {"query": query, **args})


def reverse(session, point, limit=10):
    return call(session, "reverse_geocode", {"point": point, "limit": limit})


def lookup(session, id):
    return call(session, "lookup", {"id": id})


def assert_entity(oracle, loc):
    """A returned place matches its stored row: representative point and area or length."""
    eid = loc["entity_id"]
    (x, y), (area, perimeter) = oracle.reps([eid])[eid], oracle.entity_measures(eid)
    assert abs(loc["lat"] - y) < 2e-7 and abs(loc["lon"] - x) < 2e-7, eid
    assert loc.get("area_m2") == pytest.approx(area, abs=2e-3) and loc.get("perimeter_m") == pytest.approx(perimeter, abs=2e-3), eid


# ---- geocode: kinds of place ----

@pytest.mark.parametrize("kind", PLACE_KINDS)
def test_geocode_named_place_returns_that_entity_first(session, oracle, fx, kind):
    """The exact name of a unique POI, building, landmark, park or boundary comes back first, with the stored row's fields."""
    eid, name = getattr(fx, kind)
    top = geocode(session, name, limit=5)["locations"][0]
    assert top["name"] == name and top["entity_id"] == eid
    assert_entity(oracle, top)


def test_geocode_house_number_and_street_returns_that_house(session, oracle, fx):
    """A unique 'number street' address comes back first as the entity carrying it."""
    eid, address = fx.house
    top = geocode(session, address, limit=5)["locations"][0]
    assert top["entity_id"] == eid
    assert_entity(oracle, top)


def test_geocode_single_segment_street_is_returned_as_is(session, oracle, fx):
    """A street with one segment comes back as that line entity, not a merged one."""
    eid, name = fx.streets["single"]
    top = geocode(session, name, limit=5)["locations"][0]
    assert top["entity_id"] == eid and "merged_from" not in top
    assert_entity(oracle, top)


# ---- geocode: street merge ----

def merged_street(session, fx):
    """The merged hit of the fixture street and its 100 m component (the one holding its first segment)."""
    name, comps = fx.streets["merged"]
    hit = next(l for l in geocode(session, name, limit=5)["locations"] if l["entity_id"].startswith("tmp:street-") and l["name"] == name)
    first = int(hit["merged_from"][0].split(":")[1])
    return hit, next(c for c in comps if first in c[0])


def test_geocode_merges_a_street_chain_into_one_entity(session, fx):
    """A same-named chain within 100 m merges: one tmp entity with the chain's total length, count and parts."""
    hit, (ids, length, _) = merged_street(session, fx)
    assert hit["merged_count"] == len(ids) and hit["perimeter_m"] == pytest.approx(length, abs=1e-2)
    shown = hit["merged_from"]
    assert shown[:5] == [f"line:{i}" for i in sorted(ids)][:5] and (shown[5:] == ["..."]) == (len(ids) > 5)


def test_geocode_merged_street_geometry_and_point(session, oracle, fx):
    """The tmp geometry is the line-merged union of the chain; lat, lon is a point on that geometry."""
    hit, (ids, _, _) = merged_street(session, fx)
    merged = "(SELECT ST_LineMerge(ST_Collect(way)) FROM planet_osm_line WHERE osm_id = ANY(%s::bigint[]))"
    same, parts, x, y = oracle.q(
        f"SELECT ST_Equals(geom, {merged}), ST_NumGeometries(geom) - ST_NumGeometries({merged}), ST_X(ST_PointOnSurface(geom)), ST_Y(ST_PointOnSurface(geom)) "
        f"FROM tmp_geom WHERE id = %s", [ids, ids, hit["entity_id"]])[0]
    assert same and parts == 0
    assert abs(hit["lon"] - x) < 2e-7 and abs(hit["lat"] - y) < 2e-7


def test_geocode_merged_street_tags_combine_distinct_values(session, fx):
    """Per tag key: one value stays plain, several become a sorted list; keys come from any segment."""
    hit, (_, _, tags) = merged_street(session, fx)
    want = {k: v[0] if len(v) == 1 else v for k, v in tags.items()}
    assert hit["tags"] == want


def test_geocode_merged_street_is_reused_and_found_by_lookup(session, fx):
    """The same street query returns the same tmp id; lookup returns the same place, and guidance flags the merge."""
    name, _ = fx.streets["merged"]
    first, again = geocode(session, name, limit=5), geocode(session, name, limit=5)
    hit = next(l for l in first["locations"] if l["entity_id"].startswith("tmp:street-"))
    assert hit["entity_id"] in [l["entity_id"] for l in again["locations"]]
    assert lookup(session, hit["entity_id"])["location"] == hit
    assert "`tmp:*` result merges all segments" in first["guidance"]


# ---- geocode: results, guidance and input ----

def test_geocode_guidance_follows_the_result_size(session, fx):
    """No match, several candidates and a result filling limit carry their guidance; a lone unfilled result carries none."""
    none = geocode(session, "zzz_not_a_real_place_zzz")
    assert none["count"] == 0 and none["locations"] == [] and "OpenStreetMap" in none["guidance"]
    counts = set()
    for query, limit in ((fx.shared_name, 5), (fx.shared_name, 1), (fx.shared_name, 50), (fx.named_building[1], 50)):
        r = geocode(session, query, limit=limit)
        ranked = r["count"] > 1 or r["count"] == limit
        assert ("raise limit" in r.get("guidance", "")) == ranked, (query, limit, r["count"])
        counts.add("several below limit" if 1 < r["count"] < limit else "full" if r["count"] == limit else "lone" if r["count"] == 1 else "none")
    assert {"several below limit", "full", "lone"} <= counts, counts


def test_geocode_limit_is_respected(session, fx):
    """A query with more matches than limit returns exactly limit places."""
    r = geocode(session, fx.shared_name, limit=2)
    assert r["count"] == len(r["locations"]) == 2


def test_geocode_skips_hits_without_an_osm_object(session, oracle, fx):
    """Photon hits with no OSM object (a postcode) are skipped, not an error: the result is exactly Photon's hits that are stored entities."""
    q = urllib.parse.urlencode({"q": fx.postcode, "limit": 5})
    with urllib.request.urlopen(f"{os.environ.get('PHOTON_URL', 'http://127.0.0.1:2322')}/api?{q}") as resp:
        hits = [f["properties"] for f in json.load(resp)["features"]]
    assert any(not h.get("osm_id") for h in hits), "the postcode query has no hit without an OSM object"
    want = []
    for h in hits:
        if h.get("osm_id"):
            n = h["osm_id"]
            kinds = {"N": [("point", n)], "W": [("line", n), ("polygon", n)], "R": [("polygon", -n), ("line", -n)]}[h["osm_type"]]
            eid = next((f"{k}:{i}" for k, i in kinds if oracle.q(f"SELECT 1 FROM {TABLES[k]} WHERE osm_id = %s", [i])), None)
            want += [eid] if eid and eid not in want else []
    assert [loc["entity_id"] for loc in geocode(session, fx.postcode)["locations"]] == want


@pytest.mark.parametrize("args, message", [
    ({"query": "  "}, "query must be a non-empty"),
    ({"query": "x", "limit": 0}, "limit must be an integer from 1 to 50"),
    ({"query": "x", "limit": 51}, "limit must be an integer from 1 to 50"),
    ({"query": "x", "limit": 2.5}, "limit must be an integer from 1 to 50"),
    ({"query": "x", "location_bias": {"lat": 200, "lon": 0}}, r"location_bias\.lat must be a number in \[-90, 90\]"),
    ({"query": "x", "location_bias": {"lat": 116.4, "lon": 39.9}}, "look swapped"),
    ({"query": "x", "location_bias": {"lat": 40.4}}, "lon"),
])
def test_geocode_rejects_bad_input_with_a_fix(session, args, message):
    """Bad arguments fail before any search, with a message that says how to correct the call."""
    rejects(session, message, "geocode", **args)


def test_geocode_location_bias_keeps_the_candidate_set_and_reorders_it(session, oracle, fx):
    """A biased query returns the same candidates as the unbiased one; biasing towards different same-named places changes the order."""
    plain = {l["entity_id"] for l in geocode(session, fx.shared_name, limit=20)["locations"]}
    orders = set()
    for x, y in oracle.q("SELECT ST_X(way), ST_Y(way) FROM planet_osm_point WHERE name = %s ORDER BY osm_id", [fx.shared_name]):
        biased = [l["entity_id"] for l in geocode(session, fx.shared_name, limit=20, location_bias={"lat": y, "lon": x})["locations"]]
        assert set(biased) == plain
        orders.add(tuple(biased))
    assert len(orders) > 1


# ---- reverse_geocode ----

@pytest.mark.parametrize("kind", ["named_point", "named_building"])
def test_reverse_geocode_finds_entity_at_its_own_location(session, oracle, fx, kind):
    """A named place is found from its own representative point."""
    eid = getattr(fx, kind)[0]
    x, y = oracle.reps([eid])[eid]
    assert eid in [loc["entity_id"] for loc in reverse(session, {"lat": y, "lon": x})["locations"]]


def test_reverse_geocode_distance_is_to_the_geometry_and_sorted(session, oracle, fx):
    """Each hit carries its geodesic distance and shortest-line bearing (absent at distance 0) to the stored geometry, nearest first."""
    locs = reverse(session, fx.anchor_ll, limit=20)["locations"]
    assert len(locs) >= 5 and any(loc["geodesic_distance_m"] > 0 for loc in locs)
    for loc in locs:
        assert loc["geodesic_distance_m"] == pytest.approx(oracle.distance_to_entity(fx.anchor_ll, loc["entity_id"]), abs=2e-3), loc["entity_id"]
        if loc["geodesic_distance_m"] == 0:
            assert "bearing_deg" not in loc and "direction" not in loc
        else:
            assert loc["bearing_deg"] == pytest.approx(oracle.bearing_to_entity(fx.anchor_ll, loc["entity_id"]), abs=2e-3), loc["entity_id"]
            assert loc["direction"] == compass_name(loc["bearing_deg"], 8) and list(loc)[-2:] == ["bearing_deg", "direction"]
    d = [loc["geodesic_distance_m"] for loc in locs]
    assert d == sorted(d)


def test_reverse_geocode_limit_is_respected(session, fx):
    """limit caps the number of places returned at a point with more places around."""
    assert reverse(session, fx.anchor_ll, limit=2)["count"] == 2


def test_reverse_geocode_guidance_follows_the_result(session, fx):
    """Over points around the anchor: nothing found, a nearest hit over 50 m away (also one under 250 m), or a result filling limit
    each carry their guidance."""
    seen = set()
    offsets = [(0, 0), (0.003, 0), (0.03, 0.03)] + [(dy / 1000, dx / 1000) for dy in range(10, 31, 5) for dx in (-3, 0, 3)]
    for dy, dx in offsets:
        r = reverse(session, {"lat": fx.anchor["lat"] + dy, "lon": fx.anchor["lon"] + dx}, limit=5)
        guidance = r.get("guidance", "")
        if r["count"] == 0:
            assert "No OSM place" in guidance
            seen.add("none")
            continue
        nearest = min(l["geodesic_distance_m"] for l in r["locations"])
        if nearest > 50:
            assert f"{round(nearest)} m away" in guidance
            seen.add("far" if nearest > 250 else "near_far")
        elif r["count"] == 5:
            assert "Raise limit" in guidance
            seen.add("full")
        else:
            assert guidance == ""
            seen.add("quiet")
    assert {"far", "near_far", "full"} <= seen, seen


@pytest.mark.parametrize("args, message", [
    ({"point": {"lat": 91, "lon": 0}}, r"point\.lat must be a number in \[-90, 90\]"),
    ({"point": {"lat": 0, "lon": 181}}, r"point\.lon must be a number in \[-180, 180\]"),
    ({"point": {"lat": 116.4, "lon": 39.9}}, "look swapped"),
    ({"point": {"lat": 40.4, "lon": -86.9}, "limit": 0}, "limit must be an integer from 1 to 50"),
    ({"lat": 40.4, "lon": -86.9}, "point"),
])
def test_reverse_geocode_rejects_bad_input_with_a_fix(session, args, message):
    """Bad arguments fail with a message that says how to correct the call."""
    rejects(session, message, "reverse_geocode", **args)


def test_explicit_nulls_equal_omitted_arguments(session, fx):
    """null for an optional geocode or reverse_geocode argument behaves like omitting it."""
    name = fx.shared_name
    assert geocode(session, name, location_bias=None, limit=None) == geocode(session, name)
    assert call(session, "reverse_geocode", {"point": fx.anchor_ll, "limit": None}) == call(session, "reverse_geocode", {"point": fx.anchor_ll})


# ---- address ----

def photon_record(eid, lat, lon):
    """Photon's properties for this exact OSM object (by type and id) from a reverse query at its point, {} if absent."""
    kind, n = eid.split(":")
    typ = "N" if kind == "point" else "R" if n.startswith("-") else "W"
    q = urllib.parse.urlencode({"lat": lat, "lon": lon, "limit": 5})
    with urllib.request.urlopen(f"{os.environ.get('PHOTON_URL', 'http://127.0.0.1:2322')}/reverse?{q}") as resp:
        features = json.load(resp)["features"]
    return next((f["properties"] for f in features if f["properties"].get("osm_type") == typ and str(f["properties"].get("osm_id")) == n.lstrip("-")), {})


def test_address_is_built_from_tags_then_photon(session, fx):
    """address joins number street, city, state postcode from the addr:* tags, each missing part from Photon's record of the
    same object; addr:* keys leave tags."""
    filled = 0
    for eid, _, tags in fx.names[:60]:
        loc = lookup(session, eid)["location"]
        photon = photon_record(eid, loc["lat"], loc["lon"])
        parts = {}
        for tag, prop in (("housenumber", "housenumber"), ("street", "street"), ("city", "city"), ("state", "state"), ("postcode", "postcode")):
            parts[tag] = tags.get(f"addr:{tag}") or (str(photon[prop]) if photon.get(prop) else None)
            filled += not tags.get(f"addr:{tag}") and bool(photon.get(prop))
        if parts["state"] and len(parts["state"]) > 2:
            continue  # full state names: see test_full_state_name_becomes_its_code
        want = ", ".join(filter(None, [" ".join(filter(None, [parts["housenumber"], parts["street"]])), parts["city"],
                                       " ".join(filter(None, [parts["state"], parts["postcode"]]))])) or None
        assert loc["address"] == want, (eid, tags, photon)
        assert not any(k.startswith("addr:") and k[5:] in parts for k in loc["tags"]), eid
    assert filled, "no part came from Photon"


def test_full_state_name_becomes_its_code(session, oracle):
    """An addr:state spelled out as a US state name is reported as its two-letter code."""
    rows = oracle.q("SELECT 'point:' || osm_id, tags -> 'addr:state' FROM planet_osm_point WHERE tags -> 'addr:state' ~ '^[A-Z][a-z]+( [A-Z][a-z]+)*$' "
                    "UNION ALL SELECT 'polygon:' || osm_id, tags -> 'addr:state' FROM planet_osm_polygon WHERE tags -> 'addr:state' ~ '^[A-Z][a-z]+( [A-Z][a-z]+)*$' LIMIT 1")
    if not rows:
        pytest.skip("no addr:state spelled out as a name")
    eid, name = rows[0]
    address = lookup(session, eid)["location"]["address"]
    assert name not in address and re.search(r", [A-Z]{2}( \d+)?$", address), address


# ---- lookup ----

def test_lookup_resolves_osm_ids_to_entities(session, fx):
    """n is a point, w a line or a polygon by lookup, r a polygon or else a line with the id negated; the result carries the entity_id."""
    ids = fx.osm_ids
    way_polygon = "w" + fx.way_polygon.split(":")[1]
    want = {ids["n"]: fx.anchor["id"], ids["w"]: fx.line, way_polygon: fx.way_polygon, ids["r"]: f"polygon:-{ids['r'][1:]}"}
    if fx.line_relation:
        want["r" + fx.line_relation.split(":-")[1]] = fx.line_relation
    for qualified, entity_id in want.items():
        r = lookup(session, qualified)
        assert r["location"]["entity_id"] == entity_id and entity_id in r["guidance"]
    rejects(session, "Unrecognized id", "lookup", id=f"w-{ids['r'][1:]}")  # a negative way id is not a relation


def test_lookup_accepts_upper_case_and_padding(session, fx):
    """An OSM id is case-insensitive and surrounding spaces are ignored."""
    assert lookup(session, f" {fx.osm_ids['n'].upper()} ")["location"]["entity_id"] == fx.anchor["id"]


def test_lookup_returns_the_stored_row(session, oracle, fx):
    """Point, line, way polygon, relation polygon and multi-row entities resolve to their own id, and the point and measures of all their rows."""
    for eid in filter(None, (fx.anchor["id"], fx.line, fx.way_polygon, fx.overlap_polygons[0], fx.multirow_line)):
        r = lookup(session, eid)
        assert r["location"]["entity_id"] == eid and "guidance" not in r
        assert_entity(oracle, r["location"])


@pytest.mark.parametrize("id, message", [
    (f"point:{MAX_ID}", "Unrecognized id"),
    (f"w{MAX_ID}", "Unrecognized id"),
    (f"point:{MAX_ID}0", "Unrecognized id"),
    (f"w{MAX_ID}0", "Unrecognized id"),
    ("not-an-id", "Unrecognized id"),
    ("123", "Unrecognized id"),
])
def test_lookup_errors_say_how_to_get_a_valid_id(session, id, message):
    """An unknown or malformed id fails with the accepted formats."""
    rejects(session, message, "lookup", id=id)
