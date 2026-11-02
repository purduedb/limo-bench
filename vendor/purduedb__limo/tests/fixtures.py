"""Fixture entities chosen by property from whatever extract is loaded; every query is bounded around one dense anchor."""
import math
import re
from collections import Counter
from functools import cached_property

from oracle import CAP, CATEGORY_KEYS, RADIUS_M, TABLES, tag_sql

SIMPLE_NAME = re.compile(r"^[A-Za-z][A-Za-z' ]{5,}$")  # names a text geocoder matches unambiguously
WIDE_DEG = 0.1  # box around the anchor in which names must be unique


class Fixtures:
    """Fixture entities chosen by property around one dense anchor; computed lazily, once per session."""

    def __init__(self, oracle):
        self.o = oracle

    @cached_property
    def anchor(self):
        """The point entity with the most other points within 500 m, among a deterministic sample."""
        row = self.o.q(
            "SELECT 'point:' || s.osm_id, ST_Y(s.way), ST_X(s.way) FROM (SELECT osm_id, way FROM planet_osm_point ORDER BY osm_id LIMIT 200) s "
            "ORDER BY (SELECT count(*) FROM planet_osm_point p WHERE p.way && ST_Expand(s.way, 0.01) AND ST_DWithin(p.way::geography, s.way::geography, 500)) DESC, s.osm_id LIMIT 1"
        )
        assert row, "planet_osm_point is empty"
        eid, lat, lon = row[0]
        return {"id": eid, "lat": lat, "lon": lon}

    @property
    def anchor_ll(self):
        return {"lat": self.anchor["lat"], "lon": self.anchor["lon"]}

    @cached_property
    def categories(self):
        """[(key, value, count)] of OSM categories within RADIUS_M of the anchor, 8 to 150 members, largest first."""
        out = []
        for key in CATEGORY_KEYS:
            parts = " UNION ALL ".join(
                f"SELECT {tag_sql('%(k)s', key, self.o.columns[kind])} AS v FROM {tbl} t, an "
                f"WHERE t.way && ST_Expand(an.a, 0.05) AND ST_DWithin(an.a::geography, t.way::geography, %(r)s)" for kind, tbl in TABLES.items()
            )
            out += [(key, v, c) for v, c in self.o.q(
                f"WITH an AS (SELECT ST_SetSRID(ST_MakePoint(%(x)s, %(y)s), 4326) AS a) SELECT v, count(*) FROM ({parts}) u "
                "WHERE v IS NOT NULL GROUP BY v HAVING count(*) BETWEEN 8 AND 150",
                {"k": key, "x": self.anchor["lon"], "y": self.anchor["lat"], "r": RADIUS_M},
            )]
        return sorted(out, key=lambda r: (-r[2], r[0], r[1]))

    def _cat(self, lo, hi, other=None):
        for k, v, c in self.categories:
            if lo <= c <= hi and (k, v) != other:
                return {k: v}
        raise AssertionError(f"no category with {lo}..{hi} members near the anchor")

    @cached_property
    def cat_a(self):
        return self._cat(20, 80)

    @cached_property
    def cat_b(self):
        return self._cat(8, 60, other=next(iter(self.cat_a.items())))

    def ids(self, include, exclude=None):
        return [i for i, _ in self.o.near(self.anchor_ll, RADIUS_M, include, exclude)]

    @cached_property
    def cat_c(self):
        """{key: value} of a category under another key than cat_a, 8..60 members near the anchor, or None."""
        (ka, _), = self.cat_a.items()
        return next(({k: v} for k, v, c in self.categories if k != ka and c <= 60), None)

    @cached_property
    def a_ids(self):
        return self.ids(self.cat_a)

    @cached_property
    def b_ids(self):
        return self.ids(self.cat_b)

    @cached_property
    def dense(self):
        """[(entity_id, d)] of all entities near the anchor, nearest first, at least 260 of them."""
        r = 200.0
        while len(rows := self.o.near(self.anchor_ll, r)) < 260:
            assert r < 2e4, "fewer than 260 entities within 20 km of the anchor"
            r *= 2
        return rows

    @cached_property
    def tie(self):
        """Two distinct point entities at identical coordinates near the anchor, or None."""
        row = self.o.q(
            "SELECT 'point:' || a.osm_id, 'point:' || b.osm_id FROM (SELECT osm_id, way FROM planet_osm_point "
            "WHERE way && ST_Expand(ST_SetSRID(ST_MakePoint(%s, %s), 4326), 0.05) ORDER BY osm_id LIMIT 5000) a "
            "JOIN LATERAL (SELECT osm_id FROM planet_osm_point b WHERE b.way && a.way AND ST_Equals(b.way, a.way) AND b.osm_id > a.osm_id LIMIT 1) b ON true LIMIT 1",
            [self.anchor["lon"], self.anchor["lat"]],
        )
        return list(row[0]) if row else None

    def _near_anchor(self, sql, wide=False):
        """Rows of sql, where {box} is a 0.02 degree box around the anchor and %(x)s, %(y)s are its coordinates."""
        return self.o.q(sql if wide else sql.format(box="ST_Expand(ST_SetSRID(ST_MakePoint(%(x)s, %(y)s), 4326), 0.02)"), {"x": self.anchor["lon"], "y": self.anchor["lat"]})

    @cached_property
    def line(self):
        """The ordinary (positive id) line entity nearest the anchor."""
        return self._near_anchor("SELECT 'line:' || osm_id FROM planet_osm_line WHERE osm_id > 0 AND way && {box} "
                                 "ORDER BY way <-> ST_Centroid({box}), osm_id LIMIT 1")[0][0]

    @cached_property
    def relation_polygon(self):
        """A relation (negative id) polygon near the anchor intersecting 2..199 entities, or None."""
        for (eid,) in self._near_anchor("SELECT 'polygon:' || osm_id FROM planet_osm_polygon WHERE osm_id < 0 AND way && {box} ORDER BY way_area, osm_id LIMIT 20"):
            if 2 <= len(self.o.near(eid, 0)) < CAP:
                return eid
        return None

    @cached_property
    def polygon_with_inner_point(self):
        """(polygon id, point id): the point lies inside the polygon but over 50 m from its representative point, or None."""
        rows = self._near_anchor(
            "SELECT 'polygon:' || g.osm_id, 'point:' || p.osm_id FROM (SELECT osm_id, way FROM planet_osm_polygon WHERE osm_id > 0 AND way && {box} "
            "ORDER BY way_area DESC, osm_id LIMIT 50) g JOIN LATERAL (SELECT osm_id FROM planet_osm_point p WHERE p.way && g.way AND ST_Intersects(g.way, p.way) "
            "AND ST_Distance(p.way::geography, ST_PointOnSurface(g.way)::geography) > 50 ORDER BY osm_id LIMIT 1) p ON true LIMIT 1")
        return tuple(rows[0]) if rows else None

    @cached_property
    def value_list(self):
        """{key: [cat_a value, another value]} whose members together stay under the cap."""
        (k, v), = self.cat_a.items()
        other = next((v2 for k2, v2, c in self.categories if k2 == k and v2 != v and len(self.a_ids) + c < CAP), None)
        assert other, f"no second {k} value near the anchor"
        return {k: [v, other]}

    @cached_property
    def partial_tag(self):
        """{key: value} of a tag carried by some but not all cat_a members, or None."""
        counts = {}
        for kind, tbl in TABLES.items():
            ns = [i.split(":")[1] for i in self.a_ids if i.startswith(kind + ":")]
            for kv in self.o.q(f"SELECT (each(tags)).* FROM {tbl} WHERE osm_id = ANY(%s::bigint[])", [ns]) if ns else []:
                counts[kv] = counts.get(kv, 0) + 1
        best = sorted((-c, kv) for kv, c in counts.items() if c < len(self.a_ids) and not kv[0].startswith("addr:"))
        return {best[0][1][0]: best[0][1][1]} if best else None

    @cached_property
    def hstore_category(self):
        """{key: value} of a tag stored only in the hstore (no column on any table) with 5..150 entities nearby, or None."""
        cols = set().union(*self.o.columns.values())
        parts = " UNION ALL ".join(
            f"SELECT (each(t.tags)).* FROM {tbl} t, an WHERE t.way && ST_Expand(an.a, 0.05) AND ST_DWithin(an.a::geography, t.way::geography, %(r)s)"
            for tbl in TABLES.values()
        )
        for k, v, _ in self.o.q(
            f"WITH an AS (SELECT ST_SetSRID(ST_MakePoint(%(x)s, %(y)s), 4326) AS a) SELECT key, value, count(*) FROM ({parts}) u "
            "WHERE key NOT LIKE 'addr:%%' GROUP BY 1, 2 HAVING count(*) BETWEEN 5 AND 150 ORDER BY 3 DESC, 1, 2 LIMIT 50",
            {"x": self.anchor["lon"], "y": self.anchor["lat"], "r": RADIUS_M},
        ):
            if k not in cols:
                return {k: v}
        return None

    @cached_property
    def names(self):
        """[(entity_id, name, tags)] of entities with a unique, plainly spelt name within WIDE_DEG of the anchor, by osm_id."""
        out = []
        for kind, tbl in TABLES.items():
            out += [(f"{kind}:{i}", n, t or {}) for i, n, t in self._near_anchor(
                f"SELECT osm_id, name, jsonb_strip_nulls(to_jsonb(t) - 'tags' - 'way') || hstore_to_jsonb(COALESCE(tags, ''::hstore)) FROM {tbl} t "
                f"WHERE name IS NOT NULL AND way && ST_Expand(ST_SetSRID(ST_MakePoint(%(x)s, %(y)s), 4326), {WIDE_DEG}) ORDER BY osm_id", wide=True)]
        count = Counter(n for _, n, _ in out)
        return [r for r in out if count[r[1]] == 1 and SIMPLE_NAME.match(r[1])]

    def _named(self, kind, test):
        for eid, name, tags in self.names:
            if eid.startswith(kind + ":") and test(eid, tags):
                return eid, name
        raise AssertionError(f"no uniquely named {kind} near the anchor")

    @cached_property
    def named_point(self):
        """(entity_id, name) of a uniquely named point POI (amenity or shop)."""
        return self._named("point", lambda e, t: "amenity" in t or "shop" in t)

    @cached_property
    def spaced_name(self):
        """(entity_id, name) of a uniquely named entity whose name holds a space, or None."""
        return next(((e, n) for e, n, _ in self.names if " " in n), None)

    @cached_property
    def named_building(self):
        """(entity_id, name) of a uniquely named way polygon."""
        return self._named("polygon", lambda e, t: not e.startswith("polygon:-") and "building" in t)

    @cached_property
    def named_landmark(self):
        """(entity_id, name) of a uniquely named tourism or historic feature."""
        for eid, name, tags in self.names:
            if "tourism" in tags or "historic" in tags:
                return eid, name
        raise AssertionError("no uniquely named tourism or historic feature near the anchor")

    @cached_property
    def named_park(self):
        """(entity_id, name) of a uniquely named leisure polygon."""
        return self._named("polygon", lambda e, t: "leisure" in t)

    @cached_property
    def named_city(self):
        """(entity_id, name) of a uniquely named administrative boundary."""
        return self._named("polygon", lambda e, t: e.startswith("polygon:-") and t.get("boundary") == "administrative")

    @cached_property
    def house(self):
        """(entity_id, 'number street') of a polygon or point whose house number and street are unique in the extract."""
        rows = self._near_anchor(
            "SELECT kind || ':' || osm_id, num || ' ' || street FROM ("
            "SELECT 'polygon' AS kind, osm_id, \"addr:housenumber\" AS num, tags -> 'addr:street' AS street FROM planet_osm_polygon "
            "WHERE osm_id > 0 AND \"addr:housenumber\" IS NOT NULL AND tags ? 'addr:street' AND way && {box} "
            "UNION ALL SELECT 'point', osm_id, \"addr:housenumber\", tags -> 'addr:street' FROM planet_osm_point "
            "WHERE \"addr:housenumber\" IS NOT NULL AND tags ? 'addr:street' AND way && {box}) h "
            "WHERE (SELECT count(*) FROM (SELECT 1 FROM planet_osm_polygon WHERE \"addr:housenumber\" = h.num AND tags -> 'addr:street' = h.street "
            "UNION ALL SELECT 1 FROM planet_osm_point WHERE \"addr:housenumber\" = h.num AND tags -> 'addr:street' = h.street) c) = 1 "
            "ORDER BY osm_id LIMIT 1")
        assert rows, "no uniquely addressed house near the anchor"
        return tuple(rows[0])

    @cached_property
    def postcode(self):
        """The most common addr:postcode among polygons near the anchor."""
        rows = self._near_anchor(
            "SELECT tags -> 'addr:postcode' FROM planet_osm_polygon WHERE tags ? 'addr:postcode' AND way && {box} GROUP BY 1 ORDER BY count(*) DESC, 1 LIMIT 1")
        assert rows, "no addr:postcode near the anchor"
        return rows[0][0]

    @cached_property
    def streets(self):
        """{single: (entity_id, name), merged: (name, components)}: a street with one segment, and one with a chain of 3+ same-named segments."""
        names = self._near_anchor(
            "SELECT name, count(*) FROM planet_osm_line WHERE name IS NOT NULL AND highway IS NOT NULL AND way && ST_Expand(ST_SetSRID(ST_MakePoint(%(x)s, %(y)s), 4326), 0.05) "
            "GROUP BY name ORDER BY name", wide=True)
        single = next(((f"line:{ids[0]}", n) for n, c in names if c == 1 and SIMPLE_NAME.match(n) and len(ids := [i for i, in self.o.q("SELECT osm_id FROM planet_osm_line WHERE name = %s AND highway IS NOT NULL", [n])]) == 1), None)
        merged, best = None, (False, -1)
        for n, c in names:
            if 3 <= c <= 80 and SIMPLE_NAME.match(n):
                comps = self.o.street_components(n)
                big = max(comps, key=lambda comp: len(comp[0]))
                if len(big[0]) < 3:
                    continue
                parts = self.o.q("SELECT ST_NumGeometries(ST_LineMerge(ST_Collect(way))) FROM planet_osm_line WHERE osm_id = ANY(%s::bigint[])", [big[0]])[0][0]
                # prefer a chain linked by a 50..100 m gap, then truncation (6+), conflicting tag values, line joins, a second same-named group
                score = (len(self.o.street_components(n, 50.0)) > len(comps), (len(big[0]) >= 6) + any(len(v) > 1 for v in big[2].values()) + (parts < len(big[0])) + (len(comps) > 1))
                if score > best:
                    merged, best = (n, comps), score
                if best == (True, 4):
                    break
        assert single and merged, "no single-segment street and no street chained from 3+ same-named segments near the anchor"
        return {"single": single, "merged": merged}

    @cached_property
    def shared_name(self):
        """A plainly spelt name carried by 3 to 8 point entities near the anchor."""
        rows = self._near_anchor(
            "SELECT name FROM planet_osm_point WHERE name IS NOT NULL AND way && ST_Expand(ST_SetSRID(ST_MakePoint(%(x)s, %(y)s), 4326), 0.1) "
            "GROUP BY name HAVING count(*) BETWEEN 3 AND 8 ORDER BY count(*) DESC, name", wide=True)
        return next(n for (n,) in rows if SIMPLE_NAME.match(n))

    @cached_property
    def route_trio(self):
        """[(entity_id, {lat, lon})] of three named point entities: B is 1-3 km from A and C at least twice as far as B."""
        cols = self.o.columns["point"]
        poi = " OR ".join(f"{tag_sql(repr(k), k, cols)} IS NOT NULL" for k in ("amenity", "shop"))
        pts = self._near_anchor(
            f"SELECT 'point:' || osm_id, ST_Y(way), ST_X(way), ST_Distance(way::geography, ST_SetSRID(ST_MakePoint(%(x)s, %(y)s), 4326)::geography) FROM planet_osm_point t WHERE name IS NOT NULL AND ({poi}) "
            "AND way && ST_Expand(ST_SetSRID(ST_MakePoint(%(x)s, %(y)s), 4326), 0.08) ORDER BY osm_id LIMIT 400", wide=True)
        a = (self.anchor["id"], self.anchor["lat"], self.anchor["lon"])
        d = lambda p: p[3]
        b = next((p for p in pts if 1000 <= d(p) <= 3000), None)
        c = next((p for p in pts if b and d(p) >= 2 * d(b)), None)
        assert b and c, "no named points 1-3 km and 2x farther from the anchor"
        return [(p[0], {"lat": p[1], "lon": p[2]}) for p in (a, b, c)]

    @cached_property
    def overlap_polygons(self):
        """(relation polygon id, way polygon id inside it): two polygons with a non-empty intersection."""
        rows = self._near_anchor(
            "SELECT 'polygon:' || r.osm_id, 'polygon:' || p.osm_id FROM (SELECT osm_id, way FROM planet_osm_polygon WHERE osm_id > 0 AND way && {box} "
            "ORDER BY way_area DESC, osm_id LIMIT 30) p JOIN LATERAL (SELECT osm_id FROM planet_osm_polygon r WHERE r.osm_id < 0 "
            "AND r.way && p.way AND ST_Contains(r.way, p.way) ORDER BY r.way_area, r.osm_id LIMIT 1) r ON true LIMIT 1")
        assert rows, "no way polygon inside a relation polygon near the anchor"
        return tuple(rows[0])

    @cached_property
    def area(self):
        """{id, name, a, b}: a named relation polygon around the anchor and two categories with 5..CAP members inside it."""
        for eid, name in self._near_anchor(
                "SELECT 'polygon:' || osm_id, name FROM planet_osm_polygon WHERE osm_id < 0 AND name IS NOT NULL AND ST_Contains(way, ST_SetSRID(ST_MakePoint(%(x)s, %(y)s), 4326)) "
                "ORDER BY way_area, osm_id LIMIT 6", wide=True):
            sets = []
            for key in CATEGORY_KEYS:
                parts = " UNION ALL ".join(
                    f"SELECT {tag_sql('%(k)s', key, self.o.columns[kind])} AS v FROM {tbl} t, an WHERE t.way && an.g AND ST_Intersects(an.g, t.way)" for kind, tbl in TABLES.items())
                sets += [{key: v} for v, c in self.o.q(
                    f"WITH an AS (SELECT (SELECT way FROM planet_osm_polygon WHERE osm_id = %(i)s::bigint) AS g) SELECT v, count(*) FROM ({parts}) u "
                    "WHERE v IS NOT NULL GROUP BY v HAVING count(*) BETWEEN 5 AND " + str(CAP) + " ORDER BY 2 DESC, 1", {"k": key, "i": eid.split(":")[1]})]
            if len(sets) >= 2 and SIMPLE_NAME.match(name):
                return {"id": eid, "name": name, "a": sets[0], "b": sets[1]}
        raise AssertionError("no named relation polygon around the anchor with two categories inside")

    @cached_property
    def osm_ids(self):
        """{n, w, r}: qualified-id strings of a point, a way (line) and a relation (polygon) present in the extract."""
        way = self.line.split(":")[1]
        rel = self._near_anchor("SELECT -osm_id FROM planet_osm_polygon WHERE osm_id < 0 AND way && {box} ORDER BY osm_id LIMIT 1")[0][0]
        return {"n": "n" + self.anchor["id"].split(":")[1], "w": "w" + way, "r": f"r{rel}"}

    @cached_property
    def multirow_line(self):
        """The line entity stored as several rows (a split relation or long way) nearest the anchor, or None."""
        rows = self._near_anchor("SELECT 'line:' || osm_id FROM planet_osm_line WHERE way && ST_Expand({box}, 0.1) GROUP BY osm_id HAVING count(*) > 1 "
                                 "ORDER BY min(ST_Distance(way, ST_Centroid({box}))), osm_id LIMIT 1")
        return rows[0][0] if rows else None

    @cached_property
    def line_relation(self):
        """A relation stored only in the line table (e.g. a bus route), as its line entity_id, or None."""
        rows = self.o.q("SELECT 'line:' || osm_id FROM planet_osm_line l WHERE osm_id < 0 AND NOT EXISTS (SELECT 1 FROM planet_osm_polygon p WHERE p.osm_id = l.osm_id) "
                        "ORDER BY osm_id LIMIT 1")
        return rows[0][0] if rows else None

    @cached_property
    def way_polygon(self):
        """A way (positive id) polygon entity near the anchor."""
        return self._near_anchor("SELECT 'polygon:' || osm_id FROM planet_osm_polygon WHERE osm_id > 0 AND way && {box} ORDER BY osm_id LIMIT 1")[0][0]


def deg_box(center, radius_m):
    """("envelope", ...) of half-size radius_m around a {lat, lon}."""
    dlat = radius_m / 111320
    dlon = radius_m / (111320 * math.cos(math.radians(center["lat"])))
    return ("envelope", center["lon"] - dlon, center["lat"] - dlat, center["lon"] + dlon, center["lat"] + dlat)


def far_references(fx):
    """Four coordinates 5 km off the anchor, where planar index order and geodesic distance disagree most."""
    lat, lon = fx.anchor["lat"], fx.anchor["lon"]
    dlat, dlon = 5000 / 111320, 5000 / (111320 * math.cos(math.radians(lat)))
    return [{"lat": lat + sy * dlat, "lon": lon + sx * dlon} for sy, sx in ((1, 0), (-1, 0), (0, 1), (0, -1))]
