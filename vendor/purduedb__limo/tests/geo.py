"""Pure spherical geodesy, independent of server.ts: destination point and compass names (ellipsoidal distance and bearing: see oracle)."""
import math

EARTH_R = 6371000.0
COMPASS = {
    4: ["N", "E", "S", "W"],
    8: ["N", "NE", "E", "SE", "S", "SW", "W", "NW"],
    16: ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE", "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"],
}


def destination(lat, lon, azimuth_deg, dist_m):
    """(lat, lon) reached from a point along an initial azimuth for dist_m."""
    p1, l1, th, d = math.radians(lat), math.radians(lon), math.radians(azimuth_deg), dist_m / EARTH_R
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(th))
    l2 = l1 + math.atan2(math.sin(th) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    return math.degrees(p2), (math.degrees(l2) + 540) % 360 - 180


def haversine(p, q):
    """Great-circle metres between two {lat, lon} points."""
    a, b, dl = math.radians(p["lat"]), math.radians(q["lat"]), math.radians(q["lon"] - p["lon"])
    return 2 * EARTH_R * math.asin(math.sqrt(math.sin((b - a) / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(dl / 2) ** 2))


def compass_name(bearing, rose):
    names = COMPASS[rose]
    return names[round(bearing / (360 / len(names))) % len(names)]
