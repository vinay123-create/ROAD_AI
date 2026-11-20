
import re
import json
from pathlib import Path
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
from math import radians, sin, cos, sqrt, atan2

import httpx


# ============================================================
# CONFIGURATION
# ============================================================

OVERPASS_URL = "https://overpass.atownsend.org.uk/api/interpreter"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

EARTH_RADIUS = 6371000
UK_TIMEZONE = ZoneInfo("Europe/London")
GALE_THRESHOLD_KMH = 56.3

VEHICLE_ROAD_TYPES = {
    "motorway", "trunk", "primary", "secondary", "tertiary",
    "unclassified", "residential", "living_street",
    "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link"
}

PEDESTRIAN_WAY_TYPES = {"footway", "path", "steps"}

MODEL_FEATURES = [
    "longitude", "latitude", "day_of_week",
    "first_road_class", "road_type", "speed_limit",
    "junction_detail", "junction_control", "second_road_class",
    "pedestrian_crossing", "light_conditions",
    "weather_conditions", "road_surface_conditions",
    "urban_or_rural_area", "trunk_road_flag",
    "year", "month", "hour"
]

# Load the exact category mappings used by the trained model.
ML_DIR = Path(__file__).resolve().parent.parent / "ml"
MAPPING_FILE = ML_DIR / "roadguard_category_mappings.json"

try:
    with open(MAPPING_FILE, "r", encoding="utf-8") as file:
        CATEGORY_MAPPINGS = json.load(file)
except (FileNotFoundError, json.JSONDecodeError):
    CATEGORY_MAPPINGS = {}


def valid_model_category(feature, value):
    """Check a value against the trained model's category mapping."""
    allowed = CATEGORY_MAPPINGS.get(feature, [])
    return value in allowed


def apply_unknown_fallback(features):
    """
    Apply only unknown/unallocated values that already exist
    in the trained model's category mappings.

    These are explicit unknown values, not inferred observations.
    """
    fallbacks = {
        "first_road_class": -1,
        "road_type": 9,
        "junction_detail": 99,
        "junction_control": -1,
        "second_road_class": -1,
        "pedestrian_crossing": -1,
        "weather_conditions": 9,
        "road_surface_conditions": 9,
        "urban_or_rural_area": 3,
        "trunk_road_flag": -1
    }

    applied = {}

    for feature, fallback in fallbacks.items():
        if features.get(feature) is None and valid_model_category(
            feature, fallback
        ):
            features[feature] = fallback
            applied[feature] = {
                "value": fallback,
                "reason": "Unknown/unallocated category from model mappings"
            }

    return applied


# ============================================================
# 1. CURRENT UK TIME FEATURES
# ============================================================

def get_current_time_features():
    current_time = datetime.now(UK_TIMEZONE)

    return {
        "day_of_week": current_time.weekday(),
        "year": current_time.year,
        "month": current_time.month,
        "hour": current_time.hour
    }


# ============================================================
# 2. CURRENT WEATHER
# ============================================================

async def get_current_weather(latitude, longitude):
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "current": (
            "temperature_2m,relative_humidity_2m,precipitation,rain,"
            "showers,snowfall,weather_code,wind_speed_10m,wind_gusts_10m"
        ),
        "wind_speed_unit": "kmh",
        "timezone": "Europe/London"
    }

    async with httpx.AsyncClient(timeout=20) as client:
        response = await client.get(OPEN_METEO_URL, params=params)
        response.raise_for_status()
        payload = response.json()

    current = payload.get("current")

    if not isinstance(current, dict) or current.get("weather_code") is None:
        raise ValueError("Open-Meteo response did not contain current weather data")

    return current


def map_weather_to_stats20(weather):
    if not isinstance(weather, dict):
        return None, {"reason": "No current weather observation supplied"}

    try:
        wmo_code = int(weather["weather_code"])
    except (KeyError, TypeError, ValueError):
        return None, {"reason": "Weather code missing or invalid"}

    wind_values = [
        value for value in (
            weather.get("wind_speed_10m"),
            weather.get("wind_gusts_10m")
        )
        if isinstance(value, (int, float))
    ]

    max_wind = max(wind_values) if wind_values else None

    evidence = {
        "provider": "Open-Meteo",
        "wmo_weather_code": wmo_code,
        "wind_speed_kmh": weather.get("wind_speed_10m"),
        "wind_gusts_kmh": weather.get("wind_gusts_10m")
    }

    if max_wind is not None and max_wind >= GALE_THRESHOLD_KMH:
        return None, {
            **evidence,
            "reason": "High wind detected; driving impact cannot be established"
        }

    if wmo_code in {0, 1, 2, 3}:
        return 1, {**evidence, "reason": "Clear or cloudy conditions"}

    if wmo_code in {
        51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82
    }:
        return 2, {**evidence, "reason": "Rain or drizzle reported"}

    if wmo_code in {71, 73, 75, 77, 85, 86}:
        return 3, {**evidence, "reason": "Snow reported"}

    if wmo_code in {45, 48}:
        return None, {
            **evidence,
            "reason": "Fog reported; visibility impact cannot be established"
        }

    if wmo_code in {95, 96, 99}:
        return 8, {**evidence, "reason": "Thunderstorm reported"}

    return None, {
        **evidence,
        "reason": "WMO code cannot be confidently mapped"
    }


# ============================================================
# 3. SUNRISE AND SUNSET
# ============================================================

def calculate_sunrise_sunset(latitude, longitude, date):
    day_number = date.timetuple().tm_yday

    def calculate_event(is_sunrise):
        longitude_hour = longitude / 15

        approximate_time = day_number + (
            (6 - longitude_hour) / 24
            if is_sunrise
            else (18 - longitude_hour) / 24
        )

        mean_anomaly = 0.9856 * approximate_time - 3.289

        true_longitude = (
            mean_anomaly
            + 1.916 * sin(radians(mean_anomaly))
            + 0.020 * sin(2 * radians(mean_anomaly))
            + 282.634
        ) % 360

        right_ascension = (
            atan2(
                0.91764 * sin(radians(true_longitude)),
                cos(radians(true_longitude))
            ) * 180 / 3.141592653589793
        ) % 360

        right_ascension /= 15

        longitude_quadrant = int(true_longitude / 90) * 90
        right_ascension_quadrant = int(right_ascension * 15 / 90) * 90

        right_ascension += (
            longitude_quadrant - right_ascension_quadrant
        ) / 15

        sin_declination = 0.39782 * sin(radians(true_longitude))
        cos_declination = cos(
            atan2(sin_declination, sqrt(1 - sin_declination ** 2))
        )

        cos_hour_angle = (
            cos(radians(90.833))
            - sin_declination * sin(radians(latitude))
        ) / (cos_declination * cos(radians(latitude)))

        if not -1 <= cos_hour_angle <= 1:
            return None

        angle = (
            atan2(
                sqrt(1 - cos_hour_angle ** 2),
                cos_hour_angle
            ) * 180 / 3.141592653589793
        )

        hour_angle = 360 - angle if is_sunrise else angle
        hour_angle /= 15

        local_mean_time = (
            hour_angle + right_ascension
            - 0.06571 * approximate_time - 6.622
        )

        utc_hour = (local_mean_time - longitude_hour) % 24

        utc_midnight = datetime(
            date.year, date.month, date.day, tzinfo=timezone.utc
        )

        return utc_midnight + timedelta(hours=utc_hour)

    sunrise_utc = calculate_event(True)
    sunset_utc = calculate_event(False)

    if sunrise_utc is None or sunset_utc is None:
        return None, None

    return (
        sunrise_utc.astimezone(UK_TIMEZONE),
        sunset_utc.astimezone(UK_TIMEZONE)
    )


# ============================================================
# 4. LIGHT CONDITIONS
# ============================================================

def infer_light_conditions(
    latitude, longitude, road, current_time=None
):
    if current_time is None:
        current_time = datetime.now(UK_TIMEZONE)

    if current_time.tzinfo is None:
        current_time = current_time.replace(tzinfo=UK_TIMEZONE)
    else:
        current_time = current_time.astimezone(UK_TIMEZONE)

    sunrise, sunset = calculate_sunrise_sunset(
        latitude, longitude, current_time.date()
    )

    if sunrise is None or sunset is None:
        return None, {"reason": "Sunrise or sunset could not be calculated"}

    darkness_starts = sunset + timedelta(minutes=30)
    darkness_ends = sunrise - timedelta(minutes=30)

    is_dark = (
        current_time >= darkness_starts
        or current_time < darkness_ends
    )

    if not is_dark:
        return 1, {
            "reason": "Daylight based on calculated sunrise and sunset",
            "sunrise": sunrise.isoformat(),
            "sunset": sunset.isoformat()
        }

    if not road:
        return None, {"reason": "Darkness detected but no road found"}

    lighting = str(road.get("tags", {}).get("lit", "")).lower()

    if lighting == "yes":
        return 4, {
            "reason": "Darkness with OSM lit=yes",
            "lighting_tag": lighting,
            "sunrise": sunrise.isoformat(),
            "sunset": sunset.isoformat(),
            "note": "Actual lamp operation is not verified"
        }

    return None, {
        "reason": "Darkness detected but street-light status is unknown",
        "lighting_tag": lighting or None,
        "sunrise": sunrise.isoformat(),
        "sunset": sunset.isoformat()
    }


# ============================================================
# 5. RETRIEVE ROADS AND TRAFFIC FEATURES FROM OSM
# ============================================================

async def get_nearby_roads(latitude, longitude, radius=500):
    if not isinstance(latitude, (int, float)) or isinstance(latitude, bool):
        raise ValueError("latitude must be numeric")

    if not isinstance(longitude, (int, float)) or isinstance(longitude, bool):
        raise ValueError("longitude must be numeric")

    if not -90 <= latitude <= 90:
        raise ValueError("latitude must be between -90 and 90")

    if not -180 <= longitude <= 180:
        raise ValueError("longitude must be between -180 and 180")

    if not isinstance(radius, (int, float)) or isinstance(radius, bool) or radius <= 0:
        raise ValueError("radius must be positive")

    query = f"""
    [out:json][timeout:25];
    (
      way(around:{radius},{latitude},{longitude})["highway"];
      node(around:{radius},{latitude},{longitude})["highway"="traffic_signals"];
      node(around:{radius},{latitude},{longitude})["highway"="mini_roundabout"];
      node(around:{radius},{latitude},{longitude})["junction"="mini_roundabout"];
      node(around:{radius},{latitude},{longitude})["highway"="stop"];
      node(around:{radius},{latitude},{longitude})["highway"="give_way"];
      node(around:{radius},{latitude},{longitude})["highway"="crossing"];
    );
    out body;
    >;
    out skel qt;
    """

    async with httpx.AsyncClient(timeout=90) as client:
        response = await client.post(
            OVERPASS_URL,
            data={"data": query},
            headers={"User-Agent": "RoadGuardAI/1.0"}
        )

        if response.is_error:
            raise httpx.HTTPStatusError(
                f"Overpass API returned HTTP {response.status_code}: "
                f"{response.text[:1000]}",
                request=response.request,
                response=response
            )

        try:
            data = response.json()
        except ValueError as exc:
            raise ValueError("Overpass returned invalid JSON") from exc

    if not isinstance(data, dict) or "elements" not in data:
        raise ValueError("Overpass response did not contain elements")

    elements = data.get("elements", [])

    nodes = {
        element["id"]: element
        for element in elements
        if element.get("type") == "node" and "id" in element
    }

    roads = []
    traffic_signals = []
    mini_roundabouts = []
    stop_signs = []
    give_way_signs = []
    pedestrian_crossings = []

    for element in elements:
        if element.get("type") != "way":
            continue

        tags = element.get("tags", {})
        node_ids = element.get("nodes", [])

        geometry = [
            {
                "lat": nodes[node_id]["lat"],
                "lon": nodes[node_id]["lon"],
                "id": node_id
            }
            for node_id in node_ids
            if node_id in nodes
            and "lat" in nodes[node_id]
            and "lon" in nodes[node_id]
        ]

        roads.append({
            "id": element["id"],
            "name": tags.get("name"),
            "tags": tags,
            "nodes": node_ids,
            "geometry": geometry,
            "latitude": (
                sum(p["lat"] for p in geometry) / len(geometry)
                if geometry else None
            ),
            "longitude": (
                sum(p["lon"] for p in geometry) / len(geometry)
                if geometry else None
            )
        })

    for element in elements:
        if element.get("type") != "node":
            continue

        tags = element.get("tags", {})
        location = {
            "latitude": element.get("lat"),
            "longitude": element.get("lon")
        }

        if tags.get("highway") == "traffic_signals":
            traffic_signals.append({
                "name": tags.get("name", "Traffic signal"),
                "tags": tags,
                **location
            })

        elif (
            tags.get("highway") == "mini_roundabout"
            or tags.get("junction") == "mini_roundabout"
        ):
            mini_roundabouts.append({
                "name": tags.get("name", "Mini roundabout"),
                "type": "mini_roundabout",
                "tags": tags,
                **location
            })

        elif tags.get("highway") == "stop":
            stop_signs.append({
                "name": tags.get("name", "Stop sign"),
                "tags": tags,
                **location
            })

        elif tags.get("highway") == "give_way":
            give_way_signs.append({
                "name": tags.get("name", "Give way sign"),
                "tags": tags,
                **location
            })

        elif tags.get("highway") == "crossing":
            pedestrian_crossings.append({
                "id": element["id"],
                "name": tags.get("name", "Pedestrian crossing"),
                "tags": tags,
                **location
            })

    return {
        "roads": roads,
        "traffic_signals": traffic_signals,
        "mini_roundabouts": mini_roundabouts,
        "stop_signs": stop_signs,
        "give_way_signs": give_way_signs,
        "pedestrian_crossings": pedestrian_crossings
    }


# ============================================================
# 6. DISTANCE CALCULATIONS
# ============================================================

def calculate_distance(lat1, lon1, lat2, lon2):
    lat1, lon1 = radians(lat1), radians(lon1)
    lat2, lon2 = radians(lat2), radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        sin(dlat / 2) ** 2
        + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    )

    return EARTH_RADIUS * 2 * atan2(sqrt(a), sqrt(1 - a))


def point_to_segment_distance(latitude, longitude, start, end):
    latitude_radians = radians(latitude)

    start_x = (
        radians(start["lon"] - longitude)
        * EARTH_RADIUS * cos(latitude_radians)
    )
    start_y = radians(start["lat"] - latitude) * EARTH_RADIUS

    end_x = (
        radians(end["lon"] - longitude)
        * EARTH_RADIUS * cos(latitude_radians)
    )
    end_y = radians(end["lat"] - latitude) * EARTH_RADIUS

    segment_x = end_x - start_x
    segment_y = end_y - start_y
    length_squared = segment_x ** 2 + segment_y ** 2

    if length_squared == 0:
        return sqrt(start_x ** 2 + start_y ** 2)

    t = -(
        start_x * segment_x + start_y * segment_y
    ) / length_squared

    t = max(0, min(1, t))

    closest_x = start_x + t * segment_x
    closest_y = start_y + t * segment_y

    return sqrt(closest_x ** 2 + closest_y ** 2)


def distance_to_road(latitude, longitude, road):
    geometry = road.get("geometry", [])

    if not geometry:
        return None

    if len(geometry) == 1:
        point = geometry[0]
        return calculate_distance(
            latitude, longitude, point["lat"], point["lon"]
        )

    return min(
        point_to_segment_distance(
            latitude, longitude, geometry[index], geometry[index + 1]
        )
        for index in range(len(geometry) - 1)
    )


def find_nearest_road(latitude, longitude, roads):
    nearest_road = None
    minimum_distance = float("inf")

    for road in roads:
        distance = distance_to_road(latitude, longitude, road)

        if distance is not None and distance < minimum_distance:
            minimum_distance = distance
            nearest_road = road

    return {
        "road": nearest_road,
        "distance_meters": (
            round(minimum_distance, 2) if nearest_road else None
        )
    }


# ============================================================
# 7. ROAD TYPE AND ROAD FEATURE MAPPING
# ============================================================

def map_road_type(road):
    if not road:
        return None, "No road found"

    tags = road.get("tags", {})
    highway = str(tags.get("highway", "")).lower()
    junction = str(tags.get("junction", "")).lower()
    dual = str(tags.get("dual_carriageway", "")).lower()
    oneway = str(tags.get("oneway", "")).lower()

    if junction in {"roundabout", "circular"}:
        return 1, "OSM roundabout tag"

    if highway.endswith("_link"):
        return 7, "OSM link road; slip-road candidate"

    if dual == "yes":
        return 3, "Explicit dual_carriageway=yes"

    if dual == "no" and oneway in {"yes", "true", "1"}:
        return 2, "Explicit single-carriageway and one-way evidence"

    if dual == "no":
        return 6, "Explicit dual_carriageway=no"

    return None, "Insufficient road-type evidence"


def parse_speed_limit(value):
    if not value:
        return None

    match = re.search(r"\d+", str(value))
    return int(match.group()) if match else None


def extract_road_features(road):
    tags = road.get("tags", {})
    road_type, evidence = map_road_type(road)

    return {
        "road_name": road.get("name"),
        "osm_road_type": tags.get("highway"),
        "speed_limit": parse_speed_limit(tags.get("maxspeed")),
        "lanes": tags.get("lanes"),
        "street_lighting": tags.get("lit"),
        "surface_material": tags.get("surface"),
        "oneway": tags.get("oneway"),
        "dual_carriageway": tags.get("dual_carriageway"),
        "road_reference": tags.get("ref"),
        "junction": tags.get("junction"),
        "road_type": road_type,
        "road_type_evidence": evidence
    }


def map_first_road_class(road_reference):
    if not road_reference:
        return None

    references = re.split(r"[;,/]", str(road_reference).upper())
    detected_classes = []

    for raw_reference in references:
        reference = re.sub(r"\s+", "", raw_reference.strip())

        if not reference:
            continue

        if re.fullmatch(r"A\d+[A-Z]?\(M\)", reference):
            detected_classes.append(2)
        elif re.fullmatch(r"M\d+[A-Z]?", reference):
            detected_classes.append(1)
        elif re.fullmatch(r"A\d+[A-Z]?", reference):
            detected_classes.append(3)
        elif re.fullmatch(r"B\d+[A-Z]?", reference):
            detected_classes.append(4)
        elif re.fullmatch(r"C\d+[A-Z]?", reference):
            detected_classes.append(5)

    return min(detected_classes) if detected_classes else None


# ============================================================
# 8. ROAD IDENTITY AND JUNCTION DETECTION
# ============================================================

def get_road_identity(road):
    tags = road.get("tags", {})
    road_type = tags.get("highway")
    reference = tags.get("ref")
    name = tags.get("name")

    if reference:
        return ("reference", reference.upper().replace(" ", ""), road_type)

    if name:
        return ("name", name.strip().casefold(), road_type)

    return ("osm_id", road.get("id"))


def find_nearby_junctions(latitude, longitude, roads, radius=500):
    shared_nodes = {}

    for road in roads:
        if road.get("tags", {}).get("highway") not in VEHICLE_ROAD_TYPES:
            continue

        road_id = road.get("id")

        if road_id is None:
            continue

        for node_id in set(road.get("nodes", [])):
            shared_nodes.setdefault(node_id, {})[road_id] = road

    junctions = []

    for node_id, connected_roads in shared_nodes.items():
        if len(connected_roads) < 2:
            continue

        node_location = None

        for road in connected_roads.values():
            node_location = next(
                (
                    point for point in road.get("geometry", [])
                    if point.get("id") == node_id
                ),
                None
            )

            if node_location is not None:
                break

        if node_location is None:
            continue

        lat = node_location["lat"]
        lon = node_location["lon"]

        distance = calculate_distance(latitude, longitude, lat, lon)

        if distance > radius:
            continue

        unique_roads = {}
        connected_arm_count = 0

        for road in connected_roads.values():
            node_ids = road.get("nodes", [])

            for index, current_node in enumerate(node_ids):
                if current_node != node_id:
                    continue

                if index > 0:
                    connected_arm_count += 1

                if index < len(node_ids) - 1:
                    connected_arm_count += 1

            identity = get_road_identity(road)

            if identity not in unique_roads:
                unique_roads[identity] = {
                    "name": road.get("name"),
                    "road_type": road.get("tags", {}).get("highway"),
                    "reference": road.get("tags", {}).get("ref"),
                    "osm_way_ids": [road["id"]]
                }
            else:
                unique_roads[identity]["osm_way_ids"].append(road["id"])

        junctions.append({
            "node_id": node_id,
            "latitude": lat,
            "longitude": lon,
            "distance_meters": round(distance, 2),
            "connected_road_count": len(unique_roads),
            "connected_arm_count": connected_arm_count,
            "connected_roads": list(unique_roads.values())
        })

    junctions.sort(key=lambda item: item["distance_meters"])
    return junctions


def infer_junction_detail(
    latitude, longitude, roads, road_type=None, mini_roundabouts=None
):
    if road_type in (1, 7):
        return None

    mini_roundabouts = mini_roundabouts or []

    for item in mini_roundabouts:
        lat = item.get("latitude")
        lon = item.get("longitude")

        if lat is not None and lon is not None:
            if calculate_distance(latitude, longitude, lat, lon) <= 20:
                return 1

    junctions = find_nearby_junctions(
        latitude, longitude, roads, radius=20
    )

    if not junctions:
        return 0

    arms = junctions[0]["connected_arm_count"]

    if arms == 3:
        return 13

    if arms == 4:
        return 16

    if arms > 4:
        return 17

    return None


def infer_junction_control(
    junction_detail,
    junctions,
    traffic_signals=None,
    stop_signs=None,
    give_way_signs=None,
    radius=20
):
    if junction_detail in (None, 0):
        return None, "Junction control not established"

    if junction_detail not in (13, 16, 17, 18, 19):
        return None, "Junction control not applicable"

    if not junctions:
        return None, "No junction location available"

    nearest = junctions[0]
    junction_lat = nearest["latitude"]
    junction_lon = nearest["longitude"]

    controls = {
        2: traffic_signals or [],
        3: stop_signs or [],
        4: give_way_signs or []
    }

    detected_codes = set()

    for code, items in controls.items():
        for item in items:
            lat = item.get("latitude")
            lon = item.get("longitude")

            if lat is None or lon is None:
                continue

            distance = calculate_distance(
                junction_lat, junction_lon, lat, lon
            )

            if distance <= radius:
                detected_codes.add(code)

    if len(detected_codes) > 1:
        return None, "Conflicting control evidence"

    if len(detected_codes) == 1:
        code = detected_codes.pop()
        descriptions = {
            2: "Nearby OSM traffic signal",
            3: "Nearby OSM stop sign",
            4: "Nearby OSM give-way sign"
        }
        return code, descriptions[code]

    return None, "No reliable junction-control evidence"


# ============================================================
# 9. SECOND ROAD CLASS
# ============================================================

def infer_second_road_class(
    latitude, longitude, roads, first_road, junction_detail
):
    if junction_detail in (None, 0):
        return None, "Second road class not established"

    if junction_detail == 18:
        return 6, "STATS20 private drive/entrance rule"

    if junction_detail not in (13, 16, 17, 19):
        return None, "Unsupported junction detail"

    junctions = find_nearby_junctions(
        latitude, longitude, roads, radius=20
    )

    if not junctions:
        return None, "No nearby junction identified"

    junction_node_id = junctions[0]["node_id"]
    first_identity = get_road_identity(first_road) if first_road else None

    candidate_classes = []

    for road in roads:
        if junction_node_id not in road.get("nodes", []):
            continue

        if road.get("tags", {}).get("highway") not in VEHICLE_ROAD_TYPES:
            continue

        identity = get_road_identity(road)

        if identity == first_identity:
            continue

        reference = road.get("tags", {}).get("ref")
        road_class = map_first_road_class(reference)

        if road_class is not None:
            candidate_classes.append({
                "road_class": road_class,
                "road_name": road.get("name"),
                "road_reference": reference,
                "road_identity": identity
            })

    if not candidate_classes:
        return None, "No reliably classified second road found"

    highest_class = min(item["road_class"] for item in candidate_classes)

    selected = [
        item for item in candidate_classes
        if item["road_class"] == highest_class
    ]

    return highest_class, {
        "reason": "Highest class among connected second roads",
        "candidate_roads": selected
    }


# ============================================================
# 10. PEDESTRIAN CROSSING
# ============================================================

def infer_pedestrian_crossing(
    latitude,
    longitude,
    roads,
    pedestrian_crossings=None,
    junction_detail=None,
    radius=50
):
    pedestrian_crossings = pedestrian_crossings or []
    candidates = []

    for crossing in pedestrian_crossings:
        crossing_lat = crossing.get("latitude")
        crossing_lon = crossing.get("longitude")

        if crossing_lat is None or crossing_lon is None:
            continue

        distance = calculate_distance(
            latitude, longitude, crossing_lat, crossing_lon
        )

        if distance > radius:
            continue

        tags = crossing.get("tags", {})
        crossing_type = tags.get("crossing", "")
        crossing_island = tags.get("crossing:island", "")

        code = None
        evidence = None

        if crossing_type == "zebra":
            code = 13
            evidence = "OSM zebra crossing tag"

        elif crossing_type in {
            "traffic_signals", "pelican", "puffin", "toucan"
        }:
            if junction_detail in (13, 16, 17, 18, 19):
                code = 15
                evidence = "Signal-controlled crossing at a detected junction"
            elif junction_detail == 0:
                code = 14
                evidence = "Signal-controlled crossing away from junction"

        elif crossing_island == "yes":
            code = 17
            evidence = "OSM pedestrian crossing island tag"

        if code is not None:
            candidates.append({
                "code": code,
                "distance_meters": round(distance, 2),
                "evidence": evidence,
                "crossing": crossing
            })

    for road in roads:
        tags = road.get("tags", {})

        if tags.get("highway") not in PEDESTRIAN_WAY_TYPES:
            continue

        bridge = tags.get("bridge", "")
        tunnel = tags.get("tunnel", "")

        if bridge not in ("yes", "viaduct") and tunnel not in (
            "yes", "building_passage"
        ):
            continue

        distance = distance_to_road(latitude, longitude, road)

        if distance is None or distance > radius:
            continue

        candidates.append({
            "code": 16,
            "distance_meters": round(distance, 2),
            "evidence": "OSM pedestrian bridge or subway",
            "crossing": {
                "id": road.get("id"),
                "name": road.get("name"),
                "tags": tags
            }
        })

    if not candidates:
        return None, "No mapped crossing found"

    candidates.sort(key=lambda item: item["distance_meters"])
    nearest = candidates[0]

    equally_near = [
        item for item in candidates
        if abs(item["distance_meters"] - nearest["distance_meters"]) < 1
    ]

    codes = {item["code"] for item in equally_near}

    if len(codes) > 1:
        return None, {
            "reason": "Conflicting crossing types",
            "candidates": equally_near
        }

    return nearest["code"], {
        "reason": nearest["evidence"],
        "distance_meters": nearest["distance_meters"],
        "candidate": nearest["crossing"]
    }


# ============================================================
# 11. BUILD MODEL FEATURES
# ============================================================

def build_model_feature_candidates(
    latitude,
    longitude,
    roads,
    traffic_signals=None,
    stop_signs=None,
    give_way_signs=None,
    pedestrian_crossings=None,
    weather_data=None,
    road_surface_override=None,
    mini_roundabouts=None
):
    traffic_signals = traffic_signals or []
    stop_signs = stop_signs or []
    give_way_signs = give_way_signs or []
    pedestrian_crossings = pedestrian_crossings or []
    mini_roundabouts = mini_roundabouts or []

    vehicle_roads = [
        road for road in roads
        if road.get("tags", {}).get("highway") in VEHICLE_ROAD_TYPES
    ]

    nearest = find_nearest_road(latitude, longitude, vehicle_roads)
    road = nearest["road"]

    features = {feature: None for feature in MODEL_FEATURES}

    features["latitude"] = latitude
    features["longitude"] = longitude
    features.update(get_current_time_features())

    road_details = None

    if road:
        road_details = extract_road_features(road)

        features["speed_limit"] = road_details["speed_limit"]

        features["first_road_class"] = map_first_road_class(
            road_details["road_reference"]
        )

        features["road_type"] = road_details["road_type"]

    nearby_junctions = find_nearby_junctions(
        latitude, longitude, vehicle_roads, radius=500
    )

    features["junction_detail"] = infer_junction_detail(
        latitude,
        longitude,
        vehicle_roads,
        road_type=features["road_type"],
        mini_roundabouts=mini_roundabouts
    )

    junction_control, junction_control_evidence = infer_junction_control(
        junction_detail=features["junction_detail"],
        junctions=nearby_junctions,
        traffic_signals=traffic_signals,
        stop_signs=stop_signs,
        give_way_signs=give_way_signs
    )

    features["junction_control"] = junction_control

    second_road_class, second_road_class_evidence = infer_second_road_class(
        latitude,
        longitude,
        vehicle_roads,
        first_road=road,
        junction_detail=features["junction_detail"]
    )

    features["second_road_class"] = second_road_class

    pedestrian_crossing, pedestrian_crossing_evidence = (
        infer_pedestrian_crossing(
            latitude,
            longitude,
            roads,
            pedestrian_crossings=pedestrian_crossings,
            junction_detail=features["junction_detail"]
        )
    )

    features["pedestrian_crossing"] = pedestrian_crossing

    weather_conditions, weather_conditions_evidence = (
        map_weather_to_stats20(weather_data)
    )

    features["weather_conditions"] = weather_conditions

    valid_surface_codes = {1, 2, 3, 4, 5, 9}

    if road_surface_override is not None:
        if (
            isinstance(road_surface_override, bool)
            or road_surface_override not in valid_surface_codes
        ):
            raise ValueError(
                "road_surface_override must be one of 1, 2, 3, 4, 5, or 9"
            )

        features["road_surface_conditions"] = int(road_surface_override)

        road_surface_conditions_evidence = {
            "source": "user selection",
            "code": int(road_surface_override)
        }
    else:
        features["road_surface_conditions"] = None

        road_surface_conditions_evidence = {
            "reason": "Road surface condition was not supplied"
        }

    # These fields cannot reliably be derived from OSM alone.
    features["urban_or_rural_area"] = None
    features["trunk_road_flag"] = None

    area_and_trunk_evidence = {
        "urban_or_rural_area": (
            "Official geographic classification not available"
        ),
        "trunk_road_flag": (
            "Authoritative trunk-road membership not available"
        )
    }

    light_conditions, light_conditions_evidence = infer_light_conditions(
        latitude, longitude, road
    )

    features["light_conditions"] = light_conditions

    referenced_roads = [
        candidate for candidate in vehicle_roads
        if candidate.get("tags", {}).get("ref")
    ]

    nearest_referenced = find_nearest_road(
        latitude, longitude, referenced_roads
    )

    referenced_road_details = None

    if nearest_referenced["road"]:
        referenced_road_details = extract_road_features(
            nearest_referenced["road"]
        )

    # Apply valid unknown/unallocated categories only where the
    # trained model's own category mapping explicitly supports them.
    fallback_evidence = apply_unknown_fallback(features)

    missing_features = [
        feature for feature, value in features.items()
        if value is None
    ]

    return {
        "features": features,
        "nearest_road": road_details,
        "nearest_road_distance_meters": nearest["distance_meters"],
        "nearest_referenced_road": referenced_road_details,
        "nearest_referenced_road_distance_meters": (
            nearest_referenced["distance_meters"]
        ),
        "nearby_junctions": nearby_junctions,
        "junction_control_evidence": junction_control_evidence,
        "second_road_class_evidence": second_road_class_evidence,
        "pedestrian_crossing_evidence": pedestrian_crossing_evidence,
        "light_conditions_evidence": light_conditions_evidence,
        "weather_conditions_evidence": weather_conditions_evidence,
        "road_surface_conditions_evidence": road_surface_conditions_evidence,
        "area_and_trunk_evidence": area_and_trunk_evidence,
        "fallback_evidence": fallback_evidence,
        "missing_features": missing_features,
        "ready_for_prediction": len(missing_features) == 0
    }
