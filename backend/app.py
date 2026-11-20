import os
import httpx

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from ml.predict import predict_severity
from backend import road_data


# =========================================================
# ROADGUARD AI - FASTAPI BACKEND
# =========================================================

app = FastAPI(
    title="RoadGuard AI API",
    description=(
        "Location-based road-collision severity candidate analysis "
        "and experimental severity prediction."
    ),
    version="2.0.0",
)

# Set FRONTEND_ORIGINS in deployment, e.g.
# FRONTEND_ORIGINS=https://your-site.vercel.app
frontend_origins = os.getenv("FRONTEND_ORIGINS", "*")
allowed_origins = [
    origin.strip()
    for origin in frontend_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_HEADERS = {
    "User-Agent": (
        "RoadGuardAI/2.0 "
        "(https://github.com/vinay123-create/ROAD_AI)"
    ),
    "Accept": "application/json",
}
OVERPASS_URL = "https://overpass.atownsend.org.uk/api/interpreter"


# =========================================================
# REQUEST MODELS
# =========================================================

class RoadGuardRequest(BaseModel):
    location: str = Field(min_length=2, max_length=200)
    # STATS20 road_surface_conditions codes observed in training:
    # 1, 2, 3, 4, 5, 9
    road_surface_conditions: Optional[int] = None


# =========================================================
# SHARED HELPERS
# =========================================================

async def geocode_uk_location(query: str) -> dict:
    """Resolve a postcode, address, town, or road within the UK."""
    params = {
        "q": query,
        "format": "jsonv2",
        "countrycodes": "gb",
        "limit": 1,
    }

    try:
        async with httpx.AsyncClient(
            timeout=20,
            headers=NOMINATIM_HEADERS,
        ) as client:
            response = await client.get(
                NOMINATIM_URL,
                params=params,
            )
            response.raise_for_status()
            results = response.json()

    except httpx.HTTPError as exc:
        raise HTTPException(
            status_code=502,
            detail="The UK location service is temporarily unavailable.",
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=502,
            detail="The location service returned invalid data.",
        ) from exc

    if not results:
        raise HTTPException(
            status_code=404,
            detail="No matching UK location was found.",
        )

    result = results[0]
    return {
        "location": result.get("display_name", query),
        "latitude": float(result["lat"]),
        "longitude": float(result["lon"]),
    }


async def collect_location_candidates(
    location_query: str,
    road_surface_override: Optional[int] = None,
) -> dict:
    """
    Geocode the location, collect OSM/weather data, and build the
    18 model-feature candidates. This function does NOT invent values
    for features that the current data sources cannot establish.
    """
    if road_surface_override is not None:
        if isinstance(road_surface_override, bool) or road_surface_override not in {
            1, 2, 3, 4, 5, 9
        }:
            raise HTTPException(
                status_code=422,
                detail=(
                    "road_surface_conditions must be one of "
                    "1, 2, 3, 4, 5, or 9."
                ),
            )

    resolved = await geocode_uk_location(location_query)
    latitude = resolved["latitude"]
    longitude = resolved["longitude"]

    # Retrieve road-network information.
    try:
        osm_data = await road_data.get_nearby_roads(
            latitude,
            longitude,
            radius=500,
        )
    except (httpx.HTTPError, ValueError) as exc:
        raise HTTPException(
            status_code=502,
            detail=f"Unable to retrieve nearby road data: {str(exc)}",
        ) from exc

    # Weather is useful, but a temporary weather-service failure should
    # not erase the road analysis. The missing feature will be reported.
    weather = None
    weather_error = None
    try:
        weather = await road_data.get_current_weather(
            latitude,
            longitude,
        )
    except (httpx.HTTPError, ValueError) as exc:
        weather_error = str(exc)

    try:
        result = road_data.build_model_feature_candidates(
            latitude=latitude,
            longitude=longitude,
            roads=osm_data.get("roads", []),
            traffic_signals=osm_data.get("traffic_signals", []),
            stop_signs=osm_data.get("stop_signs", []),
            give_way_signs=osm_data.get("give_way_signs", []),
            pedestrian_crossings=osm_data.get("pedestrian_crossings", []),
            weather_data=weather,
            road_surface_override=road_surface_override,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    features = result["features"]
    missing_features = [
        name for name, value in features.items()
        if value is None
    ]

    # This checks completeness only. It does not certify the quality
    # or safety of the inferred feature values.
    can_run_model = len(missing_features) == 0

    return {
        "location": resolved["location"],
        "latitude": latitude,
        "longitude": longitude,
        "features": features,
        "missing_features": missing_features,
        "can_run_model": can_run_model,
        "nearest_road": result.get("nearest_road"),
        "nearest_road_distance_meters": result.get(
            "nearest_road_distance_meters"
        ),
        "evidence": {
            "junction_control": result.get("junction_control_evidence"),
            "second_road_class": result.get("second_road_class_evidence"),
            "pedestrian_crossing": result.get(
                "pedestrian_crossing_evidence"
            ),
            "light_conditions": result.get("light_conditions_evidence"),
            "weather_conditions": result.get(
                "weather_conditions_evidence"
            ),
            "road_surface_conditions": result.get(
                "road_surface_conditions_evidence"
            ),
            "urban_and_trunk": result.get("area_and_trunk_evidence"),
            "weather_service_error": weather_error,
        },
        "notice": (
            "This is an experimental prediction system. A complete set "
            "of features does not mean the result is safety-certified."
        ),
    }



import os
import httpx

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ml.predict import predict_severity
from backend import road_data


# =========================================================
# ROADGUARD AI - FASTAPI BACKEND
# =========================================================

app = FastAPI(
    title="RoadGuard AI API",
    description=(
        "Location-based road-collision severity analysis, "
        "experimental severity prediction, and nearby "
        "UK hospital finder."
    ),
    version="2.1.0",
)


# =========================================================
# CORS CONFIGURATION
# =========================================================

frontend_origins = os.getenv("FRONTEND_ORIGINS", "*")

allowed_origins = [
    origin.strip()
    for origin in frontend_origins.split(",")
    if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins or ["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


# =========================================================
# EXTERNAL API CONFIGURATION
# =========================================================

NOMINATIM_URL = (
    "https://nominatim.openstreetmap.org/search"
)

NOMINATIM_REVERSE_URL = (
    "https://nominatim.openstreetmap.org/reverse"
)

NOMINATIM_HEADERS = {
    "User-Agent": (
        "RoadGuardAI/2.1 "
        "(https://github.com/vinay123-create/ROAD_AI)"
    ),
    "Accept": "application/json",
}

OVERPASS_URL = (
    "https://overpass.atownsend.org.uk/api/interpreter"
)


# Approximate UK geographic bounding box.
# Reverse geocoding is also used for country verification.

UK_BOUNDS = {
    "min_lat": 49.8,
    "max_lat": 60.9,
    "min_lon": -8.7,
    "max_lon": 2.1,
}


# =========================================================
# REQUEST MODELS
# =========================================================

class RoadGuardRequest(BaseModel):

    location: str = Field(
        min_length=2,
        max_length=200,
    )

    road_surface_conditions: Optional[int] = None


# =========================================================
# 1. UK LOCATION SEARCH
# =========================================================

async def geocode_uk_location(query: str) -> dict:

    """
    Resolve a postcode, address, town, or road
    within the United Kingdom.
    """

    params = {
        "q": query,
        "format": "jsonv2",
        "countrycodes": "gb",
        "limit": 1,
    }

    try:

        async with httpx.AsyncClient(
            timeout=20,
            headers=NOMINATIM_HEADERS,
        ) as client:

            response = await client.get(
                NOMINATIM_URL,
                params=params,
            )

            response.raise_for_status()

            results = response.json()

    except httpx.HTTPError as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                "The UK location service "
                "is temporarily unavailable."
            ),
        ) from exc

    except ValueError as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                "The location service returned invalid data."
            ),
        ) from exc

    if not results:

        raise HTTPException(
            status_code=404,
            detail="No matching UK location was found.",
        )

    result = results[0]

    return {
        "location": result.get(
            "display_name",
            query,
        ),
        "latitude": float(result["lat"]),
        "longitude": float(result["lon"]),
    }


# =========================================================
# 2. UK COORDINATE VALIDATION
# =========================================================

async def validate_uk_coordinates(
    latitude: float,
    longitude: float,
):

    """
    Verify that the selected coordinates belong
    to the United Kingdom.
    """

    # Basic coordinate validation

    if not -90 <= latitude <= 90:

        raise HTTPException(
            status_code=400,
            detail="Invalid latitude.",
        )

    if not -180 <= longitude <= 180:

        raise HTTPException(
            status_code=400,
            detail="Invalid longitude.",
        )

    # Quick geographic bounds validation

    if not (
        UK_BOUNDS["min_lat"] <= latitude <= UK_BOUNDS["max_lat"]
        and
        UK_BOUNDS["min_lon"] <= longitude <= UK_BOUNDS["max_lon"]
    ):

        raise HTTPException(
            status_code=422,
            detail=(
                "Hospital searches are restricted "
                "to the United Kingdom."
            ),
        )

    # Verify actual country using reverse geocoding

    params = {
        "lat": latitude,
        "lon": longitude,
        "format": "jsonv2",
        "zoom": 10,
        "addressdetails": 1,
    }

    try:

        async with httpx.AsyncClient(
            timeout=20,
            headers=NOMINATIM_HEADERS,
        ) as client:

            response = await client.get(
                NOMINATIM_REVERSE_URL,
                params=params,
            )

            response.raise_for_status()

            result = response.json()

    except httpx.HTTPError as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                "Unable to verify the selected location."
            ),
        ) from exc

    except ValueError as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                "Invalid response from location service."
            ),
        ) from exc

    country_code = (
        result.get("address", {})
        .get("country_code", "")
        .lower()
    )

    if country_code != "gb":

        raise HTTPException(
            status_code=422,
            detail=(
                "Selected location is not within "
                "the United Kingdom."
            ),
        )

    return True


# =========================================================
# 3. LOCATION-BASED MODEL FEATURE COLLECTION
# =========================================================

async def collect_location_candidates(
    location_query: str,
    road_surface_override: Optional[int] = None,
) -> dict:

    """
    Geocode the location, collect OSM/weather data,
    and construct the 18 model-feature candidates.

    Does not invent unavailable feature values.
    """

    # Validate optional road surface input

    if road_surface_override is not None:

        if (
            isinstance(road_surface_override, bool)
            or road_surface_override not in {
                1, 2, 3, 4, 5, 9
            }
        ):

            raise HTTPException(
                status_code=422,
                detail=(
                    "road_surface_conditions must be "
                    "one of 1, 2, 3, 4, 5, or 9."
                ),
            )

    # Geocode location

    resolved = await geocode_uk_location(
        location_query
    )

    latitude = resolved["latitude"]
    longitude = resolved["longitude"]

    # Retrieve nearby road-network information

    try:

        osm_data = await road_data.get_nearby_roads(
            latitude,
            longitude,
            radius=500,
        )

    except (httpx.HTTPError, ValueError) as exc:

        raise HTTPException(
            status_code=502,
            detail=(
                f"Unable to retrieve nearby road data: {str(exc)}"
            ),
        ) from exc

    # Retrieve current weather

    weather = None
    weather_error = None

    try:

        weather = await road_data.get_current_weather(
            latitude,
            longitude,
        )

    except (httpx.HTTPError, ValueError) as exc:

        weather_error = str(exc)

    # Construct model features

    try:

        result = road_data.build_model_feature_candidates(

            latitude=latitude,

            longitude=longitude,

            roads=osm_data.get("roads", []),

            traffic_signals=osm_data.get(
                "traffic_signals", []
            ),

            stop_signs=osm_data.get(
                "stop_signs", []
            ),

            give_way_signs=osm_data.get(
                "give_way_signs", []
            ),

            pedestrian_crossings=osm_data.get(
                "pedestrian_crossings", []
            ),

            weather_data=weather,

            road_surface_override=road_surface_override,

        )

    except ValueError as exc:

        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc

    features = result["features"]

    missing_features = [
        name
        for name, value in features.items()
        if value is None
    ]

    can_run_model = len(missing_features) == 0

    return {

        "location": resolved["location"],

        "latitude": latitude,

        "longitude": longitude,

        "features": features,

        "missing_features": missing_features,

        "can_run_model": can_run_model,

        "nearest_road": result.get(
            "nearest_road"
        ),

        "nearest_road_distance_meters": result.get(
            "nearest_road_distance_meters"
        ),

        "evidence": {

            "junction_control": result.get(
                "junction_control_evidence"
            ),

            "second_road_class": result.get(
                "second_road_class_evidence"
            ),

            "pedestrian_crossing": result.get(
                "pedestrian_crossing_evidence"
            ),

            "light_conditions": result.get(
                "light_conditions_evidence"
            ),

            "weather_conditions": result.get(
                "weather_conditions_evidence"
            ),

            "road_surface_conditions": result.get(
                "road_surface_conditions_evidence"
            ),

            "urban_and_trunk": result.get(
                "area_and_trunk_evidence"
            ),

            "weather_service_error": weather_error,

        },

        "notice": (
            "This is an experimental prediction system. "
            "A complete set of features does not mean "
            "the result is safety-certified."
        ),

    }


# =========================================================
# 4. HEALTH CHECK
# =========================================================

@app.get("/")
def home():

    return {

        "message": "RoadGuard AI API is running",

        "version": app.version,

    }


@app.get("/health")
def health():

    return {
        "status": "ok"
    }


# =========================================================
# 5. UK LOCATION SEARCH ENDPOINT
# =========================================================

@app.get("/location/search")
async def search_location(
    query: str = Query(
        ...,
        min_length=2,
        max_length=200,
    ),
):

    return await geocode_uk_location(query)


# =========================================================
# 6. LOCATION-BASED FEATURE ANALYSIS
# =========================================================

@app.post("/analyze")
async def analyze_location(
    data: RoadGuardRequest,
):

    """
    Frontend sends a UK location and optionally
    a road-surface selection.

    Backend constructs model feature candidates.
    """

    return await collect_location_candidates(

        location_query=data.location,

        road_surface_override=(
            data.road_surface_conditions
        ),

    )


# =========================================================
# 7. LOCATION-BASED SEVERITY PREDICTION
# =========================================================

@app.post("/predict")
async def predict_from_location(
    data: RoadGuardRequest,
):

    result = await collect_location_candidates(

        location_query=data.location,

        road_surface_override=(
            data.road_surface_conditions
        ),

    )

    # Check missing model features

    if not result["can_run_model"]:

        raise HTTPException(

            status_code=409,

            detail={

                "message": (
                    "A prediction cannot be generated yet "
                    "because some required model features "
                    "are unavailable."
                ),

                "missing_features": (
                    result["missing_features"]
                ),

                "features": result["features"],

                "evidence": result["evidence"],

            },

        )

    # Run Random Forest prediction

    try:

        prediction = predict_severity(
            result["features"]
        )

    except (ValueError, TypeError) as exc:

        raise HTTPException(
            status_code=422,
            detail=(
                f"Model input validation failed: {str(exc)}"
            ),
        ) from exc

    return {

        "location": result["location"],

        "latitude": result["latitude"],

        "longitude": result["longitude"],

        "predicted_severity": prediction,

        "features": result["features"],

        "evidence": result["evidence"],

        "notice": result["notice"],

        "interpretation": (
            "Predicted severity conditional on a collision "
            "occurring; this does not predict whether "
            "a collision will occur."
        ),

    }


# =========================================================
# 8. NEARBY UK HOSPITAL FINDER
# =========================================================

@app.get("/hospitals/nearby")
async def find_nearby_hospitals(

    latitude: float = Query(...),

    longitude: float = Query(...),

    radius: int = Query(
        5000,
        ge=100,
        le=50000,
    ),

):

    """
    Find nearby hospitals using OpenStreetMap Overpass API.

    Only UK locations are accepted.

    Default search radius: 5000 metres.
    """

    # -----------------------------------------------------
    # STEP 1: VERIFY UK LOCATION
    # -----------------------------------------------------

    await validate_uk_coordinates(
        latitude,
        longitude,
    )

    # -----------------------------------------------------
    # STEP 2: BUILD OVERPASS QUERY
    # -----------------------------------------------------

    query = f"""

    [out:json][timeout:25];

    (
        nwr["amenity"="hospital"]
        (around:{radius},{latitude},{longitude});

        nwr["healthcare"="hospital"]
        (around:{radius},{latitude},{longitude});
    );

    out center tags;

    """

    # -----------------------------------------------------
    # STEP 3: FETCH HOSPITAL DATA
    # -----------------------------------------------------

    try:

        async with httpx.AsyncClient(

            timeout=60,

            headers=NOMINATIM_HEADERS,

        ) as client:

            response = await client.post(

                OVERPASS_URL,

                data={
                    "data": query
                },

            )

            response.raise_for_status()

            payload = response.json()

    except httpx.HTTPError as exc:

        raise HTTPException(

            status_code=502,

            detail=(
                "Unable to retrieve nearby UK hospital data."
            ),

        ) from exc

    except ValueError as exc:

        raise HTTPException(

            status_code=502,

            detail=(
                "The hospital service returned invalid data."
            ),

        ) from exc

    # -----------------------------------------------------
    # STEP 4: PROCESS HOSPITAL DATA
    # -----------------------------------------------------

    hospitals = []

    seen_locations = set()

    for item in payload.get("elements", []):

        hospital_lat = item.get("lat")

        hospital_lon = item.get("lon")

        # Ways and relations may use center coordinates

        if (
            hospital_lat is None
            or hospital_lon is None
        ):

            center = item.get("center") or {}

            hospital_lat = center.get("lat")

            hospital_lon = center.get("lon")

        if (
            hospital_lat is None
            or hospital_lon is None
        ):

            continue

        hospital_lat = float(hospital_lat)

        hospital_lon = float(hospital_lon)

        # Remove duplicate coordinates

        location_key = (

            round(hospital_lat, 6),

            round(hospital_lon, 6),

        )

        if location_key in seen_locations:

            continue

        seen_locations.add(location_key)

        tags = item.get("tags") or {}

        hospitals.append({

            "name": tags.get(
                "name",
                "Unnamed Hospital",
            ),

            "latitude": hospital_lat,

            "longitude": hospital_lon,

            "phone": (

                tags.get("phone")

                or tags.get("contact:phone")

            ),

        })

    # -----------------------------------------------------
    # STEP 5: RETURN RESULTS
    # -----------------------------------------------------

    return {

        "country": "United Kingdom",

        "search_center": {

            "latitude": latitude,

            "longitude": longitude,

        },

        "search_radius_meters": radius,

        "count": len(hospitals),

        "hospitals": hospitals,

        "data_source": "OpenStreetMap",

    }
