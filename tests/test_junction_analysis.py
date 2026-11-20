
import pytest

from backend.road_data import (
    infer_junction_detail,
    find_nearby_junctions,
)


# --------------------------------------------------
# HELPER: Create an OSM road
# --------------------------------------------------

def create_road(road_id, name, node_ids, coordinates, **tags):
    return {
        "id": road_id,
        "name": name,
        "tags": {
            "highway": "residential",
            "name": name,
            **tags,
        },
        "nodes": node_ids,
        "geometry": [
            {
                "id": node_id,
                "lat": lat,
                "lon": lon,
            }
            for node_id, lat, lon in coordinates
        ],
    }


# --------------------------------------------------
# TEST 1: T-Junction
# One continuous road + one joining road = 3 arms
# Expected STATS20 code: 13
# --------------------------------------------------

def test_t_junction():

    main_road = create_road(
        101,
        "Main Road",
        [1, 1000, 2],
        [
            (1, -0.001, 0.0),
            (1000, 0.0, 0.0),
            (2, 0.001, 0.0),
        ],
    )

    side_road = create_road(
        102,
        "Side Road",
        [1000, 3],
        [
            (1000, 0.0, 0.0),
            (3, 0.0, 0.001),
        ],
    )

    roads = [main_road, side_road]

    result = infer_junction_detail(
        latitude=0.0,
        longitude=0.0,
        roads=roads,
    )

    assert result == 13


# --------------------------------------------------
# TEST 2: Crossroads
# Two continuous roads = 4 arms
# Expected STATS20 code: 16
# --------------------------------------------------

def test_crossroads():

    north_south_road = create_road(
        201,
        "North South Road",
        [1, 1000, 2],
        [
            (1, -0.001, 0.0),
            (1000, 0.0, 0.0),
            (2, 0.001, 0.0),
        ],
    )

    east_west_road = create_road(
        202,
        "East West Road",
        [3, 1000, 4],
        [
            (3, 0.0, -0.001),
            (1000, 0.0, 0.0),
            (4, 0.0, 0.001),
        ],
    )

    roads = [north_south_road, east_west_road]

    result = infer_junction_detail(
        latitude=0.0,
        longitude=0.0,
        roads=roads,
    )

    assert result == 16


# --------------------------------------------------
# TEST 3: Roundabout
# STATS20 junction detail is not applicable
# --------------------------------------------------

def test_roundabout():

    result = infer_junction_detail(
        latitude=0.0,
        longitude=0.0,
        roads=[],
        road_type=1,
    )

    assert result is None


# --------------------------------------------------
# HELPER: Create grade-separated crossing roads
# They cross geometrically but have no shared node.
# --------------------------------------------------

def create_grade_separated_roads(structure_tag):

    road_a = create_road(
        301,
        "Road A",
        [1, 2],
        [
            (1, -0.001, 0.0),
            (2, 0.001, 0.0),
        ],
        **{structure_tag: "yes"},
        layer="1" if structure_tag == "bridge" else "-1",
    )

    road_b = create_road(
        302,
        "Road B",
        [3, 4],
        [
            (3, 0.0, -0.001),
            (4, 0.0, 0.001),
        ],
    )

    return [road_a, road_b]


# --------------------------------------------------
# TEST 4: Bridge crossing
# No shared OSM node = no junction
# --------------------------------------------------

def test_bridge_crossing():

    roads = create_grade_separated_roads("bridge")

    junctions = find_nearby_junctions(
        latitude=0.0,
        longitude=0.0,
        roads=roads,
        radius=20,
    )

    assert junctions == []

    result = infer_junction_detail(
        latitude=0.0,
        longitude=0.0,
        roads=roads,
    )

    assert result == 0


# --------------------------------------------------
# TEST 5: Tunnel crossing
# No shared OSM node = no junction
# --------------------------------------------------

def test_tunnel_crossing():

    roads = create_grade_separated_roads("tunnel")

    junctions = find_nearby_junctions(
        latitude=0.0,
        longitude=0.0,
        roads=roads,
        radius=20,
    )

    assert junctions == []

    result = infer_junction_detail(
        latitude=0.0,
        longitude=0.0,
        roads=roads,
    )

    assert result == 0
