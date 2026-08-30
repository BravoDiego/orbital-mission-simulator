import numpy as np
import pytest

from src.physics.bodies import EARTH
from src.physics.constants import EARTH_RADIUS

from src.mission.hohmann import (
    create_circular_orbit_state,
    prepare_hohmann_transfer,
    orbital_period,
    simulate_hohmann_mission,
)

from src.orbital.transfers import (
    circular_velocity,
)

def test_final_orbital_period():

    r = 42164e3

    period = orbital_period(
        r,
        EARTH,
    )

    # GEO period is approximately one sidereal day
    assert period / 3600.0 == pytest.approx(
        23.934,
        rel=1e-3,
    )

def test_numerical_hohmann_reaches_target_radius():

    r1 = EARTH_RADIUS + 300e3
    r2 = 42164e3

    mission = simulate_hohmann_mission(
        r1=r1,
        r2=r2,
        body=EARTH,
        dt=10.0,
    )

    arrival_position = (
        mission.transfer_result
        .final_state
        .position
    )

    arrival_radius = np.linalg.norm(
        arrival_position
    )

    assert arrival_radius == pytest.approx(
        r2,
        rel=1e-5,
    )

def test_numerical_hohmann_arrives_at_opposite_apsis():

    r1 = EARTH_RADIUS + 300e3
    r2 = 42164e3

    mission = simulate_hohmann_mission(
        r1=r1,
        r2=r2,
        body=EARTH,
        dt=10.0,
    )

    position = (
        mission.transfer_result
        .final_state
        .position
    )

    assert position[0] == pytest.approx(
        -r2,
        rel=1e-5,
    )

    assert position[1] == pytest.approx(
        0.0,
        abs=1e3,
    )

    assert position[2] == pytest.approx(
        0.0,
        abs=1e-9,
    )

def test_transfer_arrival_velocity_matches_theory():

    r1 = EARTH_RADIUS + 300e3
    r2 = 42164e3

    mission = simulate_hohmann_mission(
        r1=r1,
        r2=r2,
        body=EARTH,
        dt=10.0,
    )

    numerical_speed = np.linalg.norm(
        mission.transfer_result
        .final_state
        .velocity
    )

    theoretical_speed = (
        mission.setup
        .transfer
        .v_transfer_2
    )

    assert numerical_speed == pytest.approx(
        theoretical_speed,
        rel=1e-5,
    )

def test_burn2_circularizes_final_orbit():

    r1 = EARTH_RADIUS + 300e3
    r2 = 42164e3

    mission = simulate_hohmann_mission(
        r1=r1,
        r2=r2,
        body=EARTH,
        dt=10.0,
    )

    speed_after_burn2 = np.linalg.norm(
        mission.burn2
        .state_after
        .velocity
    )

    circular_speed = circular_velocity(
        r2,
        EARTH.mu,
    )

    assert speed_after_burn2 == pytest.approx(
        circular_speed,
        rel=1e-5,
    )

def test_burn2_circularizes_final_orbit():

    r1 = EARTH_RADIUS + 300e3
    r2 = 42164e3

    mission = simulate_hohmann_mission(
        r1=r1,
        r2=r2,
        body=EARTH,
        dt=10.0,
    )

    speed_after_burn2 = np.linalg.norm(
        mission.burn2
        .state_after
        .velocity
    )

    circular_speed = circular_velocity(
        r2,
        EARTH.mu,
    )

    assert speed_after_burn2 == pytest.approx(
        circular_speed,
        rel=1e-5,
    )

def test_leo_to_geo_delta_v_budget():

    r1 = EARTH_RADIUS + 300e3
    r2 = 42164e3

    mission = simulate_hohmann_mission(
        r1=r1,
        r2=r2,
        body=EARTH,
        dt=10.0,
    )

    dv1 = (
        mission.setup
        .transfer
        .delta_v1
    )

    dv2 = (
        mission.setup
        .transfer
        .delta_v2
    )

    total = (
        mission.setup
        .transfer
        .delta_v_total
    )

    assert total == pytest.approx(
        abs(dv1) + abs(dv2)
    )

    assert total == pytest.approx(
        3892.55,
        rel=1e-4,
    )