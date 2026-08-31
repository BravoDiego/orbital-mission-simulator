import numpy as np
import pytest

from src.orbital.state import OrbitalState
from src.physics.bodies import EARTH
from src.physics.constants import EARTH_RADIUS

from src.orbital.transfers import (
    circular_velocity,
)

from src.mission.mission import (
    Mission,
)


def create_test_state():

    radius = (
        EARTH_RADIUS
        + 400e3
    )

    speed = circular_velocity(
        radius,
        EARTH.mu,
    )

    return OrbitalState(
        position=np.array([
            radius,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            0.0,
            speed,
            0.0,
        ]),
    )


def test_mission_accepts_custom_initial_state():

    state = OrbitalState(
        position=np.array([
            0.0,
            8.0e6,
            0.0,
        ]),
        velocity=np.array([
            -7000.0,
            500.0,
            0.0,
        ]),
    )

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
    )

    result = mission.result()

    assert np.allclose(
        result.initial_state.position,
        state.position,
    )

    assert np.allclose(
        result.initial_state.velocity,
        state.velocity,
    )

    assert np.allclose(
        result.final_state.position,
        state.position,
    )

    assert np.allclose(
        result.final_state.velocity,
        state.velocity,
    )


def test_coast_advances_mission_time():

    state = create_test_state()

    mission = Mission(
        state,
        EARTH,
        dt=10.0,
    )

    mission.coast(
        120.0
    )

    assert mission.elapsed_time == pytest.approx(
        120.0
    )


def test_coast_changes_position():

    state = create_test_state()

    mission = Mission(
        state,
        EARTH,
        dt=10.0,
    )

    position_before = (
        state.position.copy()
    )

    mission.coast(
        120.0
    )

    position_after = (
        mission
        .current_state
        .position
    )

    assert not np.allclose(
        position_before,
        position_after,
    )


def test_coast_preserves_circular_orbit_radius():

    state = create_test_state()

    initial_radius = np.linalg.norm(
        state.position
    )

    mission = Mission(
        state,
        EARTH,
        dt=10.0,
    )

    mission.coast(
        600.0
    )

    final_radius = np.linalg.norm(
        mission.current_state.position
    )

    assert final_radius == pytest.approx(
        initial_radius,
        rel=1e-8,
    )


def test_burn_does_not_advance_time():

    state = create_test_state()

    mission = Mission(
        state,
        EARTH,
    )

    mission.prograde_burn(
        100.0
    )

    assert mission.elapsed_time == pytest.approx(
        0.0
    )


def test_burn_does_not_change_position():

    state = create_test_state()

    mission = Mission(
        state,
        EARTH,
    )

    position_before = (
        mission
        .current_state
        .position
        .copy()
    )

    mission.prograde_burn(
        100.0
    )

    position_after = (
        mission
        .current_state
        .position
    )

    assert np.allclose(
        position_before,
        position_after,
    )


def test_multiple_burn_delta_v_budget():

    state = create_test_state()

    mission = Mission(
        state,
        EARTH,
    )

    mission.prograde_burn(
        100.0
    )

    mission.retrograde_burn(
        50.0
    )

    mission.tangential_burn(
        -25.0
    )

    result = mission.result()

    assert result.total_delta_v == pytest.approx(
        175.0
    )


def test_mission_phase_order():

    state = create_test_state()

    mission = Mission(
        state,
        EARTH,
        dt=10.0,
    )

    mission.prograde_burn(
        100.0,
        label="Departure burn",
    )

    mission.coast(
        300.0,
        label="Transfer coast",
    )

    mission.retrograde_burn(
        50.0,
        label="Arrival burn",
    )

    result = mission.result()

    assert len(
        result.phases
    ) == 3

    assert result.phases[0].kind == "burn"
    assert result.phases[1].kind == "coast"
    assert result.phases[2].kind == "burn"

    assert (
        result.phases[0].label
        == "Departure burn"
    )

    assert (
        result.phases[1].label
        == "Transfer coast"
    )

    assert (
        result.phases[2].label
        == "Arrival burn"
    )


def test_multiple_coasts_accumulate_time():

    state = create_test_state()

    mission = Mission(
        state,
        EARTH,
        dt=10.0,
    )

    mission.coast(
        100.0
    )

    mission.coast(
        250.0
    )

    mission.coast(
        50.0
    )

    result = mission.result()

    assert result.elapsed_time == pytest.approx(
        400.0
    )


def test_trajectory_positions_are_combined():

    state = create_test_state()

    mission = Mission(
        state,
        EARTH,
        dt=10.0,
    )

    mission.coast(
        100.0
    )

    mission.prograde_burn(
        50.0
    )

    mission.coast(
        100.0
    )

    result = mission.result()

    positions = (
        result
        .trajectory_positions()
    )

    times = (
        result
        .trajectory_times()
    )

    assert positions.ndim == 2

    assert positions.shape[1] == 3

    assert len(positions) == len(times)

    assert times[0] == pytest.approx(
        0.0
    )

    assert times[-1] == pytest.approx(
        200.0
    )