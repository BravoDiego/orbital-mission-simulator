import numpy as np
import pytest

from src.orbital.state import OrbitalState

from src.physics.bodies import EARTH

from src.physics.constants import (
    EARTH_J2,
)

from src.physics.perturbations.j2 import (
    j2_acceleration,
    make_j2_perturbation,
)

from src.physics.dynamics import (
    build_state_derivative,
)

from src.mission.mission import (
    Mission,
)


def create_equatorial_state():

    radius = (
        EARTH.radius
        + 700e3
    )

    speed = np.sqrt(
        EARTH.mu / radius
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


def test_zero_j2_gives_zero_acceleration():

    state = create_equatorial_state()

    acceleration = j2_acceleration(
        state,
        EARTH,
        j2=0.0,
    )

    assert np.allclose(
        acceleration,
        np.zeros(3),
    )


def test_equatorial_j2_direction():

    state = create_equatorial_state()

    acceleration = j2_acceleration(
        state,
        EARTH,
        EARTH_J2,
    )

    # At x > 0, z = 0:
    # J2 correction points in -x.
    assert acceleration[0] < 0.0

    assert acceleration[1] == pytest.approx(
        0.0,
        abs=1e-15,
    )

    assert acceleration[2] == pytest.approx(
        0.0,
        abs=1e-15,
    )


def test_equatorial_j2_matches_formula():

    state = create_equatorial_state()

    radius = np.linalg.norm(
        state.position
    )

    acceleration = j2_acceleration(
        state,
        EARTH,
        EARTH_J2,
    )

    expected_ax = (
        -1.5
        * EARTH_J2
        * EARTH.mu
        * EARTH.radius**2
        / radius**4
    )

    assert acceleration[0] == pytest.approx(
        expected_ax,
        rel=1e-12,
    )


def test_j2_acceleration_is_small_relative_to_gravity():

    state = create_equatorial_state()

    radius = np.linalg.norm(
        state.position
    )

    central_gravity = (
        EARTH.mu
        / radius**2
    )

    j2_magnitude = np.linalg.norm(
        j2_acceleration(
            state,
            EARTH,
            EARTH_J2,
        )
    )

    assert j2_magnitude > 0.0

    assert (
        j2_magnitude
        < 0.01
        * central_gravity
    )


def test_j2_mission_can_propagate():

    state = create_equatorial_state()

    j2 = make_j2_perturbation(
        EARTH_J2
    )

    derivative = build_state_derivative(
        body=EARTH,
        perturbations=[
            j2
        ],
    )

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
        derivative=derivative,
    )

    mission.coast(
        600.0
    )

    result = mission.result()

    assert np.all(
        np.isfinite(
            result.final_state.position
        )
    )

    assert np.all(
        np.isfinite(
            result.final_state.velocity
        )
    )