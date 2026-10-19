import numpy as np
import pytest

from src.orbital.state import OrbitalState
from src.orbital.conversions import state_from_classical_elements
from src.physics.bodies import EARTH, MOON
from src.physics.constants import MOON_MEAN_ORBIT_RADIUS
from src.physics.dynamics import build_state_derivative
from src.physics.ephemeris import moon_circular_ephemeris
from src.physics.perturbations.third_body import (
    third_body_acceleration,
    make_third_body_perturbation,
)
from src.mission.mission import Mission


def test_third_body_acceleration_is_zero_at_origin():
    state = OrbitalState(
        position=np.zeros(3),
        velocity=np.zeros(3),
    )

    moon_position = np.array([
        MOON_MEAN_ORBIT_RADIUS,
        0.0,
        0.0,
    ])

    acceleration = third_body_acceleration(
        state=state,
        third_body_position=moon_position,
        third_body_mu=MOON.mu,
    )

    assert np.allclose(
        acceleration,
        np.zeros(3),
        atol=1e-18,
    )


def test_third_body_acceleration_matches_differential_formula():
    spacecraft_x = 20_000_000.0
    moon_x = MOON_MEAN_ORBIT_RADIUS

    state = OrbitalState(
        position=np.array([
            spacecraft_x,
            0.0,
            0.0,
        ]),
        velocity=np.zeros(3),
    )

    moon_position = np.array([
        moon_x,
        0.0,
        0.0,
    ])

    acceleration = third_body_acceleration(
        state=state,
        third_body_position=moon_position,
        third_body_mu=MOON.mu,
    )

    expected_x = MOON.mu * (
        1.0 / (moon_x - spacecraft_x)**2
        - 1.0 / moon_x**2
    )

    assert acceleration[0] == pytest.approx(
        expected_x,
        rel=1e-12,
    )

    assert acceleration[1] == pytest.approx(
        0.0,
        abs=1e-18,
    )

    assert acceleration[2] == pytest.approx(
        0.0,
        abs=1e-18,
    )


def test_moon_perturbation_depends_on_time():
    ephemeris = moon_circular_ephemeris()

    perturbation = make_third_body_perturbation(
        third_body=MOON,
        position_function=ephemeris,
    )

    state = OrbitalState(
        position=np.array([
            50_000_000.0,
            10_000_000.0,
            0.0,
        ]),
        velocity=np.zeros(3),
    )

    acceleration_0 = perturbation(
        state,
        EARTH,
        0.0,
    )

    acceleration_quarter = perturbation(
        state,
        EARTH,
        ephemeris.period / 4.0,
    )

    assert not np.allclose(
        acceleration_0,
        acceleration_quarter,
        rtol=1e-6,
        atol=1e-12,
    )


def test_lunar_third_body_changes_high_orbit():
    state = state_from_classical_elements(
        semi_major_axis=70_000_000.0,
        eccentricity=0.3,
        inclination=np.deg2rad(30.0),
        raan=np.deg2rad(40.0),
        argument_of_periapsis=np.deg2rad(20.0),
        true_anomaly=0.0,
        body=EARTH,
    )

    ephemeris = moon_circular_ephemeris()

    derivative = build_state_derivative(
        body=EARTH,
        perturbations=[
            make_third_body_perturbation(
                third_body=MOON,
                position_function=ephemeris,
            )
        ],
    )

    two_body = Mission(
        initial_state=state,
        body=EARTH,
        dt=300.0,
    )

    lunar = Mission(
        initial_state=state,
        body=EARTH,
        dt=300.0,
        derivative=derivative,
    )

    two_body.coast(
        5.0 * 86400.0
    )

    lunar.coast(
        5.0 * 86400.0
    )

    separation = np.linalg.norm(
        lunar.current_state.position
        - two_body.current_state.position
    )

    assert separation > 1000.0
