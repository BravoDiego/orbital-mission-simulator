import numpy as np
import pytest

from src.orbital.state import OrbitalState
from src.orbital.elements import (
    compute_orbital_elements,
    right_ascension_of_ascending_node,
)
from src.orbital.precession import (
    j2_nodal_precession_rate,
)
from src.physics.bodies import EARTH
from src.physics.constants import EARTH_J2
from src.physics.dynamics import build_state_derivative
from src.physics.perturbations.j2 import make_j2_perturbation
from src.mission.mission import Mission


def rotation_z(angle):
    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def create_inclined_circular_state(
    altitude=700e3,
    inclination=np.deg2rad(63.4),
    raan=0.0,
):
    radius = EARTH.radius + altitude
    speed = np.sqrt(EARTH.mu / radius)

    position = np.array([
        radius,
        0.0,
        0.0,
    ])

    velocity = np.array([
        0.0,
        speed * np.cos(inclination),
        speed * np.sin(inclination),
    ])

    rotation = rotation_z(
        raan
    )

    return OrbitalState(
        position=rotation @ position,
        velocity=rotation @ velocity,
    )


def test_raan_of_known_rotated_orbit():
    expected_raan = np.deg2rad(60.0)

    state = create_inclined_circular_state(
        raan=expected_raan,
    )

    raan = right_ascension_of_ascending_node(
        state
    )

    assert raan == pytest.approx(
        expected_raan,
        abs=1e-12,
    )


def test_equatorial_orbit_has_no_raan():
    radius = EARTH.radius + 700e3
    speed = np.sqrt(EARTH.mu / radius)

    state = OrbitalState(
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

    elements = compute_orbital_elements(
        state,
        EARTH,
    )

    assert elements.raan is None
    assert elements.argument_of_periapsis is None
    assert elements.true_anomaly is None


def test_j2_nodal_rate_is_negative_for_prograde_orbit():
    state = create_inclined_circular_state()

    rate = j2_nodal_precession_rate(
        state,
        EARTH,
        EARTH_J2,
    )

    assert rate < 0.0


def test_numerical_j2_raan_rate_matches_theory():
    state = create_inclined_circular_state()

    derivative = build_state_derivative(
        body=EARTH,
        perturbations=[
            make_j2_perturbation(
                EARTH_J2
            )
        ],
    )

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=30.0,
        derivative=derivative,
    )

    result = mission.coast(
        24.0 * 3600.0
    )

    raan_history = []

    for position, velocity in zip(
        result.positions,
        result.velocities,
    ):
        sample_state = OrbitalState(
            position=position,
            velocity=velocity,
        )

        raan = right_ascension_of_ascending_node(
            sample_state
        )

        assert raan is not None
        raan_history.append(raan)

    raan_unwrapped = np.unwrap(
        np.asarray(
            raan_history
        )
    )

    numerical_rate = np.polyfit(
        result.times,
        raan_unwrapped,
        deg=1,
    )[0]

    theoretical_rate = j2_nodal_precession_rate(
        state,
        EARTH,
        EARTH_J2,
    )

    assert numerical_rate == pytest.approx(
        theoretical_rate,
        rel=1e-2,
    )
