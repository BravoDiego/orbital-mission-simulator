import numpy as np
import pytest

from src.orbital.state import OrbitalState
from src.orbital.elements import argument_of_periapsis
from src.orbital.conversions import state_from_classical_elements
from src.orbital.precession import (
    j2_apsidal_precession_rate,
    j2_apsidal_precession_rate_from_elements,
    j2_critical_inclinations,
)
from src.physics.bodies import EARTH
from src.physics.constants import EARTH_J2
from src.physics.dynamics import build_state_derivative
from src.physics.perturbations.j2 import make_j2_perturbation
from src.mission.mission import Mission


def create_elliptical_state(
    inclination=np.deg2rad(45.0),
):
    return state_from_classical_elements(
        semi_major_axis=(
            EARTH.radius
            + 1500e3
        ),
        eccentricity=0.05,
        inclination=inclination,
        raan=np.deg2rad(25.0),
        argument_of_periapsis=np.deg2rad(40.0),
        true_anomaly=np.deg2rad(10.0),
        body=EARTH,
    )


def test_critical_inclinations_cancel_first_order_apsidal_rate():
    critical_1, critical_2 = j2_critical_inclinations()

    assert np.rad2deg(critical_1) == pytest.approx(
        63.4349488,
        abs=1e-7,
    )
    assert np.rad2deg(critical_2) == pytest.approx(
        116.5650512,
        abs=1e-7,
    )

    for inclination in (
        critical_1,
        critical_2,
    ):
        rate = j2_apsidal_precession_rate_from_elements(
            semi_major_axis=(
                EARTH.radius
                + 1500e3
            ),
            eccentricity=0.05,
            inclination=inclination,
            body=EARTH,
            j2=EARTH_J2,
        )

        assert rate == pytest.approx(
            0.0,
            abs=1e-18,
        )


def test_j2_apsidal_rate_changes_sign_across_critical_inclination():
    low_state = create_elliptical_state(
        inclination=np.deg2rad(45.0)
    )
    high_state = create_elliptical_state(
        inclination=np.deg2rad(80.0)
    )

    low_rate = j2_apsidal_precession_rate(
        low_state,
        EARTH,
        EARTH_J2,
    )
    high_rate = j2_apsidal_precession_rate(
        high_state,
        EARTH,
        EARTH_J2,
    )

    assert low_rate > 0.0
    assert high_rate < 0.0


def test_numerical_j2_apsidal_rate_matches_theory():
    state = create_elliptical_state(
        inclination=np.deg2rad(45.0)
    )

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
        3.0 * 86400.0
    )

    omega_history = []

    for position, velocity in zip(
        result.positions,
        result.velocities,
    ):
        sample_state = OrbitalState(
            position=position,
            velocity=velocity,
        )

        omega = argument_of_periapsis(
            sample_state,
            EARTH,
        )

        assert omega is not None
        omega_history.append(omega)

    omega_unwrapped = np.unwrap(
        np.asarray(omega_history)
    )

    numerical_rate = np.polyfit(
        result.times,
        omega_unwrapped,
        deg=1,
    )[0]

    theoretical_rate = j2_apsidal_precession_rate(
        state,
        EARTH,
        EARTH_J2,
    )

    assert numerical_rate == pytest.approx(
        theoretical_rate,
        rel=2e-2,
    )
