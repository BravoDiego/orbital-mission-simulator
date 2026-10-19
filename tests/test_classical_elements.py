import numpy as np
import pytest

from src.orbital.state import OrbitalState

from src.orbital.elements import (
    compute_orbital_elements,
)

from src.physics.bodies import EARTH


def rotation_z(
    angle,
):

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def rotation_x(
    angle,
):
    c = np.cos(angle)
    s = np.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def state_from_classical_elements(
    semi_major_axis,
    eccentricity,
    inclination,
    raan,
    argument_of_periapsis,
    true_anomaly,
):

    p = (
        semi_major_axis
        * (
            1.0
            - eccentricity**2
        )
    )

    radius = (
        p
        / (
            1.0
            + eccentricity
            * np.cos(
                true_anomaly
            )
        )
    )

    position_perifocal = np.array([
        radius
        * np.cos(
            true_anomaly
        ),

        radius
        * np.sin(
            true_anomaly
        ),

        0.0,
    ])

    velocity_scale = np.sqrt(
        EARTH.mu
        / p
    )

    velocity_perifocal = (
        velocity_scale
        * np.array([
            -np.sin(
                true_anomaly
            ),

            eccentricity
            + np.cos(
                true_anomaly
            ),

            0.0,
        ])
    )

    rotation = (
        rotation_z(
            raan
        )
        @ rotation_x(
            inclination
        )
        @ rotation_z(
            argument_of_periapsis
        )
    )

    return OrbitalState(
        position=(
            rotation
            @ position_perifocal
        ),

        velocity=(
            rotation
            @ velocity_perifocal
        ),
    )


def test_full_classical_elements_are_recovered():

    expected_a = 10_000_000.0

    expected_e = 0.2

    expected_i = np.deg2rad(
        50.0
    )

    expected_raan = np.deg2rad(
        30.0
    )

    expected_omega = np.deg2rad(
        40.0
    )

    expected_nu = np.deg2rad(
        70.0
    )

    state = state_from_classical_elements(
        semi_major_axis=expected_a,
        eccentricity=expected_e,
        inclination=expected_i,
        raan=expected_raan,
        argument_of_periapsis=(
            expected_omega
        ),
        true_anomaly=expected_nu,
    )

    elements = (
        compute_orbital_elements(
            state,
            EARTH,
        )
    )

    assert (
        elements.semi_major_axis
        == pytest.approx(
            expected_a,
            rel=1e-12,
        )
    )

    assert (
        elements.eccentricity
        == pytest.approx(
            expected_e,
            rel=1e-12,
        )
    )

    assert (
        elements.inclination
        == pytest.approx(
            expected_i,
            abs=1e-12,
        )
    )

    assert (
        elements.raan
        == pytest.approx(
            expected_raan,
            abs=1e-12,
        )
    )

    assert (
        elements.argument_of_periapsis
        == pytest.approx(
            expected_omega,
            abs=1e-12,
        )
    )

    assert (
        elements.true_anomaly
        == pytest.approx(
            expected_nu,
            abs=1e-12,
        )
    )