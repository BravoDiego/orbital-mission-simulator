import numpy as np
import pytest

from src.orbital.elements import (
    compute_orbital_elements,
)

from src.orbital.conversions import (
    state_from_classical_elements,
)

from src.physics.bodies import EARTH


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
        body=EARTH,
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