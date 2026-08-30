import numpy as np
import pytest

from src.orbital.elements import orbital_elements
from src.orbital.state import OrbitalState
from src.physics.bodies import EARTH


def test_circular_orbit_elements():
    r = (
        EARTH.radius
        + 500_000.0
    )

    v = np.sqrt(
        EARTH.mu / r
    )

    state = OrbitalState(
        position=np.array([
            r,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            0.0,
            v,
            0.0,
        ]),
    )

    elements = orbital_elements(
        state,
        EARTH,
    )

    assert elements.semi_major_axis == pytest.approx(
        r,
        rel=1e-10,
    )

    assert elements.eccentricity == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert elements.inclination == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert elements.periapsis_radius == pytest.approx(
        r,
        rel=1e-10,
    )

    assert elements.apoapsis_radius == pytest.approx(
        r,
        rel=1e-10,
    )

import numpy as np
import pytest

from src.orbital.elements import orbital_elements
from src.orbital.state import OrbitalState
from src.physics.bodies import EARTH


def test_circular_orbit_elements():
    r = (
        EARTH.radius
        + 500_000.0
    )

    v = np.sqrt(
        EARTH.mu / r
    )

    state = OrbitalState(
        position=np.array([
            r,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            0.0,
            v,
            0.0,
        ]),
    )

    elements = orbital_elements(
        state,
        EARTH,
    )

    assert elements.semi_major_axis == pytest.approx(
        r,
        rel=1e-10,
    )

    assert elements.eccentricity == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert elements.inclination == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert elements.periapsis_radius == pytest.approx(
        r,
        rel=1e-10,
    )

    assert elements.apoapsis_radius == pytest.approx(
        r,
        rel=1e-10,
    )

def test_elliptical_orbit_elements():
    semi_major_axis = 10_000_000.0
    eccentricity = 0.2

    periapsis = (
        semi_major_axis
        * (1.0 - eccentricity)
    )

    velocity_at_periapsis = np.sqrt(
        EARTH.mu
        * (
            2.0 / periapsis
            - 1.0 / semi_major_axis
        )
    )

    state = OrbitalState(
        position=np.array([
            periapsis,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            0.0,
            velocity_at_periapsis,
            0.0,
        ]),
    )

    elements = orbital_elements(
        state,
        EARTH,
    )

    expected_apoapsis = (
        semi_major_axis
        * (1.0 + eccentricity)
    )

    assert elements.semi_major_axis == pytest.approx(
        semi_major_axis,
        rel=1e-10,
    )

    assert elements.eccentricity == pytest.approx(
        eccentricity,
        rel=1e-10,
    )

    assert elements.periapsis_radius == pytest.approx(
        periapsis,
        rel=1e-10,
    )

    assert elements.apoapsis_radius == pytest.approx(
        expected_apoapsis,
        rel=1e-10,
    )