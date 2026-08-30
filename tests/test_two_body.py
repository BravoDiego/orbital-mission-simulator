import numpy as np
import pytest

from src.physics.bodies import EARTH
from src.physics.two_body import gravitational_acceleration


def test_gravity_points_towards_center():
    position = np.array([
        2.0 * EARTH.radius,
        0.0,
        0.0,
    ])

    acceleration = gravitational_acceleration(
        position,
        EARTH,
    )

    assert acceleration[0] < 0

    assert acceleration[1] == pytest.approx(0.0)
    assert acceleration[2] == pytest.approx(0.0)


def test_gravity_magnitude():
    r = EARTH.radius

    position = np.array([
        r,
        0.0,
        0.0,
    ])

    acceleration = gravitational_acceleration(
        position,
        EARTH,
    )

    expected = EARTH.mu / r**2

    assert np.linalg.norm(
        acceleration
    ) == pytest.approx(
        expected,
        rel=1e-12,
    )


def test_gravity_at_center_raises_error():
    with pytest.raises(ValueError):
        gravitational_acceleration(
            np.zeros(3),
            EARTH,
        )