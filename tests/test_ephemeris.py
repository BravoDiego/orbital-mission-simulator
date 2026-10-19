import numpy as np
import pytest

from src.physics.ephemeris import CircularEphemeris


def test_circular_ephemeris_quarter_period():
    ephemeris = CircularEphemeris(
        radius=10.0,
        period=4.0,
    )

    assert np.allclose(
        ephemeris.position(0.0),
        np.array([
            10.0,
            0.0,
            0.0,
        ]),
        atol=1e-12,
    )

    assert np.allclose(
        ephemeris.position(1.0),
        np.array([
            0.0,
            10.0,
            0.0,
        ]),
        atol=1e-12,
    )


def test_circular_ephemeris_preserves_radius():
    ephemeris = CircularEphemeris(
        radius=42.0,
        period=9.0,
        inclination=np.deg2rad(30.0),
        raan=np.deg2rad(20.0),
    )

    for time in np.linspace(
        0.0,
        20.0,
        15,
    ):
        assert np.linalg.norm(
            ephemeris.position(time)
        ) == pytest.approx(
            42.0,
            rel=1e-12,
        )
