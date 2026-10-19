import numpy as np
import pytest

from src.physics.atmosphere import (
    earth_exponential_density,
)


def test_reference_density_at_400_km():

    density = earth_exponential_density(
        400e3
    )

    assert density == pytest.approx(
        3.725e-12,
        rel=1e-12,
    )


def test_density_decreases_with_altitude():

    altitudes = np.array([
        0.0,
        100e3,
        200e3,
        400e3,
        700e3,
        1000e3,
    ])

    densities = np.array([
        earth_exponential_density(
            altitude
        )
        for altitude in altitudes
    ])

    assert np.all(
        np.diff(densities) < 0.0
    )


def test_density_rejects_negative_altitude():

    with pytest.raises(
        ValueError
    ):
        earth_exponential_density(
            -1.0
        )


def test_density_above_table_remains_positive():

    density = earth_exponential_density(
        1200e3
    )

    assert np.isfinite(
        density
    )

    assert density > 0.0
