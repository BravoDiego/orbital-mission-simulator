import numpy as np
import pytest

from src.mission.maneuver import (
    apply_delta_v,
    prograde_burn,
    retrograde_burn,
    prograde_direction,
)

from src.physics.constants import (
    EARTH_RADIUS,
    EARTH_MU,
)

from src.orbital.conservation import specific_orbital_energy
from src.physics.bodies import EARTH

from src.orbital.transfers import circular_velocity

def test_apply_delta_v():

    state = np.array([
        7.0e6,
        0.0,
        0.0,
        7500.0,
    ])

    delta_v = np.array([
        0.0,
        100.0,
    ])

    result = apply_delta_v(
        state,
        delta_v,
    )

    assert result.state_after[0] == pytest.approx(
        state[0]
    )

    assert result.state_after[1] == pytest.approx(
        state[1]
    )

    assert result.state_after[2] == pytest.approx(
        0.0
    )

    assert result.state_after[3] == pytest.approx(
        7600.0
    )

    assert result.delta_v_magnitude == pytest.approx(
        100.0
    )

def test_prograde_burn():

    state = np.array([
        7.0e6,
        0.0,
        0.0,
        7500.0,
    ])

    result = prograde_burn(
        state,
        100.0,
    )

    assert result.state_after[2] == pytest.approx(
        0.0
    )

    assert result.state_after[3] == pytest.approx(
        7600.0
    )

def test_retrograde_burn():

    state = np.array([
        7.0e6,
        0.0,
        0.0,
        7500.0,
    ])

    result = retrograde_burn(
        state,
        100.0,
    )

    assert result.state_after[2] == pytest.approx(
        0.0
    )

    assert result.state_after[3] == pytest.approx(
        7400.0
    )

def test_prograde_direction():

    state = np.array([
        1.0,
        2.0,
        3.0,
        4.0,
    ])

    direction = prograde_direction(
        state
    )

    expected = np.array([
        3.0 / 5.0,
        4.0 / 5.0,
    ])

    assert np.allclose(
        direction,
        expected
    )

r = EARTH_RADIUS + 400e3

def test_prograde_burn_increases_speed():

    r = EARTH_RADIUS + 400e3

    v = circular_velocity(
        r,
        EARTH_MU
    )

    state = np.array([
        r,
        0.0,
        0.0,
        v,
    ])

    result = prograde_burn(
        state,
        100.0
    )

    speed_before = np.linalg.norm(
        state[2:4]
    )

    speed_after = np.linalg.norm(
        result.state_after[2:4]
    )

    assert speed_after > speed_before

    assert speed_after - speed_before == pytest.approx(
        100.0
    )

def energy_from_array(state, mu):
    position = state[:2]
    velocity = state[2:]

    r = np.linalg.norm(position)
    v = np.linalg.norm(velocity)

    return 0.5 * v**2 - mu / r

def test_prograde_burn_increases_energy():

    r = EARTH_RADIUS + 400e3

    v = circular_velocity(
        r,
        EARTH_MU
    )

    state = np.array([
        r,
        0.0,
        0.0,
        v,
    ])

    energy_before = energy_from_array(
        state,
        EARTH_MU
    )

    result = prograde_burn(
        state,
        100.0
    )

    energy_after = energy_from_array(
        result.state_after,
        EARTH_MU
    )

    assert energy_after > energy_before

def test_retrograde_burn_decreases_energy():

    r = EARTH_RADIUS + 400e3

    v = circular_velocity(
        r,
        EARTH_MU
    )

    state = np.array([
        r,
        0.0,
        0.0,
        v,
    ])

    energy_before = energy_from_array(
        state,
        EARTH_MU
    )

    result = retrograde_burn(
        state,
        100.0
    )

    energy_after = energy_from_array(
        result.state_after,
        EARTH_MU
    )

    assert energy_after < energy_before