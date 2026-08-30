import numpy as np
import pytest

from src.mission.maneuver import (
    apply_delta_v,
    prograde_burn,
    retrograde_burn,
    prograde_direction,
    tangential_burn,
)

from src.physics.constants import (
    EARTH_RADIUS,
    EARTH_MU,
)

from src.orbital.conservation import specific_orbital_energy
from src.physics.bodies import EARTH
from src.orbital.transfers import circular_velocity
from src.orbital.state import OrbitalState


def test_apply_delta_v():

    state = OrbitalState(
        position=np.array([
            7.0e6,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            0.0,
            7500.0,
            0.0,
        ]),
    )

    delta_v = np.array([
        0.0,
        100.0,
        0.0,
    ])

    result = apply_delta_v(
        state,
        delta_v,
    )

    # Position must remain unchanged
    assert np.allclose(
        result.state_after.position,
        state.position,
    )

    # Velocity must receive the delta-v
    assert np.allclose(
        result.state_after.velocity,
        np.array([
            0.0,
            7600.0,
            0.0,
        ])
    )

    assert result.delta_v_magnitude == pytest.approx(
        100.0
    )


def test_prograde_direction():

    state = OrbitalState(
        position=np.array([
            7.0e6,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            3.0,
            4.0,
            0.0,
        ]),
    )

    direction = prograde_direction(state)

    expected = np.array([
        3.0 / 5.0,
        4.0 / 5.0,
        0.0,
    ])

    assert np.allclose(
        direction,
        expected,
    )

    # It must be a unit vector
    assert np.linalg.norm(direction) == pytest.approx(
        1.0
    )


def test_prograde_burn():

    state = OrbitalState(
        position=np.array([
            7.0e6,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            0.0,
            7500.0,
            0.0,
        ]),
    )

    result = prograde_burn(
        state,
        100.0,
    )

    assert np.allclose(
        result.state_after.velocity,
        np.array([
            0.0,
            7600.0,
            0.0,
        ])
    )


def test_retrograde_burn():

    state = OrbitalState(
        position=np.array([
            7.0e6,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            0.0,
            7500.0,
            0.0,
        ]),
    )

    result = retrograde_burn(
        state,
        100.0,
    )

    assert np.allclose(
        result.state_after.velocity,
        np.array([
            0.0,
            7400.0,
            0.0,
        ])
    )


def test_tangential_burn_signed_delta_v():

    state = OrbitalState(
        position=np.array([
            7.0e6,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            0.0,
            7500.0,
            0.0,
        ]),
    )

    prograde = tangential_burn(
        state,
        100.0,
    )

    retrograde = tangential_burn(
        state,
        -100.0,
    )

    assert prograde.state_after.velocity[1] == pytest.approx(
        7600.0
    )

    assert retrograde.state_after.velocity[1] == pytest.approx(
        7400.0
    )


def test_prograde_burn_increases_speed():

    r = EARTH_RADIUS + 400e3

    v = circular_velocity(
        r,
        EARTH_MU,
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

    result = prograde_burn(
        state,
        100.0,
    )

    speed_before = np.linalg.norm(
        state.velocity
    )

    speed_after = np.linalg.norm(
        result.state_after.velocity
    )

    assert speed_after > speed_before

    assert speed_after - speed_before == pytest.approx(
        100.0
    )


def test_retrograde_burn_decreases_speed():

    r = EARTH_RADIUS + 400e3

    v = circular_velocity(
        r,
        EARTH_MU,
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

    result = retrograde_burn(
        state,
        100.0,
    )

    speed_before = np.linalg.norm(
        state.velocity
    )

    speed_after = np.linalg.norm(
        result.state_after.velocity
    )

    assert speed_after < speed_before

    assert speed_before - speed_after == pytest.approx(
        100.0
    )


def test_prograde_burn_increases_energy():

    r = EARTH_RADIUS + 400e3

    v = circular_velocity(
        r,
        EARTH_MU,
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

    energy_before = specific_orbital_energy(
        state,
        EARTH,
    )

    result = prograde_burn(
        state,
        100.0,
    )

    energy_after = specific_orbital_energy(
        result.state_after,
        EARTH,
    )

    assert energy_after > energy_before


def test_retrograde_burn_decreases_energy():

    r = EARTH_RADIUS + 400e3

    v = circular_velocity(
        r,
        EARTH_MU,
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

    energy_before = specific_orbital_energy(
        state,
        EARTH,
    )

    result = retrograde_burn(
        state,
        100.0,
    )

    energy_after = specific_orbital_energy(
        result.state_after,
        EARTH,
    )

    assert energy_after < energy_before