import numpy as np
import pytest

from src.integrators.rk4 import RK4
from src.orbital.conservation import (
    angular_momentum_relative_drift,
    specific_angular_momentum,
    specific_orbital_energy,
    energy_relative_drift,
)
from src.orbital.state import OrbitalState
from src.physics.bodies import EARTH
from src.physics.two_body import state_derivative
from src.simulation.simulator import Simulator


def circular_state():
    r = (
        EARTH.radius
        + 500_000.0
    )

    v = np.sqrt(
        EARTH.mu / r
    )

    return OrbitalState(
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


def test_circular_orbit_energy():
    state = circular_state()

    r = np.linalg.norm(
        state.position
    )

    energy = specific_orbital_energy(
        state,
        EARTH,
    )

    expected = (
        -EARTH.mu / (2.0 * r)
    )

    assert energy == pytest.approx(
        expected,
        rel=1e-12,
    )


def test_circular_angular_momentum():
    state = circular_state()

    h = specific_angular_momentum(
        state
    )

    expected = np.cross(
        state.position,
        state.velocity,
    )

    assert np.allclose(
        h,
        expected,
    )


def test_rk4_conserves_energy_and_momentum():
    state = circular_state()

    r = np.linalg.norm(
        state.position
    )

    period = (
        2.0
        * np.pi
        * np.sqrt(
            r**3 / EARTH.mu
        )
    )

    derivative = lambda state: state_derivative(
        state,
        EARTH,
    )

    simulator = Simulator(
        integrator=RK4(),
        derivative=derivative,
        dt=10.0,
    )

    result = simulator.run(
        initial_state=state,
        duration=period,
    )

    energy_error = energy_relative_drift(
        result,
        EARTH,
    )

    momentum_error = angular_momentum_relative_drift(
        result
    )

    assert np.max(
        np.abs(energy_error)
    ) < 1e-8

    assert np.max(
        np.abs(momentum_error)
    ) < 1e-8