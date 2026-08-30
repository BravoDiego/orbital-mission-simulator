import numpy as np
import pytest

from src.integrators.euler import Euler
from src.integrators.rk4 import RK4
from src.orbital.state import OrbitalState


def exponential_derivative(
    state: OrbitalState,
) -> np.ndarray:
    """Simple test system dx/dt = x."""

    return np.array([
        state.position[0],
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])


def test_euler_step():
    state = OrbitalState(
        position=np.array([
            1.0,
            0.0,
            0.0,
        ]),
        velocity=np.zeros(3),
    )

    result = Euler().step(
        state,
        dt=0.1,
        derivative=exponential_derivative,
    )

    # Euler:
    # x1 = 1 + 0.1 * 1 = 1.1

    assert result.position[0] == pytest.approx(
        1.1
    )


def test_rk4_step():
    state = OrbitalState(
        position=np.array([
            1.0,
            0.0,
            0.0,
        ]),
        velocity=np.zeros(3),
    )

    result = RK4().step(
        state,
        dt=0.1,
        derivative=exponential_derivative,
    )

    # Exact solution:
    # x(0.1) = exp(0.1)

    expected = np.exp(0.1)

    assert result.position[0] == pytest.approx(
        expected,
        rel=1e-6,
    )