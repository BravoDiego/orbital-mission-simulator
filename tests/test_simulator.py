import numpy as np
import pytest

from src.integrators.euler import Euler
from src.orbital.state import OrbitalState
from src.simulation.simulator import Simulator


def zero_derivative(
    state: OrbitalState,
) -> np.ndarray:

    return np.zeros(6)


def test_simulator_number_of_samples():
    state = OrbitalState(
        position=np.zeros(3),
        velocity=np.zeros(3),
    )

    simulator = Simulator(
        integrator=Euler(),
        derivative=zero_derivative,
        dt=10.0,
    )

    result = simulator.run(
        initial_state=state,
        duration=100.0,
    )

    assert len(result.times) == 11

    assert result.times[-1] == pytest.approx(
        100.0
    )


def test_simulator_handles_partial_final_step():
    state = OrbitalState(
        position=np.zeros(3),
        velocity=np.zeros(3),
    )

    simulator = Simulator(
        integrator=Euler(),
        derivative=zero_derivative,
        dt=10.0,
    )

    result = simulator.run(
        initial_state=state,
        duration=25.0,
    )

    expected_times = np.array([
        0.0,
        10.0,
        20.0,
        25.0,
    ])

    assert np.allclose(
        result.times,
        expected_times,
    )