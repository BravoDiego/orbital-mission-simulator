import numpy as np
import pytest

from src.integrators.rk4 import RK4
from src.orbital.state import OrbitalState
from src.simulation.simulator import Simulator
from src.mission.mission import Mission
from src.physics.bodies import EARTH


def time_derivative(
    state: OrbitalState,
    time: float,
) -> np.ndarray:
    del state

    # dx/dt = t, everything else zero.
    return np.array([
        time,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ])


def zero_state():
    return OrbitalState(
        position=np.zeros(3),
        velocity=np.zeros(3),
    )


def test_rk4_uses_correct_stage_times():
    result = RK4().step(
        state=zero_state(),
        dt=2.0,
        derivative=time_derivative,
        time=3.0,
    )

    # Integral from 3 to 5 of t dt = (25 - 9)/2 = 8.
    assert result.position[0] == pytest.approx(
        8.0,
        abs=1e-12,
    )


def test_simulator_start_time_is_used_but_output_times_remain_relative():
    simulator = Simulator(
        integrator=RK4(),
        derivative=time_derivative,
        dt=0.5,
    )

    result = simulator.run(
        initial_state=zero_state(),
        duration=2.0,
        start_time=10.0,
    )

    # Integral from 10 to 12 of t dt = 22.
    assert result.final_state.position[0] == pytest.approx(
        22.0,
        abs=1e-12,
    )

    assert result.times[0] == pytest.approx(
        0.0
    )

    assert result.times[-1] == pytest.approx(
        2.0
    )


def test_mission_absolute_time_continues_across_coasts():
    mission = Mission(
        initial_state=zero_state(),
        body=EARTH,
        dt=0.25,
        derivative=time_derivative,
    )

    mission.coast(
        1.0
    )

    mission.coast(
        1.0
    )

    # Correct continuous-time result: integral from 0 to 2 of t dt = 2.
    # If time incorrectly restarted at each coast, this would be 1 instead.
    assert mission.current_state.position[0] == pytest.approx(
        2.0,
        abs=1e-12,
    )
