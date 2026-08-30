"""Numerical simulation engine."""

from collections.abc import Callable

import numpy as np

from src.integrators.base import Integrator
from src.orbital.state import OrbitalState
from src.simulation.result import SimulationResult


DerivativeFunction = Callable[[OrbitalState], np.ndarray]


class Simulator:
    """Propagate an orbital state through time.

    Parameters
    ----------
    integrator : Integrator
        Numerical integration method.
    derivative : callable
        Function computing the derivative of an orbital state.
    dt : float
        Default simulation time step in seconds.
    """

    def __init__(
        self,
        integrator: Integrator,
        derivative: DerivativeFunction,
        dt: float,
    ):
        if dt <= 0:
            raise ValueError(
                "Time step dt must be positive."
            )

        self.integrator = integrator
        self.derivative = derivative
        self.dt = float(dt)

    def run(
        self,
        initial_state: OrbitalState,
        duration: float,
    ) -> SimulationResult:
        """Run the simulation.

        Parameters
        ----------
        initial_state : OrbitalState
            State at t = 0.
        duration : float
            Total simulation duration in seconds.

        Returns
        -------
        SimulationResult
            Complete time history of the simulation.
        """

        if duration <= 0:
            raise ValueError(
                "Simulation duration must be positive."
            )

        current_state = OrbitalState(
            position=initial_state.position.copy(),
            velocity=initial_state.velocity.copy(),
        )

        current_time = 0.0

        times = [current_time]
        positions = [current_state.position.copy()]
        velocities = [current_state.velocity.copy()]

        while current_time < duration:
            step_dt = min(
                self.dt,
                duration - current_time,
            )

            current_state = self.integrator.step(
                state=current_state,
                dt=step_dt,
                derivative=self.derivative,
            )

            current_time += step_dt

            times.append(current_time)
            positions.append(
                current_state.position.copy()
            )
            velocities.append(
                current_state.velocity.copy()
            )

        return SimulationResult(
            times=np.array(times),
            positions=np.array(positions),
            velocities=np.array(velocities),
        )