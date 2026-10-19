"""Numerical simulation engine."""

from collections.abc import Callable

import numpy as np

from src.integrators.base import Integrator
from src.orbital.state import OrbitalState
from src.simulation.result import SimulationResult


DerivativeFunction = Callable[..., np.ndarray]


class Simulator:
    """Propagate an orbital state through time."""

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
        start_time: float = 0.0,
    ) -> SimulationResult:
        """Run the simulation.

        Parameters
        ----------
        initial_state
            State at the beginning of this propagation segment.
        duration
            Segment duration in seconds.
        start_time
            Absolute model time corresponding to the segment start.
            Returned ``SimulationResult.times`` remain relative to the
            segment and therefore still begin at zero.
        """

        if duration <= 0:
            raise ValueError(
                "Simulation duration must be positive."
            )

        if not np.isfinite(
            start_time
        ):
            raise ValueError(
                "Simulation start time must be finite."
            )

        current_state = OrbitalState(
            position=initial_state.position.copy(),
            velocity=initial_state.velocity.copy(),
        )

        elapsed_time = 0.0

        times = [elapsed_time]
        positions = [
            current_state.position.copy()
        ]
        velocities = [
            current_state.velocity.copy()
        ]

        while elapsed_time < duration:
            step_dt = min(
                self.dt,
                duration - elapsed_time,
            )

            absolute_time = (
                start_time
                + elapsed_time
            )

            current_state = self.integrator.step(
                state=current_state,
                dt=step_dt,
                derivative=self.derivative,
                time=absolute_time,
            )

            elapsed_time += step_dt

            times.append(
                elapsed_time
            )
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
