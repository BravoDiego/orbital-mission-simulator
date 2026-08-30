"""Explicit Euler numerical integrator."""

import numpy as np

from src.integrators.base import Integrator, DerivativeFunction
from src.orbital.state import OrbitalState


class Euler(Integrator):
    """Explicit first-order Euler integrator."""

    def step(
        self,
        state: OrbitalState,
        dt: float,
        derivative: DerivativeFunction,
    ) -> OrbitalState:
        """Advance the state by one time step using Euler's method."""

        if dt <= 0:
            raise ValueError("Time step dt must be positive.")

        # Current state:
        # [x, y, z, vx, vy, vz]
        state_vector = state.to_vector()

        # Current derivative:
        # [vx, vy, vz, ax, ay, az]
        derivative_vector = np.asarray(
            derivative(state),
            dtype=float,
        )

        if derivative_vector.shape != (6,):
            raise ValueError(
                "Derivative must contain exactly 6 components."
            )

        # Euler:
        # y_(n+1) = y_n + dt * f(y_n)
        next_vector = (
            state_vector
            + dt * derivative_vector
        )

        return OrbitalState.from_vector(next_vector)