"""Fourth-order Runge-Kutta numerical integrator."""

import numpy as np

from src.integrators.base import Integrator, DerivativeFunction
from src.orbital.state import OrbitalState


class RK4(Integrator):
    """Classical fourth-order Runge-Kutta integrator."""

    def step(
        self,
        state: OrbitalState,
        dt: float,
        derivative: DerivativeFunction,
    ) -> OrbitalState:
        """Advance the state by one time step using RK4."""

        if dt <= 0:
            raise ValueError("Time step dt must be positive.")

        y = state.to_vector()

        # k1: derivative at the beginning of the step
        k1 = np.asarray(
            derivative(state),
            dtype=float,
        )

        # k2: derivative at the middle,
        # estimated using k1
        state_k2 = OrbitalState.from_vector(
            y + 0.5 * dt * k1
        )

        k2 = np.asarray(
            derivative(state_k2),
            dtype=float,
        )

        # k3: another estimation at the middle,
        # this time using k2
        state_k3 = OrbitalState.from_vector(
            y + 0.5 * dt * k2
        )

        k3 = np.asarray(
            derivative(state_k3),
            dtype=float,
        )

        # k4: derivative at the end of the step,
        # estimated using k3
        state_k4 = OrbitalState.from_vector(
            y + dt * k3
        )

        k4 = np.asarray(
            derivative(state_k4),
            dtype=float,
        )

        # Classical RK4 weighted average
        next_vector = y + (
            dt / 6.0
        ) * (
            k1
            + 2.0 * k2
            + 2.0 * k3
            + k4
        )

        return OrbitalState.from_vector(
            next_vector
        )