"""Explicit Euler numerical integrator."""

from src.integrators.base import (
    DerivativeFunction,
    Integrator,
    evaluate_derivative,
)
from src.orbital.state import OrbitalState


class Euler(Integrator):
    """Explicit first-order Euler integrator."""

    def step(
        self,
        state: OrbitalState,
        dt: float,
        derivative: DerivativeFunction,
        time: float = 0.0,
    ) -> OrbitalState:
        """Advance the state by one time step using Euler's method."""

        if dt <= 0:
            raise ValueError(
                "Time step dt must be positive."
            )

        state_vector = state.to_vector()

        derivative_vector = evaluate_derivative(
            derivative=derivative,
            state=state,
            time=time,
        )

        next_vector = (
            state_vector
            + dt * derivative_vector
        )

        return OrbitalState.from_vector(
            next_vector
        )
