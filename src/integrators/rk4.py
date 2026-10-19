"""Fourth-order Runge-Kutta numerical integrator."""

from src.integrators.base import (
    DerivativeFunction,
    Integrator,
    evaluate_derivative,
)
from src.orbital.state import OrbitalState


class RK4(Integrator):
    """Classical fourth-order Runge-Kutta integrator."""

    def step(
        self,
        state: OrbitalState,
        dt: float,
        derivative: DerivativeFunction,
        time: float = 0.0,
    ) -> OrbitalState:
        """Advance the state by one time step using RK4.

        Time-dependent dynamics are evaluated at the proper RK4 stage
        times: t, t + dt/2, t + dt/2 and t + dt.
        """

        if dt <= 0:
            raise ValueError(
                "Time step dt must be positive."
            )

        y = state.to_vector()

        k1 = evaluate_derivative(
            derivative,
            state,
            time,
        )

        state_k2 = OrbitalState.from_vector(
            y + 0.5 * dt * k1
        )

        k2 = evaluate_derivative(
            derivative,
            state_k2,
            time + 0.5 * dt,
        )

        state_k3 = OrbitalState.from_vector(
            y + 0.5 * dt * k2
        )

        k3 = evaluate_derivative(
            derivative,
            state_k3,
            time + 0.5 * dt,
        )

        state_k4 = OrbitalState.from_vector(
            y + dt * k3
        )

        k4 = evaluate_derivative(
            derivative,
            state_k4,
            time + dt,
        )

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
