"""Composable orbital dynamics models."""

from collections.abc import Callable, Iterable

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody
from src.physics.two_body import state_derivative


PerturbationAcceleration = Callable[
    [OrbitalState, CelestialBody],
    np.ndarray,
]

DerivativeFunction = Callable[
    [OrbitalState],
    np.ndarray,
]


def build_state_derivative(
    body: CelestialBody,
    perturbations: Iterable[
        PerturbationAcceleration
    ] = (),
) -> DerivativeFunction:
    """Build a state derivative with optional perturbations.

    The project convention is a six-component NumPy vector::

        [vx, vy, vz, ax, ay, az]

    Central two-body gravity is always included. Each perturbation
    contributes an additional acceleration vector of shape ``(3,)``.
    """

    perturbations = tuple(perturbations)

    def derivative(
        state: OrbitalState,
    ) -> np.ndarray:
        base_derivative = np.asarray(
            state_derivative(
                state,
                body,
            ),
            dtype=float,
        )

        if base_derivative.shape != (6,):
            raise ValueError(
                "The base state derivative must have shape (6,)."
            )

        additional_acceleration = np.zeros(
            3,
            dtype=float,
        )

        for perturbation in perturbations:
            acceleration = np.asarray(
                perturbation(
                    state,
                    body,
                ),
                dtype=float,
            )

            if acceleration.shape != (3,):
                raise ValueError(
                    "A perturbation acceleration must return "
                    "a vector of shape (3,)."
                )

            if not np.all(np.isfinite(acceleration)):
                raise ValueError(
                    "Perturbation acceleration contains "
                    "non-finite values."
                )

            additional_acceleration += acceleration

        result = base_derivative.copy()
        result[3:] += additional_acceleration

        return result

    return derivative
