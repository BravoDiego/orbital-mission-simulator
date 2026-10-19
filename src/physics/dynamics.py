"""Composable orbital dynamics models."""

from collections.abc import Callable, Iterable
from functools import lru_cache
from inspect import Parameter, signature

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody
from src.physics.two_body import state_derivative


PerturbationAcceleration = Callable[..., np.ndarray]
DerivativeFunction = Callable[..., np.ndarray]


@lru_cache(maxsize=None)
def _cached_accepts_time_argument(
    perturbation: Callable,
) -> bool:
    parameters = tuple(
        signature(perturbation).parameters.values()
    )

    if any(
        parameter.kind == Parameter.VAR_POSITIONAL
        for parameter in parameters
    ):
        return True

    positional = [
        parameter
        for parameter in parameters
        if parameter.kind in (
            Parameter.POSITIONAL_ONLY,
            Parameter.POSITIONAL_OR_KEYWORD,
        )
    ]

    return len(positional) >= 3


def _perturbation_accepts_time(
    perturbation: Callable,
) -> bool:
    try:
        return _cached_accepts_time_argument(
            perturbation
        )
    except (
        TypeError,
        ValueError,
    ):
        return False


def _evaluate_perturbation(
    perturbation: PerturbationAcceleration,
    state: OrbitalState,
    body: CelestialBody,
    time: float,
) -> np.ndarray:
    if _perturbation_accepts_time(
        perturbation
    ):
        acceleration = perturbation(
            state,
            body,
            time,
        )
    else:
        acceleration = perturbation(
            state,
            body,
        )

    acceleration = np.asarray(
        acceleration,
        dtype=float,
    )

    if acceleration.shape != (3,):
        raise ValueError(
            "A perturbation acceleration must return "
            "a vector of shape (3,)."
        )

    if not np.all(
        np.isfinite(
            acceleration
        )
    ):
        raise ValueError(
            "Perturbation acceleration contains "
            "non-finite values."
        )

    return acceleration


def build_state_derivative(
    body: CelestialBody,
    perturbations: Iterable[
        PerturbationAcceleration
    ] = (),
) -> DerivativeFunction:
    """Build a state derivative with optional perturbations.

    The project convention is a six-component NumPy vector::

        [vx, vy, vz, ax, ay, az]

    The returned derivative accepts an optional absolute time argument::

        derivative(state, time=0.0)

    Time-independent perturbations from earlier versions remain supported.
    """

    perturbations = tuple(
        perturbations
    )

    def derivative(
        state: OrbitalState,
        time: float = 0.0,
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
            additional_acceleration += (
                _evaluate_perturbation(
                    perturbation=perturbation,
                    state=state,
                    body=body,
                    time=time,
                )
            )

        result = base_derivative.copy()
        result[3:] += additional_acceleration

        return result

    return derivative
