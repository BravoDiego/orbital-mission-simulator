"""Base interface and helpers for numerical integrators."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from functools import lru_cache
from inspect import Parameter, signature

import numpy as np

from src.orbital.state import OrbitalState


DerivativeFunction = Callable[..., np.ndarray]


@lru_cache(maxsize=None)
def _cached_accepts_time_argument(
    derivative: Callable,
) -> bool:
    """Return whether a callable accepts a second positional argument."""

    parameters = tuple(
        signature(derivative).parameters.values()
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

    return len(positional) >= 2


def derivative_accepts_time(
    derivative: Callable,
) -> bool:
    """Return whether ``derivative`` supports ``derivative(state, time)``.

    One-argument derivatives from V1/V2 remain fully supported.
    """

    try:
        return _cached_accepts_time_argument(
            derivative
        )
    except (
        TypeError,
        ValueError,
    ):
        # Conservative fallback for unusual callables whose signature
        # cannot be inspected: keep the historical one-argument API.
        return False


def evaluate_derivative(
    derivative: DerivativeFunction,
    state: OrbitalState,
    time: float,
) -> np.ndarray:
    """Evaluate either a time-independent or time-dependent derivative."""

    if derivative_accepts_time(
        derivative
    ):
        value = derivative(
            state,
            time,
        )
    else:
        value = derivative(
            state
        )

    value = np.asarray(
        value,
        dtype=float,
    )

    if value.shape != (6,):
        raise ValueError(
            "Derivative must contain exactly 6 components."
        )

    return value


class Integrator(ABC):
    """Abstract base class for numerical integrators."""

    @abstractmethod
    def step(
        self,
        state: OrbitalState,
        dt: float,
        derivative: DerivativeFunction,
        time: float = 0.0,
    ) -> OrbitalState:
        """Advance the state by one time step.

        ``time`` is the absolute simulation time at the beginning of the
        step. Time-independent derivatives may ignore it.
        """
        raise NotImplementedError
