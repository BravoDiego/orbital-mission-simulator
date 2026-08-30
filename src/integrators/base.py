"""Base interface for numerical integrators."""

from abc import ABC, abstractmethod
from collections.abc import Callable

import numpy as np

from src.orbital.state import OrbitalState


DerivativeFunction = Callable[[OrbitalState], np.ndarray]


class Integrator(ABC):
    """Abstract base class for numerical integrators."""

    @abstractmethod
    def step(
        self,
        state: OrbitalState,
        dt: float,
        derivative: DerivativeFunction,
    ) -> OrbitalState:
        """Advance the state by one time step.

        Parameters
        ----------
        state : OrbitalState
            Current orbital state.
        dt : float
            Time step in seconds.
        derivative : callable
            Function computing the derivative of the state.

        Returns
        -------
        OrbitalState
            State after one time step.
        """
        raise NotImplementedError