"""Simulation result representation."""

from dataclasses import dataclass

import numpy as np

from src.orbital.state import OrbitalState


@dataclass
class SimulationResult:
    """Store the complete result of an orbital simulation.

    Parameters
    ----------
    times : numpy.ndarray
        Simulation times in seconds, shape (N,).
    positions : numpy.ndarray
        Position vectors in meters, shape (N, 3).
    velocities : numpy.ndarray
        Velocity vectors in meters per second, shape (N, 3).
    """

    times: np.ndarray
    positions: np.ndarray
    velocities: np.ndarray

    def __post_init__(self):
        self.times = np.asarray(self.times, dtype=float)
        self.positions = np.asarray(self.positions, dtype=float)
        self.velocities = np.asarray(self.velocities, dtype=float)

        if self.times.ndim != 1:
            raise ValueError(
                "Times must be a one-dimensional array."
            )

        if self.positions.ndim != 2 or self.positions.shape[1] != 3:
            raise ValueError(
                "Positions must have shape (N, 3)."
            )

        if self.velocities.ndim != 2 or self.velocities.shape[1] != 3:
            raise ValueError(
                "Velocities must have shape (N, 3)."
            )

        n = len(self.times)

        if len(self.positions) != n or len(self.velocities) != n:
            raise ValueError(
                "Times, positions and velocities must "
                "contain the same number of samples."
            )

    @property
    def initial_state(self) -> OrbitalState:
        """Return the initial orbital state."""
        return OrbitalState(
            position=self.positions[0].copy(),
            velocity=self.velocities[0].copy(),
        )

    @property
    def final_state(self) -> OrbitalState:
        """Return the final orbital state."""
        return OrbitalState(
            position=self.positions[-1].copy(),
            velocity=self.velocities[-1].copy(),
        )

    @property
    def duration(self) -> float:
        """Return the total simulation duration in seconds."""
        return self.times[-1] - self.times[0]