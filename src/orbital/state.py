"""Orbital state representation."""

from dataclasses import dataclass

import numpy as np


@dataclass
class OrbitalState:
    """Represent the instantaneous state of an orbiting object.

    Parameters
    ----------
    position : numpy.ndarray
        Cartesian position vector [x, y, z] in meters.
    velocity : numpy.ndarray
        Cartesian velocity vector [vx, vy, vz] in meters per second.
    """

    position: np.ndarray
    velocity: np.ndarray

    def __post_init__(self):
        self.position = np.asarray(self.position, dtype=float)
        self.velocity = np.asarray(self.velocity, dtype=float)

        if self.position.shape != (3,):
            raise ValueError(
                "Position must be a three-dimensional vector."
            )

        if self.velocity.shape != (3,):
            raise ValueError(
                "Velocity must be a three-dimensional vector."
            )

    def to_vector(self) -> np.ndarray:
        """Return the state as [x, y, z, vx, vy, vz]."""
        return np.concatenate((self.position, self.velocity))

    @classmethod
    def from_vector(cls, vector: np.ndarray) -> "OrbitalState":
        """Create an OrbitalState from [x, y, z, vx, vy, vz]."""

        vector = np.asarray(vector, dtype=float)

        if vector.shape != (6,):
            raise ValueError(
                "State vector must contain exactly 6 components."
            )

        return cls(
            position=vector[:3],
            velocity=vector[3:],
        )