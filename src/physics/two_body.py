"""Two-body gravitational dynamics."""

import numpy as np

from src.physics.bodies import CelestialBody
from src.orbital.state import OrbitalState


def gravitational_acceleration(
    position: np.ndarray,
    body: CelestialBody,
) -> np.ndarray:
    """Compute gravitational acceleration in the two-body model.

    Parameters
    ----------
    position : numpy.ndarray
        Position relative to the central body in meters.
    body : CelestialBody
        Central celestial body.

    Returns
    -------
    numpy.ndarray
        Acceleration vector in m/s^2.
    """

    position = np.asarray(position, dtype=float)

    if position.shape != (3,):
        raise ValueError(
            "Position must be a three-dimensional vector."
        )

    r = np.linalg.norm(position)

    if r == 0:
        raise ValueError(
            "Position cannot be located at the center of the body."
        )

    return -body.mu * position / r**3

def state_derivative(
    state: OrbitalState,
    body: CelestialBody,
) -> np.ndarray:
    """Compute the time derivative of an orbital state.

    Returns
    -------
    numpy.ndarray
        [vx, vy, vz, ax, ay, az]
    """

    acceleration = gravitational_acceleration(
        state.position,
        body,
    )

    return np.concatenate(
        (
            state.velocity,
            acceleration,
        )
    )