"""J2 oblateness perturbation."""

from collections.abc import Callable

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody


def j2_acceleration(
    state: OrbitalState,
    body: CelestialBody,
    j2: float,
) -> np.ndarray:
    """
    Return the acceleration caused by the J2 oblateness term.

    Parameters
    ----------
    state
        Current spacecraft Cartesian state.

    body
        Central body.

    j2
        Dimensionless second zonal harmonic coefficient.

    Returns
    -------
    np.ndarray
        J2 acceleration vector in m/s².

    Notes
    -----
    The body's z axis is assumed to be its rotation axis.

    The body's radius is used as the reference equatorial radius.
    """

    if j2 < 0.0:
        raise ValueError(
            "J2 must be non-negative."
        )

    position = np.asarray(
        state.position,
        dtype=float,
    )

    if position.shape != (3,):
        raise ValueError(
            "Position must be a three-dimensional vector."
        )

    x, y, z = position

    r_squared = float(
        np.dot(
            position,
            position,
        )
    )

    if r_squared == 0.0:
        raise ValueError(
            "J2 acceleration is undefined at the body's center."
        )

    r = np.sqrt(
        r_squared
    )

    z_squared_over_r_squared = (
        z**2
        / r_squared
    )

    factor = (
        1.5
        * j2
        * body.mu
        * body.radius**2
        / r**5
    )

    common_xy = (
        5.0
        * z_squared_over_r_squared
        - 1.0
    )

    common_z = (
        5.0
        * z_squared_over_r_squared
        - 3.0
    )

    return factor * np.array([
        x * common_xy,
        y * common_xy,
        z * common_z,
    ])


def make_j2_perturbation(
    j2: float,
) -> Callable[
    [OrbitalState, CelestialBody],
    np.ndarray,
]:
    """
    Create a J2 perturbation function with a fixed J2 value.

    This allows it to be passed directly to the generic
    perturbation dynamics system.
    """

    if j2 < 0.0:
        raise ValueError(
            "J2 must be non-negative."
        )

    def perturbation(
        state: OrbitalState,
        body: CelestialBody,
    ) -> np.ndarray:

        return j2_acceleration(
            state=state,
            body=body,
            j2=j2,
        )

    return perturbation