"""Differential third-body gravitational perturbations."""

from collections.abc import Callable

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody


PositionFunction = Callable[[float], np.ndarray]


def third_body_acceleration(
    state: OrbitalState,
    third_body_position: np.ndarray,
    third_body_mu: float,
) -> np.ndarray:
    """Return differential acceleration due to a third body.

    All positions are expressed in a central-body-centered inertial frame.
    The acceleration is

        mu_3 * [
            (r_3 - r) / |r_3 - r|^3
            - r_3 / |r_3|^3
        ]

    The second term subtracts the acceleration of the central-body origin
    itself due to the third body. This is essential in a geocentric frame.
    """

    if third_body_mu <= 0.0:
        raise ValueError(
            "Third-body gravitational parameter must be positive."
        )

    spacecraft_position = np.asarray(
        state.position,
        dtype=float,
    )

    third_body_position = np.asarray(
        third_body_position,
        dtype=float,
    )

    if third_body_position.shape != (3,):
        raise ValueError(
            "Third-body position must have shape (3,)."
        )

    third_body_distance = float(
        np.linalg.norm(
            third_body_position
        )
    )

    if third_body_distance == 0.0:
        raise ValueError(
            "Third-body position cannot coincide with the origin."
        )

    relative_position = (
        third_body_position
        - spacecraft_position
    )

    relative_distance = float(
        np.linalg.norm(
            relative_position
        )
    )

    if relative_distance == 0.0:
        raise ValueError(
            "Spacecraft position cannot coincide with the third body."
        )

    spacecraft_term = (
        relative_position
        / relative_distance**3
    )

    origin_term = (
        third_body_position
        / third_body_distance**3
    )

    return (
        third_body_mu
        * (
            spacecraft_term
            - origin_term
        )
    )


def make_third_body_perturbation(
    third_body: CelestialBody,
    position_function: PositionFunction,
) -> Callable[..., np.ndarray]:
    """Create a time-dependent perturbation for a moving third body."""

    def perturbation(
        state: OrbitalState,
        body: CelestialBody,
        time: float = 0.0,
    ) -> np.ndarray:
        del body  # Interface compatibility: the central body is not needed.

        position = np.asarray(
            position_function(
                time
            ),
            dtype=float,
        )

        return third_body_acceleration(
            state=state,
            third_body_position=position,
            third_body_mu=third_body.mu,
        )

    return perturbation
