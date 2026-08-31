"""High-level impulsive orbit-change planning."""

from dataclasses import dataclass

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody

from src.orbital.elements import (
    radial_velocity,
    apsis_radii,
)


@dataclass(frozen=True)
class OrbitChangePlan:
    """
    Description of a planned instantaneous orbit change.
    """

    name: str

    current_radius: float
    target_radius: float

    current_speed: float
    target_speed: float

    delta_v_vector: np.ndarray
    delta_v_magnitude: float


def _tangential_direction(
    state: OrbitalState,
) -> np.ndarray:
    """
    Return the instantaneous tangential direction of motion.

    The radial component of velocity is removed.
    """

    position = np.asarray(
        state.position,
        dtype=float,
    )

    velocity = np.asarray(
        state.velocity,
        dtype=float,
    )

    radius = np.linalg.norm(
        position
    )

    if radius == 0.0:
        raise ValueError(
            "Tangential direction is undefined at the center."
        )

    radial_direction = (
        position
        / radius
    )

    radial_component = (
        np.dot(
            velocity,
            radial_direction,
        )
        * radial_direction
    )

    tangential_velocity = (
        velocity
        - radial_component
    )

    tangential_speed = np.linalg.norm(
        tangential_velocity
    )

    if tangential_speed == 0.0:
        raise ValueError(
            "Tangential direction cannot be determined "
            "from a purely radial trajectory."
        )

    return (
        tangential_velocity
        / tangential_speed
    )


def plan_circularization(
    state: OrbitalState,
    body: CelestialBody,
) -> OrbitChangePlan:
    """
    Compute the impulsive delta-v required to circularize
    at the spacecraft's current position.

    Any radial velocity component is removed.
    """

    position = np.asarray(
        state.position,
        dtype=float,
    )

    velocity = np.asarray(
        state.velocity,
        dtype=float,
    )

    radius = np.linalg.norm(
        position
    )

    if radius == 0.0:
        raise ValueError(
            "Circularization is undefined at the center."
        )

    current_speed = np.linalg.norm(
        velocity
    )

    target_speed = np.sqrt(
        body.mu
        / radius
    )

    direction = _tangential_direction(
        state
    )

    target_velocity = (
        target_speed
        * direction
    )

    delta_v_vector = (
        target_velocity
        - velocity
    )

    return OrbitChangePlan(
        name="Circularization",
        current_radius=float(
            radius
        ),
        target_radius=float(
            radius
        ),
        current_speed=float(
            current_speed
        ),
        target_speed=float(
            target_speed
        ),
        delta_v_vector=(
            delta_v_vector
        ),
        delta_v_magnitude=float(
            np.linalg.norm(
                delta_v_vector
            )
        ),
    )


def plan_set_apoapsis(
    state: OrbitalState,
    target_radius: float,
    body: CelestialBody,
    apsis_tolerance: float = 1.0,
) -> OrbitChangePlan:
    """
    Plan a burn at an apsis that makes the current point
    the periapsis and sets the opposite apoapsis.
    """

    current_radius = np.linalg.norm(
        state.position
    )

    if target_radius <= current_radius:
        raise ValueError(
            "Target apoapsis must be greater than "
            "the current radius."
        )

    if target_radius <= body.radius:
        raise ValueError(
            "Target apoapsis must be above the body's surface."
        )

    if abs(
        radial_velocity(
            state
        )
    ) > apsis_tolerance:
        raise ValueError(
            "set_apoapsis must be performed at an apsis."
        )

    semi_major_axis_target = (
        current_radius
        + target_radius
    ) / 2.0

    target_speed = np.sqrt(
        body.mu
        * (
            2.0 / current_radius
            - 1.0
            / semi_major_axis_target
        )
    )

    current_speed = np.linalg.norm(
        state.velocity
    )

    direction = _tangential_direction(
        state
    )

    target_velocity = (
        target_speed
        * direction
    )

    delta_v_vector = (
        target_velocity
        - state.velocity
    )

    return OrbitChangePlan(
        name="Set apoapsis",
        current_radius=float(
            current_radius
        ),
        target_radius=float(
            target_radius
        ),
        current_speed=float(
            current_speed
        ),
        target_speed=float(
            target_speed
        ),
        delta_v_vector=(
            delta_v_vector
        ),
        delta_v_magnitude=float(
            np.linalg.norm(
                delta_v_vector
            )
        ),
    )


def plan_set_periapsis(
    state: OrbitalState,
    target_radius: float,
    body: CelestialBody,
    apsis_tolerance: float = 1.0,
) -> OrbitChangePlan:
    """
    Plan a burn at an apsis that makes the current point
    the apoapsis and sets the opposite periapsis.
    """

    current_radius = np.linalg.norm(
        state.position
    )

    if target_radius >= current_radius:
        raise ValueError(
            "Target periapsis must be smaller than "
            "the current radius."
        )

    if target_radius <= body.radius:
        raise ValueError(
            "Target periapsis must remain above "
            "the body's surface."
        )

    if abs(
        radial_velocity(
            state
        )
    ) > apsis_tolerance:
        raise ValueError(
            "set_periapsis must be performed at an apsis."
        )

    semi_major_axis_target = (
        current_radius
        + target_radius
    ) / 2.0

    target_speed = np.sqrt(
        body.mu
        * (
            2.0 / current_radius
            - 1.0
            / semi_major_axis_target
        )
    )

    current_speed = np.linalg.norm(
        state.velocity
    )

    direction = _tangential_direction(
        state
    )

    target_velocity = (
        target_speed
        * direction
    )

    delta_v_vector = (
        target_velocity
        - state.velocity
    )

    return OrbitChangePlan(
        name="Set periapsis",
        current_radius=float(
            current_radius
        ),
        target_radius=float(
            target_radius
        ),
        current_speed=float(
            current_speed
        ),
        target_speed=float(
            target_speed
        ),
        delta_v_vector=(
            delta_v_vector
        ),
        delta_v_magnitude=float(
            np.linalg.norm(
                delta_v_vector
            )
        ),
    )