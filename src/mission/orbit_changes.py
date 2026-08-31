from dataclasses import dataclass

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody


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
    Return the unit tangential direction of motion.

    The radial component of velocity is removed first.
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
        position / radius
    )

    radial_velocity_vector = (
        np.dot(
            velocity,
            radial_direction,
        )
        * radial_direction
    )

    tangential_velocity = (
        velocity
        - radial_velocity_vector
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


def radial_velocity(
    state: OrbitalState,
) -> float:
    """
    Return radial velocity.

    Positive -> moving outward.
    Negative -> moving inward.
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
            "Radial velocity is undefined at the center."
        )

    return float(
        np.dot(
            position,
            velocity,
        )
        / radius
    )


def apsis_radii(
    state: OrbitalState,
    body: CelestialBody,
) -> tuple[float, float]:
    """
    Return periapsis and apoapsis radii of a bound orbit.

    Returns
    -------
    tuple
        (periapsis_radius, apoapsis_radius)
    """

    position = np.asarray(
        state.position,
        dtype=float,
    )

    velocity = np.asarray(
        state.velocity,
        dtype=float,
    )

    r = np.linalg.norm(
        position
    )

    v = np.linalg.norm(
        velocity
    )

    if r == 0.0:
        raise ValueError(
            "Orbital elements are undefined at the center."
        )

    energy = (
        0.5 * v**2
        - body.mu / r
    )

    if energy >= 0.0:
        raise ValueError(
            "Apoapsis is not finite for an unbound orbit."
        )

    semi_major_axis = (
        -body.mu
        / (2.0 * energy)
    )

    angular_momentum = np.cross(
        position,
        velocity,
    )

    eccentricity_vector = (
        np.cross(
            velocity,
            angular_momentum,
        )
        / body.mu
        - position / r
    )

    eccentricity = np.linalg.norm(
        eccentricity_vector
    )

    periapsis = (
        semi_major_axis
        * (1.0 - eccentricity)
    )

    apoapsis = (
        semi_major_axis
        * (1.0 + eccentricity)
    )

    return (
        float(periapsis),
        float(apoapsis),
    )


def plan_circularization(
    state: OrbitalState,
    body: CelestialBody,
) -> OrbitChangePlan:
    """
    Compute the instantaneous delta-v needed to circularize
    at the spacecraft's current position.

    Unlike a purely tangential burn, this also removes any
    radial velocity component.
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

    current_speed = np.linalg.norm(
        velocity
    )

    target_speed = np.sqrt(
        body.mu / radius
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
        current_radius=float(radius),
        target_radius=float(radius),
        current_speed=float(current_speed),
        target_speed=float(target_speed),
        delta_v_vector=delta_v_vector,
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
    Plan a burn at periapsis that sets the opposite
    apoapsis to target_radius.

    The spacecraft must currently be approximately at an apsis.
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

    vr = radial_velocity(
        state
    )

    if abs(vr) > apsis_tolerance:
        raise ValueError(
            "set_apoapsis must be performed at an apsis."
        )

    semi_major_axis = (
        current_radius
        + target_radius
    ) / 2.0

    target_speed = np.sqrt(
        body.mu
        * (
            2.0 / current_radius
            - 1.0 / semi_major_axis
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
        delta_v_vector=delta_v_vector,
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
    Plan a burn at apoapsis that sets the opposite
    periapsis to target_radius.
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

    vr = radial_velocity(
        state
    )

    if abs(vr) > apsis_tolerance:
        raise ValueError(
            "set_periapsis must be performed at an apsis."
        )

    semi_major_axis = (
        current_radius
        + target_radius
    ) / 2.0

    target_speed = np.sqrt(
        body.mu
        * (
            2.0 / current_radius
            - 1.0 / semi_major_axis
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
        delta_v_vector=delta_v_vector,
        delta_v_magnitude=float(
            np.linalg.norm(
                delta_v_vector
            )
        ),
    )