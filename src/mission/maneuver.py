import numpy as np
from dataclasses import dataclass

from src.orbital.state import OrbitalState


@dataclass(frozen=True)
class ManeuverResult:
    """
    Result of an instantaneous impulsive maneuver.
    """

    state_before: OrbitalState
    state_after: OrbitalState

    delta_v_vector: np.ndarray
    delta_v_magnitude: float


def velocity_magnitude(state: OrbitalState) -> float:
    """
    Return the spacecraft speed.
    """

    return float(np.linalg.norm(state.velocity))


def prograde_direction(state: OrbitalState) -> np.ndarray:
    """
    Return the unit vector tangent to the trajectory
    in the direction of motion.
    """

    speed = velocity_magnitude(state)

    if speed == 0.0:
        raise ValueError(
            "Cannot determine prograde direction "
            "when velocity is zero."
        )

    return state.velocity / speed


def retrograde_direction(state: OrbitalState) -> np.ndarray:
    """
    Return the unit vector opposite to the direction of motion.
    """

    return -prograde_direction(state)


def apply_delta_v(
    state: OrbitalState,
    delta_v_vector: np.ndarray,
) -> ManeuverResult:
    """
    Apply an instantaneous velocity change.

    Position remains unchanged.
    """

    delta_v_vector = np.asarray(
        delta_v_vector,
        dtype=float,
    )

    if delta_v_vector.shape != (3,):
        raise ValueError(
            "delta_v_vector must have shape (3,)."
        )

    new_velocity = (
        state.velocity
        + delta_v_vector
    )

    state_after = OrbitalState(
        position=state.position.copy(),
        velocity=new_velocity,
    )

    return ManeuverResult(
        state_before=state,
        state_after=state_after,
        delta_v_vector=delta_v_vector.copy(),
        delta_v_magnitude=float(
            np.linalg.norm(delta_v_vector)
        ),
    )


def prograde_burn(
    state: OrbitalState,
    delta_v: float,
) -> ManeuverResult:
    """
    Apply a positive prograde burn.
    """

    if delta_v < 0:
        raise ValueError(
            "delta_v must be positive for prograde_burn."
        )

    direction = prograde_direction(state)

    return apply_delta_v(
        state,
        delta_v * direction,
    )


def retrograde_burn(
    state: OrbitalState,
    delta_v: float,
) -> ManeuverResult:
    """
    Apply a positive retrograde burn.
    """

    if delta_v < 0:
        raise ValueError(
            "delta_v must be positive for retrograde_burn."
        )

    direction = retrograde_direction(state)

    return apply_delta_v(
        state,
        delta_v * direction,
    )


def tangential_burn(
    state: OrbitalState,
    delta_v: float,
) -> ManeuverResult:
    """
    Apply a signed tangential burn.

    delta_v > 0 : prograde
    delta_v < 0 : retrograde
    """

    direction = prograde_direction(state)

    return apply_delta_v(
        state,
        delta_v * direction,
    )