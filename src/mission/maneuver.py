import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class ManeuverResult:
    """
    Result of an impulsive maneuver.
    """

    state_before: np.ndarray
    state_after: np.ndarray

    delta_v_vector: np.ndarray
    delta_v_magnitude: float


def velocity_vector(state: np.ndarray) -> np.ndarray:
    """
    Extract velocity vector [vx, vy] from state.
    """

    state = np.asarray(state, dtype=float)

    if state.shape != (4,):
        raise ValueError(
            "State must have shape (4,) = [x, y, vx, vy]."
        )

    return state[2:4]


def velocity_magnitude(state: np.ndarray) -> float:
    """
    Return velocity magnitude.
    """

    return np.linalg.norm(
        velocity_vector(state)
    )


def prograde_direction(state: np.ndarray) -> np.ndarray:
    """
    Return unit vector pointing in the instantaneous
    direction of motion.
    """

    velocity = velocity_vector(state)

    speed = np.linalg.norm(velocity)

    if speed == 0:
        raise ValueError(
            "Cannot determine prograde direction "
            "when velocity is zero."
        )

    return velocity / speed


def retrograde_direction(state: np.ndarray) -> np.ndarray:
    """
    Return unit vector opposite to the direction of motion.
    """

    return -prograde_direction(state)


def apply_delta_v(
    state: np.ndarray,
    delta_v_vector: np.ndarray,
) -> ManeuverResult:
    """
    Apply an instantaneous velocity change.

    Position is unchanged.
    Velocity is modified by delta_v_vector.
    """

    state = np.asarray(state, dtype=float)
    delta_v_vector = np.asarray(
        delta_v_vector,
        dtype=float
    )

    if state.shape != (4,):
        raise ValueError(
            "State must have shape (4,) = [x, y, vx, vy]."
        )

    if delta_v_vector.shape != (2,):
        raise ValueError(
            "delta_v_vector must have shape (2,)."
        )

    state_after = state.copy()

    state_after[2:4] += delta_v_vector

    return ManeuverResult(
        state_before=state.copy(),
        state_after=state_after,
        delta_v_vector=delta_v_vector.copy(),
        delta_v_magnitude=np.linalg.norm(
            delta_v_vector
        ),
    )


def prograde_burn(
    state: np.ndarray,
    delta_v: float,
) -> ManeuverResult:
    """
    Apply a prograde impulsive burn.

    delta_v must be positive.
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
    state: np.ndarray,
    delta_v: float,
) -> ManeuverResult:
    """
    Apply a retrograde impulsive burn.

    delta_v must be positive.
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
    state: np.ndarray,
    delta_v: float,
) -> ManeuverResult:
    """
    Apply a signed tangential burn.

    Positive delta_v -> prograde.
    Negative delta_v -> retrograde.
    """

    direction = prograde_direction(state)

    delta_v_vector = delta_v * direction

    return apply_delta_v(
        state,
        delta_v_vector,
    )