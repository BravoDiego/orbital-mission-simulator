"""Conservation quantities for orbital dynamics."""

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody
from src.simulation.result import SimulationResult


def specific_orbital_energy(
    state: OrbitalState,
    body: CelestialBody,
) -> float:
    """
    Return the specific orbital energy in J/kg.
    """

    r = np.linalg.norm(
        state.position
    )

    if r == 0.0:
        raise ValueError(
            "Orbital energy is undefined at r = 0."
        )

    v_squared = np.dot(
        state.velocity,
        state.velocity,
    )

    return float(
        0.5 * v_squared
        - body.mu / r
    )


def specific_angular_momentum(
    state: OrbitalState,
) -> np.ndarray:
    """
    Return the specific angular momentum vector.
    """

    return np.cross(
        state.position,
        state.velocity,
    )


def energy_history(
    result: SimulationResult,
    body: CelestialBody,
) -> np.ndarray:
    """
    Compute specific orbital energy at every simulation step.
    """

    radii = np.linalg.norm(
        result.positions,
        axis=1,
    )

    if np.any(
        radii == 0.0
    ):
        raise ValueError(
            "Orbital energy is undefined at r = 0."
        )

    velocity_squared = np.sum(
        result.velocities**2,
        axis=1,
    )

    return (
        0.5 * velocity_squared
        - body.mu / radii
    )


def angular_momentum_history(
    result: SimulationResult,
) -> np.ndarray:
    """
    Compute specific angular momentum at every simulation step.
    """

    return np.cross(
        result.positions,
        result.velocities,
    )


def relative_drift(
    values: np.ndarray,
) -> np.ndarray:
    """
    Return relative drift from the initial value.

    (value - value_0) / |value_0|
    """

    values = np.asarray(
        values,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "Values must be a one-dimensional array."
        )

    if len(values) == 0:
        raise ValueError(
            "Values cannot be empty."
        )

    initial_value = (
        values[0]
    )

    if initial_value == 0.0:
        raise ValueError(
            "Relative drift cannot use a zero reference."
        )

    return (
        values
        - initial_value
    ) / abs(
        initial_value
    )


def energy_relative_drift(
    result: SimulationResult,
    body: CelestialBody,
) -> np.ndarray:
    """
    Return relative specific-energy drift.
    """

    energies = energy_history(
        result,
        body,
    )

    return relative_drift(
        energies
    )


def angular_momentum_relative_drift(
    result: SimulationResult,
) -> np.ndarray:
    """
    Return relative angular-momentum magnitude drift.
    """

    angular_momenta = (
        angular_momentum_history(
            result
        )
    )

    magnitudes = np.linalg.norm(
        angular_momenta,
        axis=1,
    )

    return relative_drift(
        magnitudes
    )