"""Orbital elements and geometric quantities."""

from dataclasses import dataclass
from math import pi, sqrt

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody

from src.orbital.conservation import (
    specific_orbital_energy,
    specific_angular_momentum,
)


@dataclass(frozen=True)
class OrbitalElements:
    """
    Main orbital elements derived from a Cartesian state.

    This project currently focuses on quantities useful for
    two-body mission analysis rather than the full classical
    six-element representation.
    """

    semi_major_axis: float
    eccentricity: float
    inclination: float

    periapsis_radius: float | None
    apoapsis_radius: float | None

    eccentricity_vector: np.ndarray
    angular_momentum_vector: np.ndarray

    # --------------------------------------------------
    # Convenience aliases
    # --------------------------------------------------

    @property
    def a(self) -> float:
        return self.semi_major_axis

    @property
    def e(self) -> float:
        return self.eccentricity

    @property
    def i(self) -> float:
        return self.inclination

    @property
    def rp(self) -> float | None:
        return self.periapsis_radius

    @property
    def ra(self) -> float | None:
        return self.apoapsis_radius


def orbital_radius(
    state: OrbitalState,
) -> float:
    """
    Return the distance from the central body.
    """

    radius = np.linalg.norm(
        state.position
    )

    if radius == 0.0:
        raise ValueError(
            "Orbital radius is undefined at the body's center."
        )

    return float(
        radius
    )


def orbital_speed(
    state: OrbitalState,
) -> float:
    """
    Return spacecraft speed.
    """

    return float(
        np.linalg.norm(
            state.velocity
        )
    )


def radial_velocity(
    state: OrbitalState,
) -> float:
    """
    Return radial velocity.

    Positive
        Spacecraft moving away from the central body.

    Negative
        Spacecraft moving toward the central body.
    """

    radius = orbital_radius(
        state
    )

    return float(
        np.dot(
            state.position,
            state.velocity,
        )
        / radius
    )


def eccentricity_vector(
    state: OrbitalState,
    body: CelestialBody,
) -> np.ndarray:
    """
    Return the orbital eccentricity vector.
    """

    radius = orbital_radius(
        state
    )

    angular_momentum = (
        specific_angular_momentum(
            state
        )
    )

    return (
        np.cross(
            state.velocity,
            angular_momentum,
        )
        / body.mu
        - state.position / radius
    )


def orbital_eccentricity(
    state: OrbitalState,
    body: CelestialBody,
) -> float:
    """
    Return orbital eccentricity.
    """

    return float(
        np.linalg.norm(
            eccentricity_vector(
                state,
                body,
            )
        )
    )


def orbital_inclination(
    state: OrbitalState,
) -> float:
    """
    Return orbital inclination in radians.

    The inclination is the angle between the specific
    angular-momentum vector and the +z axis.
    """

    h_vector = (
        specific_angular_momentum(
            state
        )
    )

    h = np.linalg.norm(
        h_vector
    )

    if h == 0.0:
        raise ValueError(
            "Orbital inclination is undefined "
            "for zero angular momentum."
        )

    # Numerical clipping protects arccos from values such as
    # 1.0000000000000002 caused by floating-point roundoff.
    cos_i = np.clip(
        h_vector[2] / h,
        -1.0,
        1.0,
    )

    return float(
        np.arccos(
            cos_i
        )
    )


def semi_major_axis(
    state: OrbitalState,
    body: CelestialBody,
) -> float:
    """
    Return semi-major axis from specific orbital energy.

    For a parabolic orbit, the semi-major axis is infinite.
    Hyperbolic trajectories naturally return a negative value.
    """

    energy = specific_orbital_energy(
        state,
        body,
    )

    tolerance = 1e-12

    if abs(energy) <= tolerance:
        return float("inf")

    return float(
        -body.mu
        / (
            2.0
            * energy
        )
    )


def apsis_radii(
    state: OrbitalState,
    body: CelestialBody,
) -> tuple[float, float]:
    """
    Return periapsis and apoapsis radii for a bound
    elliptical orbit.

    Returns
    -------
    tuple
        (periapsis_radius, apoapsis_radius)
    """

    energy = specific_orbital_energy(
        state,
        body,
    )

    if energy >= 0.0:
        raise ValueError(
            "A finite apoapsis exists only for a bound orbit."
        )

    a = semi_major_axis(
        state,
        body,
    )

    e = orbital_eccentricity(
        state,
        body,
    )

    if e >= 1.0:
        raise ValueError(
            "A finite apoapsis exists only for an elliptical orbit."
        )

    periapsis = (
        a
        * (1.0 - e)
    )

    apoapsis = (
        a
        * (1.0 + e)
    )

    return (
        float(periapsis),
        float(apoapsis),
    )


def periapsis_radius(
    state: OrbitalState,
    body: CelestialBody,
) -> float:
    """
    Return periapsis radius for a bound orbit.
    """

    periapsis, _ = apsis_radii(
        state,
        body,
    )

    return periapsis


def apoapsis_radius(
    state: OrbitalState,
    body: CelestialBody,
) -> float:
    """
    Return apoapsis radius for a bound orbit.
    """

    _, apoapsis = apsis_radii(
        state,
        body,
    )

    return apoapsis


def orbital_period_from_state(
    state: OrbitalState,
    body: CelestialBody,
) -> float:
    """
    Return orbital period from a Cartesian state.

    Only bound elliptical orbits possess a finite period.
    """

    energy = specific_orbital_energy(
        state,
        body,
    )

    if energy >= 0.0:
        raise ValueError(
            "The current orbit is not bound. "
            "A finite orbital period does not exist."
        )

    a = semi_major_axis(
        state,
        body,
    )

    return float(
        2.0
        * pi
        * sqrt(
            a**3
            / body.mu
        )
    )


def compute_orbital_elements(
    state: OrbitalState,
    body: CelestialBody,
) -> OrbitalElements:
    """
    Compute the main orbital elements from a Cartesian state.
    """

    a = semi_major_axis(
        state,
        body,
    )

    e_vector = eccentricity_vector(
        state,
        body,
    )

    e = float(
        np.linalg.norm(
            e_vector
        )
    )

    h_vector = (
        specific_angular_momentum(
            state
        )
    )

    inclination = orbital_inclination(
        state
    )

    energy = specific_orbital_energy(
        state,
        body,
    )

    if (
        energy < 0.0
        and e < 1.0
    ):

        rp = (
            a
            * (1.0 - e)
        )

        ra = (
            a
            * (1.0 + e)
        )

    else:

        rp = None
        ra = None

    return OrbitalElements(
        semi_major_axis=float(
            a
        ),
        eccentricity=e,
        inclination=inclination,
        periapsis_radius=(
            None
            if rp is None
            else float(rp)
        ),
        apoapsis_radius=(
            None
            if ra is None
            else float(ra)
        ),
        eccentricity_vector=(
            e_vector.copy()
        ),
        angular_momentum_vector=(
            h_vector.copy()
        ),
    )


# --------------------------------------------------
# Compatibility aliases
# --------------------------------------------------

def orbital_elements(
    state: OrbitalState,
    body: CelestialBody,
) -> OrbitalElements:
    """
    Alias for compute_orbital_elements().
    """

    return compute_orbital_elements(
        state,
        body,
    )


def state_to_orbital_elements(
    state: OrbitalState,
    body: CelestialBody,
) -> OrbitalElements:
    """
    Alias for compute_orbital_elements().
    """

    return compute_orbital_elements(
        state,
        body,
    )