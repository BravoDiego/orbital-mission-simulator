"""Orbital elements and geometric quantities."""

from dataclasses import dataclass, field
from math import pi, sqrt

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody

from src.orbital.conservation import (
    specific_orbital_energy,
    specific_angular_momentum,
)


ANGLE_TOLERANCE = 1e-12
ECCENTRICITY_TOLERANCE = 1e-10


@dataclass(frozen=True)
class OrbitalElements:
    """Main osculating orbital elements derived from a Cartesian state.

    Angular quantities are expressed in radians.

    ``raan`` is undefined for equatorial orbits.
    ``argument_of_periapsis`` and ``true_anomaly`` are undefined for
    circular orbits and are therefore represented by ``None``.
    """

    semi_major_axis: float
    eccentricity: float
    inclination: float

    periapsis_radius: float | None
    apoapsis_radius: float | None

    eccentricity_vector: np.ndarray
    angular_momentum_vector: np.ndarray

    # Full 3-D classical geometry. These defaults preserve compatibility
    # with code that may still instantiate OrbitalElements directly.
    raan: float | None = None
    argument_of_periapsis: float | None = None
    true_anomaly: float | None = None
    node_vector: np.ndarray = field(
        default_factory=lambda: np.zeros(3, dtype=float)
    )

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

    @property
    def omega(self) -> float | None:
        """Argument of periapsis ω."""
        return self.argument_of_periapsis

    @property
    def nu(self) -> float | None:
        """True anomaly ν."""
        return self.true_anomaly


def _normalize_angle(angle: float) -> float:
    """Return an angle in the interval [0, 2π)."""

    return float(
        angle % (2.0 * pi)
    )


def orbital_radius(
    state: OrbitalState,
) -> float:
    """Return the distance from the central body."""

    radius = np.linalg.norm(
        state.position
    )

    if radius == 0.0:
        raise ValueError(
            "Orbital radius is undefined at the body's center."
        )

    return float(radius)


def orbital_speed(
    state: OrbitalState,
) -> float:
    """Return spacecraft speed."""

    return float(
        np.linalg.norm(
            state.velocity
        )
    )


def radial_velocity(
    state: OrbitalState,
) -> float:
    """Return radial velocity.

    Positive means the spacecraft is moving away from the central body;
    negative means it is moving toward it.
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


def node_vector(
    state: OrbitalState,
) -> np.ndarray:
    """Return the ascending-node vector n = k × h."""

    h_vector = specific_angular_momentum(
        state
    )

    return np.array([
        -h_vector[1],
        h_vector[0],
        0.0,
    ])


def eccentricity_vector(
    state: OrbitalState,
    body: CelestialBody,
) -> np.ndarray:
    """Return the orbital eccentricity vector."""

    radius = orbital_radius(
        state
    )

    angular_momentum = specific_angular_momentum(
        state
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
    """Return orbital eccentricity."""

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
    """Return orbital inclination in radians."""

    h_vector = specific_angular_momentum(
        state
    )

    h = np.linalg.norm(
        h_vector
    )

    if h == 0.0:
        raise ValueError(
            "Orbital inclination is undefined for zero angular momentum."
        )

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


def right_ascension_of_ascending_node(
    state: OrbitalState,
) -> float | None:
    """Return the right ascension of the ascending node Ω.

    ``None`` is returned for equatorial orbits, for which the line of
    nodes does not exist.
    """

    h_vector = specific_angular_momentum(
        state
    )

    h = np.linalg.norm(
        h_vector
    )

    if h == 0.0:
        raise ValueError(
            "RAAN is undefined for zero angular momentum."
        )

    n_vector = node_vector(
        state
    )

    n = np.linalg.norm(
        n_vector
    )

    if n <= ANGLE_TOLERANCE * h:
        return None

    return _normalize_angle(
        np.arctan2(
            n_vector[1],
            n_vector[0],
        )
    )


def argument_of_periapsis(
    state: OrbitalState,
    body: CelestialBody,
) -> float | None:
    """Return the argument of periapsis ω in radians.

    It is undefined for circular or equatorial orbits.
    """

    h_vector = specific_angular_momentum(
        state
    )

    h = np.linalg.norm(
        h_vector
    )

    if h == 0.0:
        raise ValueError(
            "Argument of periapsis is undefined for zero angular momentum."
        )

    n_vector = node_vector(
        state
    )
    n = np.linalg.norm(
        n_vector
    )

    e_vector = eccentricity_vector(
        state,
        body,
    )
    e = np.linalg.norm(
        e_vector
    )

    if e <= ECCENTRICITY_TOLERANCE:
        return None

    if n <= ANGLE_TOLERANCE * h:
        return None

    cosine = np.clip(
        np.dot(
            n_vector,
            e_vector,
        )
        / (n * e),
        -1.0,
        1.0,
    )

    sine = (
        np.dot(
            np.cross(
                n_vector,
                e_vector,
            ),
            h_vector,
        )
        / (n * e * h)
    )

    return _normalize_angle(
        np.arctan2(
            sine,
            cosine,
        )
    )


def true_anomaly(
    state: OrbitalState,
    body: CelestialBody,
) -> float | None:
    """Return true anomaly ν in radians.

    Classical true anomaly is undefined for a circular orbit because
    there is no unique periapsis direction.
    """

    radius = orbital_radius(
        state
    )

    h_vector = specific_angular_momentum(
        state
    )
    h = np.linalg.norm(
        h_vector
    )

    if h == 0.0:
        raise ValueError(
            "True anomaly is undefined for zero angular momentum."
        )

    e_vector = eccentricity_vector(
        state,
        body,
    )
    e = np.linalg.norm(
        e_vector
    )

    if e <= ECCENTRICITY_TOLERANCE:
        return None

    cosine = np.clip(
        np.dot(
            e_vector,
            state.position,
        )
        / (e * radius),
        -1.0,
        1.0,
    )

    sine = (
        np.dot(
            np.cross(
                e_vector,
                state.position,
            ),
            h_vector,
        )
        / (e * radius * h)
    )

    return _normalize_angle(
        np.arctan2(
            sine,
            cosine,
        )
    )


def semi_major_axis(
    state: OrbitalState,
    body: CelestialBody,
) -> float:
    """Return semi-major axis from specific orbital energy.

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
    """Return periapsis and apoapsis radii for a bound ellipse."""

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

    periapsis = a * (1.0 - e)
    apoapsis = a * (1.0 + e)

    return (
        float(periapsis),
        float(apoapsis),
    )


def periapsis_radius(
    state: OrbitalState,
    body: CelestialBody,
) -> float:
    """Return periapsis radius for a bound orbit."""

    periapsis, _ = apsis_radii(
        state,
        body,
    )

    return periapsis


def apoapsis_radius(
    state: OrbitalState,
    body: CelestialBody,
) -> float:
    """Return apoapsis radius for a bound orbit."""

    _, apoapsis = apsis_radii(
        state,
        body,
    )

    return apoapsis


def orbital_period_from_state(
    state: OrbitalState,
    body: CelestialBody,
) -> float:
    """Return orbital period from a Cartesian state.

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
    """Compute the main osculating elements from a Cartesian state."""

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

    h_vector = specific_angular_momentum(
        state
    )

    inclination = orbital_inclination(
        state
    )

    n_vector = node_vector(
        state
    )

    raan = right_ascension_of_ascending_node(
        state
    )

    omega = argument_of_periapsis(
        state,
        body,
    )

    nu = true_anomaly(
        state,
        body,
    )

    energy = specific_orbital_energy(
        state,
        body,
    )

    if energy < 0.0 and e < 1.0:
        rp = a * (1.0 - e)
        ra = a * (1.0 + e)
    else:
        rp = None
        ra = None

    return OrbitalElements(
        semi_major_axis=float(a),
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
        raan=raan,
        argument_of_periapsis=omega,
        true_anomaly=nu,
        node_vector=(
            n_vector.copy()
        ),
    )


# --------------------------------------------------
# Compatibility aliases
# --------------------------------------------------

def orbital_elements(
    state: OrbitalState,
    body: CelestialBody,
) -> OrbitalElements:
    """Alias for compute_orbital_elements()."""

    return compute_orbital_elements(
        state,
        body,
    )


def state_to_orbital_elements(
    state: OrbitalState,
    body: CelestialBody,
) -> OrbitalElements:
    """Alias for compute_orbital_elements()."""

    return compute_orbital_elements(
        state,
        body,
    )
