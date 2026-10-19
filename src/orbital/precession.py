"""Analytical first-order secular J2 precession models."""

from math import acos, cos, pi, sqrt

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody
from src.orbital.elements import compute_orbital_elements


def _validate_elliptical_elements(
    semi_major_axis: float,
    eccentricity: float,
    j2: float,
) -> None:
    """Validate inputs shared by the secular J2 formulas."""

    if semi_major_axis <= 0.0:
        raise ValueError(
            "Semi-major axis must be positive."
        )

    if not 0.0 <= eccentricity < 1.0:
        raise ValueError(
            "The secular J2 formula requires 0 <= e < 1."
        )

    if j2 < 0.0:
        raise ValueError(
            "J2 must be non-negative."
        )


def _mean_motion_and_semi_latus_rectum(
    semi_major_axis: float,
    eccentricity: float,
    body: CelestialBody,
) -> tuple[float, float]:
    """Return Keplerian mean motion n and semi-latus rectum p."""

    mean_motion = sqrt(
        body.mu
        / semi_major_axis**3
    )

    semi_latus_rectum = (
        semi_major_axis
        * (1.0 - eccentricity**2)
    )

    return (
        mean_motion,
        semi_latus_rectum,
    )


# ==================================================
# NODAL PRECESSION
# ==================================================


def j2_nodal_precession_rate_from_elements(
    semi_major_axis: float,
    eccentricity: float,
    inclination: float,
    body: CelestialBody,
    j2: float,
) -> float:
    """Return first-order secular nodal rate dOmega/dt in rad/s."""

    _validate_elliptical_elements(
        semi_major_axis,
        eccentricity,
        j2,
    )

    (
        mean_motion,
        semi_latus_rectum,
    ) = _mean_motion_and_semi_latus_rectum(
        semi_major_axis,
        eccentricity,
        body,
    )

    return float(
        -1.5
        * j2
        * mean_motion
        * (
            body.radius
            / semi_latus_rectum
        )**2
        * cos(inclination)
    )


def j2_nodal_precession_rate(
    state: OrbitalState,
    body: CelestialBody,
    j2: float,
) -> float:
    """Return first-order secular J2 nodal rate for a Cartesian state."""

    elements = compute_orbital_elements(
        state,
        body,
    )

    return j2_nodal_precession_rate_from_elements(
        semi_major_axis=elements.semi_major_axis,
        eccentricity=elements.eccentricity,
        inclination=elements.inclination,
        body=body,
        j2=j2,
    )


# ==================================================
# APSIDAL PRECESSION
# ==================================================


def j2_apsidal_precession_rate_from_elements(
    semi_major_axis: float,
    eccentricity: float,
    inclination: float,
    body: CelestialBody,
    j2: float,
) -> float:
    """Return first-order secular apsidal rate domega/dt in rad/s."""

    _validate_elliptical_elements(
        semi_major_axis,
        eccentricity,
        j2,
    )

    (
        mean_motion,
        semi_latus_rectum,
    ) = _mean_motion_and_semi_latus_rectum(
        semi_major_axis,
        eccentricity,
        body,
    )

    return float(
        0.75
        * j2
        * mean_motion
        * (
            body.radius
            / semi_latus_rectum
        )**2
        * (
            5.0 * cos(inclination)**2
            - 1.0
        )
    )


def j2_apsidal_precession_rate(
    state: OrbitalState,
    body: CelestialBody,
    j2: float,
) -> float:
    """Return first-order secular J2 apsidal rate for a Cartesian state."""

    elements = compute_orbital_elements(
        state,
        body,
    )

    if elements.argument_of_periapsis is None:
        raise ValueError(
            "Argument of periapsis must be defined; "
            "use a non-circular, non-equatorial orbit."
        )

    return j2_apsidal_precession_rate_from_elements(
        semi_major_axis=elements.semi_major_axis,
        eccentricity=elements.eccentricity,
        inclination=elements.inclination,
        body=body,
        j2=j2,
    )


# ==================================================
# CRITICAL INCLINATIONS
# ==================================================


def j2_critical_inclinations() -> tuple[float, float]:
    """Return the two inclinations where first-order apsidal drift is zero."""

    first = acos(
        sqrt(1.0 / 5.0)
    )

    second = (
        pi
        - first
    )

    return (
        float(first),
        float(second),
    )
