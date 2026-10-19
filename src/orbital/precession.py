from math import cos, sqrt

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody
from src.orbital.elements import compute_orbital_elements


def j2_nodal_precession_rate_from_elements(
    semi_major_axis: float,
    eccentricity: float,
    inclination: float,
    body: CelestialBody,
    j2: float,
) -> float:
    """Return the first-order secular J2 nodal precession rate dΩ/dt.

    The result is in rad/s.
    """

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

    mean_motion = sqrt(
        body.mu
        / semi_major_axis**3
    )

    semi_latus_rectum = (
        semi_major_axis
        * (1.0 - eccentricity**2)
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
    """Return the first-order secular J2 nodal rate for a state."""

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
