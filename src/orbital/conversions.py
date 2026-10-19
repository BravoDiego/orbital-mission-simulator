"""Conversions between classical orbital elements and Cartesian states."""

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody


def _rotation_z(
    angle: float,
) -> np.ndarray:
    """Return the right-handed rotation matrix about +z."""

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [c, -s, 0.0],
        [s, c, 0.0],
        [0.0, 0.0, 1.0],
    ])


def _rotation_x(
    angle: float,
) -> np.ndarray:
    """Return the right-handed rotation matrix about +x."""

    c = np.cos(angle)
    s = np.sin(angle)

    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, c, -s],
        [0.0, s, c],
    ])


def state_from_classical_elements(
    semi_major_axis: float,
    eccentricity: float,
    inclination: float,
    raan: float,
    argument_of_periapsis: float,
    true_anomaly: float,
    body: CelestialBody,
) -> OrbitalState:
    """
    Create a Cartesian state from classical Keplerian elements.

    Parameters use SI units and radians.

    This conversion currently supports bound elliptical
    orbits with 0 <= e < 1.
    """

    if semi_major_axis <= 0.0:
        raise ValueError(
            "Semi-major axis must be positive."
        )

    if not (
        0.0
        <= eccentricity
        < 1.0
    ):
        raise ValueError(
            "This conversion requires "
            "0 <= eccentricity < 1."
        )

    if not (
        0.0
        <= inclination
        <= np.pi
    ):
        raise ValueError(
            "Inclination must lie in [0, pi]."
        )

    semi_latus_rectum = (
        semi_major_axis
        * (
            1.0
            - eccentricity**2
        )
    )

    denominator = (
        1.0
        + eccentricity
        * np.cos(
            true_anomaly
        )
    )

    if denominator <= 0.0:
        raise ValueError(
            "The supplied elements do not define "
            "a valid elliptical radius."
        )

    radius = (
        semi_latus_rectum
        / denominator
    )

    # --------------------------------------------------
    # State in the perifocal frame PQW
    # --------------------------------------------------

    position_perifocal = np.array([
        radius
        * np.cos(
            true_anomaly
        ),

        radius
        * np.sin(
            true_anomaly
        ),

        0.0,
    ])

    velocity_scale = np.sqrt(
        body.mu
        / semi_latus_rectum
    )

    velocity_perifocal = (
        velocity_scale
        * np.array([
            -np.sin(
                true_anomaly
            ),

            eccentricity
            + np.cos(
                true_anomaly
            ),

            0.0,
        ])
    )

    # --------------------------------------------------
    # PQW -> inertial frame
    #
    # R3(Omega) R1(i) R3(omega)
    # --------------------------------------------------

    perifocal_to_inertial = (
        _rotation_z(
            raan
        )
        @ _rotation_x(
            inclination
        )
        @ _rotation_z(
            argument_of_periapsis
        )
    )

    return OrbitalState(
        position=(
            perifocal_to_inertial
            @ position_perifocal
        ),
        velocity=(
            perifocal_to_inertial
            @ velocity_perifocal
        ),
    )