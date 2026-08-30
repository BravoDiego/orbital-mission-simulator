"""Classical orbital elements computation."""

from dataclasses import dataclass

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody


@dataclass(frozen=True)
class OrbitalElements:
    """Classical orbital parameters reconstructed from a state."""

    semi_major_axis: float
    eccentricity: float

    inclination: float

    raan: float | None
    argument_of_periapsis: float | None
    true_anomaly: float | None

    periapsis_radius: float
    apoapsis_radius: float | None

    period: float | None

    @property
    def inclination_deg(self) -> float:
        return np.degrees(self.inclination)

    @property
    def raan_deg(self) -> float | None:
        if self.raan is None:
            return None

        return np.degrees(self.raan)

    @property
    def argument_of_periapsis_deg(
        self,
    ) -> float | None:
        if self.argument_of_periapsis is None:
            return None

        return np.degrees(
            self.argument_of_periapsis
        )

    @property
    def true_anomaly_deg(self) -> float | None:
        if self.true_anomaly is None:
            return None

        return np.degrees(
            self.true_anomaly
        )


def orbital_elements(
    state: OrbitalState,
    body: CelestialBody,
) -> OrbitalElements:
    """Compute classical orbital elements from position and velocity."""

    r_vector = state.position
    v_vector = state.velocity

    mu = body.mu

    r = np.linalg.norm(r_vector)
    v = np.linalg.norm(v_vector)

    if r == 0:
        raise ValueError(
            "Orbital elements are undefined at r = 0."
        )

    # -------------------------------------------------
    # Specific angular momentum
    # -------------------------------------------------

    h_vector = np.cross(
        r_vector,
        v_vector,
    )

    h = np.linalg.norm(h_vector)

    if h == 0:
        raise ValueError(
            "Orbital elements are undefined "
            "for zero angular momentum."
        )

    # -------------------------------------------------
    # Specific orbital energy
    # -------------------------------------------------

    energy = (
        0.5 * v**2
        - mu / r
    )

    # -------------------------------------------------
    # Eccentricity vector
    # -------------------------------------------------

    e_vector = (
        np.cross(v_vector, h_vector) / mu
        - r_vector / r
    )

    eccentricity = np.linalg.norm(
        e_vector
    )

    # -------------------------------------------------
    # Semi-major axis
    # -------------------------------------------------

    energy_scale = mu / r

    if abs(energy) < 1e-12 * energy_scale:
        semi_major_axis = np.inf
    else:
        semi_major_axis = (
            -mu / (2.0 * energy)
        )

    # -------------------------------------------------
    # Inclination
    # -------------------------------------------------

    inclination = np.arccos(
        np.clip(
            h_vector[2] / h,
            -1.0,
            1.0,
        )
    )

    # -------------------------------------------------
    # Node vector
    # -------------------------------------------------

    z_axis = np.array([
        0.0,
        0.0,
        1.0,
    ])

    node_vector = np.cross(
        z_axis,
        h_vector,
    )

    node = np.linalg.norm(
        node_vector
    )

    tolerance = 1e-12

    equatorial = (
        node / h
        < tolerance
    )

    circular = (
        eccentricity
        < tolerance
    )

    # -------------------------------------------------
    # RAAN Ω
    # -------------------------------------------------

    if equatorial:
        raan = None

    else:
        raan = np.mod(
            np.arctan2(
                node_vector[1],
                node_vector[0],
            ),
            2.0 * np.pi,
        )

    # -------------------------------------------------
    # Argument of periapsis ω
    # -------------------------------------------------

    if equatorial or circular:
        argument_of_periapsis = None

    else:
        cos_argument = (
            np.dot(
                node_vector,
                e_vector,
            )
            / (node * eccentricity)
        )

        sin_argument = (
            np.dot(
                np.cross(
                    node_vector,
                    e_vector,
                ),
                h_vector,
            )
            / (
                node
                * eccentricity
                * h
            )
        )

        argument_of_periapsis = np.mod(
            np.arctan2(
                sin_argument,
                cos_argument,
            ),
            2.0 * np.pi,
        )

    # -------------------------------------------------
    # True anomaly ν
    # -------------------------------------------------

    if circular:
        true_anomaly = None

    else:
        cos_anomaly = (
            np.dot(
                e_vector,
                r_vector,
            )
            / (
                eccentricity
                * r
            )
        )

        sin_anomaly = (
            np.dot(
                np.cross(
                    e_vector,
                    r_vector,
                ),
                h_vector,
            )
            / (
                eccentricity
                * r
                * h
            )
        )

        true_anomaly = np.mod(
            np.arctan2(
                sin_anomaly,
                cos_anomaly,
            ),
            2.0 * np.pi,
        )

    # -------------------------------------------------
    # Periapsis / apoapsis
    # -------------------------------------------------

    semi_latus_rectum = (
        h**2 / mu
    )

    periapsis_radius = (
        semi_latus_rectum
        / (1.0 + eccentricity)
    )

    if eccentricity < 1.0:
        apoapsis_radius = (
            semi_latus_rectum
            / (1.0 - eccentricity)
        )
    else:
        apoapsis_radius = None

    # -------------------------------------------------
    # Orbital period
    # -------------------------------------------------

    if (
        eccentricity < 1.0
        and semi_major_axis > 0
    ):
        period = (
            2.0
            * np.pi
            * np.sqrt(
                semi_major_axis**3
                / mu
            )
        )
    else:
        period = None

    return OrbitalElements(
        semi_major_axis=semi_major_axis,
        eccentricity=eccentricity,
        inclination=inclination,
        raan=raan,
        argument_of_periapsis=argument_of_periapsis,
        true_anomaly=true_anomaly,
        periapsis_radius=periapsis_radius,
        apoapsis_radius=apoapsis_radius,
        period=period,
    )