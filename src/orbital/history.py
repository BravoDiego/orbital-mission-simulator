"""Time histories of osculating orbital elements."""

from dataclasses import dataclass

import numpy as np

from src.orbital.state import OrbitalState
from src.orbital.elements import compute_orbital_elements
from src.physics.bodies import CelestialBody
from src.simulation.result import SimulationResult


@dataclass(frozen=True)
class OrbitalElementsHistory:
    """Osculating orbital elements evaluated along a simulation.

    Distances are expressed in meters and angular quantities in radians.
    Undefined angular elements are stored as ``NaN``.
    """

    times: np.ndarray

    semi_major_axis: np.ndarray
    eccentricity: np.ndarray
    inclination: np.ndarray

    raan: np.ndarray
    argument_of_periapsis: np.ndarray
    true_anomaly: np.ndarray

    periapsis_radius: np.ndarray
    apoapsis_radius: np.ndarray

    @property
    def raan_unwrapped(self) -> np.ndarray:
        """Return RAAN with 2π discontinuities removed."""

        return unwrap_defined_angles(self.raan)

    @property
    def argument_of_periapsis_unwrapped(self) -> np.ndarray:
        """Return argument of periapsis with 2π jumps removed."""

        return unwrap_defined_angles(
            self.argument_of_periapsis
        )

    @property
    def true_anomaly_unwrapped(self) -> np.ndarray:
        """Return true anomaly with 2π jumps removed."""

        return unwrap_defined_angles(
            self.true_anomaly
        )


def unwrap_defined_angles(
    angles: np.ndarray,
) -> np.ndarray:
    """Unwrap angular data while preserving undefined ``NaN`` samples.

    Each contiguous finite segment is unwrapped independently. This makes the
    helper safe for orbital elements that can become undefined in singular
    geometries such as circular or equatorial orbits.
    """

    values = np.asarray(
        angles,
        dtype=float,
    )

    if values.ndim != 1:
        raise ValueError(
            "Angles must be a one-dimensional array."
        )

    result = values.copy()

    finite = np.isfinite(
        values
    )

    start = 0

    while start < len(values):

        while (
            start < len(values)
            and not finite[start]
        ):
            start += 1

        if start >= len(values):
            break

        end = start + 1

        while (
            end < len(values)
            and finite[end]
        ):
            end += 1

        result[start:end] = np.unwrap(
            values[start:end]
        )

        start = end

    return result


def _optional_value(
    value: float | None,
) -> float:
    """Convert an optional scalar orbital element to float/NaN."""

    if value is None:
        return float("nan")

    return float(value)


def compute_orbital_elements_history(
    result: SimulationResult,
    body: CelestialBody,
) -> OrbitalElementsHistory:
    """Compute osculating elements at every simulation sample."""

    sample_count = len(
        result.times
    )

    semi_major_axis = np.empty(
        sample_count,
        dtype=float,
    )

    eccentricity = np.empty(
        sample_count,
        dtype=float,
    )

    inclination = np.empty(
        sample_count,
        dtype=float,
    )

    raan = np.empty(
        sample_count,
        dtype=float,
    )

    argument_of_periapsis = np.empty(
        sample_count,
        dtype=float,
    )

    true_anomaly = np.empty(
        sample_count,
        dtype=float,
    )

    periapsis_radius = np.empty(
        sample_count,
        dtype=float,
    )

    apoapsis_radius = np.empty(
        sample_count,
        dtype=float,
    )

    for index, (
        position,
        velocity,
    ) in enumerate(
        zip(
            result.positions,
            result.velocities,
        )
    ):

        state = OrbitalState(
            position=np.array(
                position,
                dtype=float,
                copy=True,
            ),
            velocity=np.array(
                velocity,
                dtype=float,
                copy=True,
            ),
        )

        elements = compute_orbital_elements(
            state,
            body,
        )

        semi_major_axis[index] = (
            elements.semi_major_axis
        )

        eccentricity[index] = (
            elements.eccentricity
        )

        inclination[index] = (
            elements.inclination
        )

        raan[index] = _optional_value(
            elements.raan
        )

        argument_of_periapsis[index] = (
            _optional_value(
                elements.argument_of_periapsis
            )
        )

        true_anomaly[index] = (
            _optional_value(
                elements.true_anomaly
            )
        )

        periapsis_radius[index] = (
            _optional_value(
                elements.periapsis_radius
            )
        )

        apoapsis_radius[index] = (
            _optional_value(
                elements.apoapsis_radius
            )
        )

    return OrbitalElementsHistory(
        times=np.array(
            result.times,
            dtype=float,
            copy=True,
        ),
        semi_major_axis=semi_major_axis,
        eccentricity=eccentricity,
        inclination=inclination,
        raan=raan,
        argument_of_periapsis=(
            argument_of_periapsis
        ),
        true_anomaly=true_anomaly,
        periapsis_radius=periapsis_radius,
        apoapsis_radius=apoapsis_radius,
    )


def secular_rate(
    times: np.ndarray,
    angles: np.ndarray,
) -> float:
    """Estimate a secular angular rate by least-squares linear fitting.

    Parameters
    ----------
    times
        Time samples in seconds.
    angles
        Angle samples in radians. ``NaN`` values are ignored and the finite
        values are unwrapped before fitting.

    Returns
    -------
    float
        Best-fit angular rate in rad/s.
    """

    times = np.asarray(
        times,
        dtype=float,
    )

    angles = np.asarray(
        angles,
        dtype=float,
    )

    if (
        times.ndim != 1
        or angles.ndim != 1
    ):
        raise ValueError(
            "Times and angles must be one-dimensional arrays."
        )

    if len(times) != len(angles):
        raise ValueError(
            "Times and angles must contain the same number of samples."
        )

    unwrapped = unwrap_defined_angles(
        angles
    )

    finite = (
        np.isfinite(times)
        & np.isfinite(unwrapped)
    )

    if np.count_nonzero(
        finite
    ) < 2:
        raise ValueError(
            "At least two finite angular samples are required."
        )

    return float(
        np.polyfit(
            times[finite],
            unwrapped[finite],
            deg=1,
        )[0]
    )
