"""Simple analytical ephemerides for perturbation experiments.

These models are intentionally lightweight. They are useful for validating
third-body dynamics without introducing an external ephemeris dependency.
"""

from dataclasses import dataclass
from math import pi

import numpy as np

from src.physics.constants import (
    MOON_MEAN_ORBIT_RADIUS,
    MOON_SIDEREAL_PERIOD,
)


@dataclass(frozen=True)
class CircularEphemeris:
    """Circular orbit ephemeris around the coordinate origin.

    Parameters are expressed in SI units and radians. The circular orbit is
    first defined in the xy plane, then rotated by ``inclination`` around x
    and by ``raan`` around z.
    """

    radius: float
    period: float
    inclination: float = 0.0
    raan: float = 0.0
    phase_at_epoch: float = 0.0

    def __post_init__(self):
        if self.radius <= 0.0:
            raise ValueError(
                "Ephemeris radius must be positive."
            )

        if self.period <= 0.0:
            raise ValueError(
                "Ephemeris period must be positive."
            )

    def position(
        self,
        time: float,
    ) -> np.ndarray:
        """Return inertial position at ``time`` in meters."""

        mean_motion = (
            2.0 * pi
            / self.period
        )

        theta = (
            self.phase_at_epoch
            + mean_motion * time
        )

        position_orbital = np.array([
            self.radius * np.cos(theta),
            self.radius * np.sin(theta),
            0.0,
        ])

        ci = np.cos(
            self.inclination
        )
        si = np.sin(
            self.inclination
        )

        rotation_x = np.array([
            [1.0, 0.0, 0.0],
            [0.0, ci, -si],
            [0.0, si, ci],
        ])

        cO = np.cos(
            self.raan
        )
        sO = np.sin(
            self.raan
        )

        rotation_z = np.array([
            [cO, -sO, 0.0],
            [sO, cO, 0.0],
            [0.0, 0.0, 1.0],
        ])

        return (
            rotation_z
            @ rotation_x
            @ position_orbital
        )

    def __call__(
        self,
        time: float,
    ) -> np.ndarray:
        return self.position(
            time
        )


def moon_circular_ephemeris(
    inclination: float = 0.0,
    raan: float = 0.0,
    phase_at_epoch: float = 0.0,
) -> CircularEphemeris:
    """Return the simplified circular Moon ephemeris used in V3.7.

    The default model is coplanar with the reference xy plane. It is not a
    precision lunar ephemeris; the purpose is to validate time-dependent
    third-body gravity before introducing real ephemeris data later.
    """

    return CircularEphemeris(
        radius=MOON_MEAN_ORBIT_RADIUS,
        period=MOON_SIDEREAL_PERIOD,
        inclination=inclination,
        raan=raan,
        phase_at_epoch=phase_at_epoch,
    )
