"""Celestial bodies used by the orbital simulator."""

from dataclasses import dataclass
from .constants import *


@dataclass(frozen=True)
class CelestialBody:
    """Represent a celestial body.

    Parameters
    ----------
    name : str
        Name of the celestial body.
    mass : float
        Mass in kilograms.
    radius : float
        Mean/reference radius in meters.
    mu : float
        Standard gravitational parameter in m^3/s^2.
    """

    name: str
    mass: float
    radius: float
    mu: float

    def __post_init__(self):
        if self.mass <= 0:
            raise ValueError("Mass must be positive.")

        if self.radius <= 0:
            raise ValueError("Radius must be positive.")

        if self.mu <= 0:
            raise ValueError("Gravitational parameter mu must be positive.")


EARTH = CelestialBody(
    name="Earth",
    mass=EARTH_MASS,
    radius=EARTH_RADIUS,
    mu=EARTH_MU,
)