"""Atmospheric-drag perturbation models."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody
from src.physics.atmosphere import earth_exponential_density


DensityModel = Callable[[float], float]


@dataclass(frozen=True)
class DragProperties:
    """Spacecraft properties used by the aerodynamic drag model."""

    mass: float
    cross_section_area: float
    drag_coefficient: float = 2.2

    def __post_init__(self):
        if self.mass <= 0.0:
            raise ValueError(
                "Spacecraft mass must be positive."
            )

        if self.cross_section_area < 0.0:
            raise ValueError(
                "Cross-sectional area cannot be negative."
            )

        if self.drag_coefficient < 0.0:
            raise ValueError(
                "Drag coefficient cannot be negative."
            )

    @property
    def area_to_mass_ratio(self) -> float:
        """Return A/m in m^2/kg."""

        return float(
            self.cross_section_area
            / self.mass
        )

    @property
    def ballistic_coefficient(self) -> float:
        """Return m/(Cd A) in kg/m^2."""

        denominator = (
            self.drag_coefficient
            * self.cross_section_area
        )

        if denominator == 0.0:
            return float("inf")

        return float(
            self.mass
            / denominator
        )


def atmosphere_velocity(
    state: OrbitalState,
    rotation_rate: float,
) -> np.ndarray:
    """Return inertial velocity of a rigidly co-rotating atmosphere.

    The central body's rotation axis is assumed to be +z.
    """

    if rotation_rate < 0.0:
        raise ValueError(
            "Atmospheric rotation rate must be non-negative."
        )

    angular_velocity = np.array([
        0.0,
        0.0,
        rotation_rate,
    ])

    return np.cross(
        angular_velocity,
        state.position,
    )


def atmospheric_relative_velocity(
    state: OrbitalState,
    rotation_rate: float,
) -> np.ndarray:
    """Return spacecraft velocity relative to the atmosphere."""

    return (
        state.velocity
        - atmosphere_velocity(
            state,
            rotation_rate,
        )
    )


def drag_acceleration(
    state: OrbitalState,
    body: CelestialBody,
    properties: DragProperties,
    density_model: DensityModel = earth_exponential_density,
    rotation_rate: float = 0.0,
) -> np.ndarray:
    """Return atmospheric-drag acceleration in m/s^2.

    The model is

        a_D = -0.5 rho Cd (A/m) |v_rel| v_rel

    where ``v_rel`` is measured relative to a rigidly co-rotating
    atmosphere.
    """

    radius = float(
        np.linalg.norm(
            state.position
        )
    )

    if radius == 0.0:
        raise ValueError(
            "Drag acceleration is undefined at the body's center."
        )

    altitude = (
        radius
        - body.radius
    )

    density = float(
        density_model(
            altitude
        )
    )

    if not np.isfinite(density):
        raise ValueError(
            "Atmospheric density must be finite."
        )

    if density < 0.0:
        raise ValueError(
            "Atmospheric density cannot be negative."
        )

    relative_velocity = (
        atmospheric_relative_velocity(
            state,
            rotation_rate,
        )
    )

    relative_speed = float(
        np.linalg.norm(
            relative_velocity
        )
    )

    if (
        density == 0.0
        or relative_speed == 0.0
        or properties.cross_section_area == 0.0
        or properties.drag_coefficient == 0.0
    ):
        return np.zeros(
            3,
            dtype=float,
        )

    coefficient = (
        -0.5
        * density
        * properties.drag_coefficient
        * properties.area_to_mass_ratio
        * relative_speed
    )

    return (
        coefficient
        * relative_velocity
    )


def make_drag_perturbation(
    properties: DragProperties,
    density_model: DensityModel = earth_exponential_density,
    rotation_rate: float = 0.0,
) -> Callable[
    [OrbitalState, CelestialBody],
    np.ndarray,
]:
    """Create a drag perturbation compatible with ``build_state_derivative``."""

    def perturbation(
        state: OrbitalState,
        body: CelestialBody,
    ) -> np.ndarray:

        return drag_acceleration(
            state=state,
            body=body,
            properties=properties,
            density_model=density_model,
            rotation_rate=rotation_rate,
        )

    return perturbation
