"""Numerical Hohmann mission simulation."""

from dataclasses import dataclass

import numpy as np

from src.orbital.state import OrbitalState

from src.physics.bodies import CelestialBody
from src.physics.two_body import state_derivative

from src.integrators.rk4 import RK4

from src.simulation.simulator import Simulator
from src.simulation.result import SimulationResult

from src.orbital.elements import (
    orbital_period_from_state,
)

from src.orbital.transfers import (
    HohmannTransfer,
    circular_velocity,
    hohmann_transfer,
)

from src.mission.maneuver import (
    ManeuverResult,
    tangential_burn,
)


@dataclass(frozen=True)
class HohmannMissionSetup:
    """
    Initial setup of a Hohmann transfer.
    """

    transfer: HohmannTransfer

    initial_state: OrbitalState

    burn1: ManeuverResult


@dataclass(frozen=True)
class HohmannMissionResult:
    """
    Complete numerical Hohmann mission.
    """

    setup: HohmannMissionSetup

    transfer_result: SimulationResult

    burn2: ManeuverResult

    final_orbit_result: SimulationResult


def create_circular_orbit_state(
    radius: float,
    body: CelestialBody,
    angle: float = 0.0,
) -> OrbitalState:
    """
    Create a prograde circular orbit in the xy plane.

    angle is measured counterclockwise from +x.
    """

    if radius <= body.radius:
        raise ValueError(
            "Orbital radius must be greater "
            "than the body's radius."
        )

    speed = circular_velocity(
        radius,
        body.mu,
    )

    radial_direction = np.array([
        np.cos(angle),
        np.sin(angle),
        0.0,
    ])

    tangential_direction = np.array([
        -np.sin(angle),
        np.cos(angle),
        0.0,
    ])

    return OrbitalState(
        position=(
            radius
            * radial_direction
        ),
        velocity=(
            speed
            * tangential_direction
        ),
    )


def orbital_period(
    radius: float,
    body: CelestialBody,
) -> float:
    """
    Compatibility helper returning the period of a
    circular orbit.

    The actual period calculation is delegated to
    orbital.elements.orbital_period_from_state().
    """

    state = create_circular_orbit_state(
        radius=radius,
        body=body,
    )

    return orbital_period_from_state(
        state,
        body,
    )


def prepare_hohmann_transfer(
    r1: float,
    r2: float,
    body: CelestialBody,
    initial_angle: float = 0.0,
) -> HohmannMissionSetup:
    """
    Prepare a Hohmann transfer without propagation.
    """

    if r1 <= body.radius:
        raise ValueError(
            "Initial orbital radius must be above "
            "the body's surface."
        )

    if r2 <= body.radius:
        raise ValueError(
            "Final orbital radius must be above "
            "the body's surface."
        )

    transfer = hohmann_transfer(
        r1=r1,
        r2=r2,
        mu=body.mu,
    )

    initial_state = (
        create_circular_orbit_state(
            radius=r1,
            body=body,
            angle=initial_angle,
        )
    )

    burn1 = tangential_burn(
        initial_state,
        transfer.delta_v1,
    )

    return HohmannMissionSetup(
        transfer=transfer,
        initial_state=initial_state,
        burn1=burn1,
    )


def create_simulator(
    body: CelestialBody,
    dt: float,
) -> Simulator:
    """
    Create an RK4 two-body simulator.
    """

    if dt <= 0.0:
        raise ValueError(
            "Time step dt must be positive."
        )

    derivative = lambda state: state_derivative(
        state,
        body,
    )

    return Simulator(
        integrator=RK4(),
        derivative=derivative,
        dt=dt,
    )


def simulate_hohmann_mission(
    r1: float,
    r2: float,
    body: CelestialBody,
    dt: float = 10.0,
    initial_angle: float = 0.0,
) -> HohmannMissionResult:
    """
    Simulate a complete ideal Hohmann transfer.
    """

    if r1 == r2:
        raise ValueError(
            "Initial and final orbital radii must differ."
        )

    setup = prepare_hohmann_transfer(
        r1=r1,
        r2=r2,
        body=body,
        initial_angle=initial_angle,
    )

    simulator = create_simulator(
        body=body,
        dt=dt,
    )

    # --------------------------------------------------
    # Transfer ellipse
    # --------------------------------------------------

    transfer_result = simulator.run(
        initial_state=(
            setup
            .burn1
            .state_after
        ),
        duration=(
            setup
            .transfer
            .transfer_time
        ),
    )

    # --------------------------------------------------
    # Circularization burn
    # --------------------------------------------------

    state_before_burn2 = (
        transfer_result
        .final_state
    )

    burn2 = tangential_burn(
        state_before_burn2,
        setup
        .transfer
        .delta_v2,
    )

    # --------------------------------------------------
    # Final circular orbit
    # --------------------------------------------------

    final_period = (
        orbital_period_from_state(
            burn2.state_after,
            body,
        )
    )

    final_orbit_result = (
        simulator.run(
            initial_state=(
                burn2.state_after
            ),
            duration=(
                final_period
            ),
        )
    )

    return HohmannMissionResult(
        setup=setup,
        transfer_result=(
            transfer_result
        ),
        burn2=burn2,
        final_orbit_result=(
            final_orbit_result
        ),
    )