from dataclasses import dataclass
from math import pi, sqrt

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody
from src.physics.two_body import state_derivative

from src.integrators.rk4 import RK4

from src.simulation.simulator import Simulator
from src.simulation.result import SimulationResult

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
    Initial setup of a Hohmann transfer mission.

    Contains:
    - analytical transfer data,
    - initial circular-orbit state,
    - first impulsive burn.
    """

    transfer: HohmannTransfer
    initial_state: OrbitalState
    burn1: ManeuverResult


@dataclass(frozen=True)
class HohmannMissionResult:
    """
    Complete numerical Hohmann mission.

    Contains:
    - mission setup,
    - numerical transfer trajectory,
    - second burn,
    - numerical trajectory on the final circular orbit.
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

    Parameters
    ----------
    radius
        Orbital radius in meters.

    body
        Central celestial body.

    angle
        Initial angular position in radians,
        measured counterclockwise from the +x axis.

    Returns
    -------
    OrbitalState
        Position and velocity corresponding to a
        prograde circular orbit.
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

    # Radial unit vector
    radial_direction = np.array([
        np.cos(angle),
        np.sin(angle),
        0.0,
    ])

    # Prograde tangential unit vector
    tangential_direction = np.array([
        -np.sin(angle),
        np.cos(angle),
        0.0,
    ])

    position = (
        radius
        * radial_direction
    )

    velocity = (
        speed
        * tangential_direction
    )

    return OrbitalState(
        position=position,
        velocity=velocity,
    )


def prepare_hohmann_transfer(
    r1: float,
    r2: float,
    body: CelestialBody,
    initial_angle: float = 0.0,
) -> HohmannMissionSetup:
    """
    Prepare a Hohmann transfer between two circular orbits.

    initial_angle gives the spacecraft's initial position
    around the central body, in radians.
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

    initial_state = create_circular_orbit_state(
        radius=r1,
        body=body,
        angle=initial_angle,
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


def orbital_period(
    radius: float,
    body: CelestialBody,
) -> float:
    """
    Return the period of a circular orbit.

    Parameters
    ----------
    radius
        Orbital radius in meters.

    body
        Central celestial body.

    Returns
    -------
    float
        Orbital period in seconds.
    """

    if radius <= 0.0:
        raise ValueError(
            "Orbital radius must be positive."
        )

    return (
        2.0
        * pi
        * sqrt(radius**3 / body.mu)
    )


def create_simulator(
    body: CelestialBody,
    dt: float,
) -> Simulator:
    """
    Create an RK4 two-body simulator for the given body.
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

    Mission sequence
    ----------------
    1. Start on a circular orbit of radius r1.
    2. Apply burn 1.
    3. Propagate numerically along the transfer ellipse.
    4. Apply burn 2 at the opposite apsis.
    5. Propagate for one complete final circular orbit.

    All propagation is performed using RK4.
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
        initial_state=setup.burn1.state_after,
        duration=setup.transfer.transfer_time,
    )

    state_before_burn2 = (
        transfer_result.final_state
    )

    # --------------------------------------------------
    # Burn 2: circularization
    # --------------------------------------------------

    burn2 = tangential_burn(
        state_before_burn2,
        setup.transfer.delta_v2,
    )

    # --------------------------------------------------
    # Final circular orbit
    # --------------------------------------------------

    final_period = orbital_period(
        radius=r2,
        body=body,
    )

    final_orbit_result = simulator.run(
        initial_state=burn2.state_after,
        duration=final_period,
    )

    return HohmannMissionResult(
        setup=setup,
        transfer_result=transfer_result,
        burn2=burn2,
        final_orbit_result=final_orbit_result,
    )