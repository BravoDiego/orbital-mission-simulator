"""Compare classical two-body propagation with an Earth J2 model."""

import numpy as np
import matplotlib.pyplot as plt

from src.orbital.state import OrbitalState
from src.orbital.elements import compute_orbital_elements
from src.physics.bodies import EARTH
from src.physics.constants import EARTH_J2
from src.physics.dynamics import build_state_derivative
from src.physics.perturbations.j2 import (
    j2_acceleration,
    make_j2_perturbation,
)
from src.mission.mission import Mission


ALTITUDE = 700e3
INCLINATION_DEG = 63.4
DURATION = 24.0 * 3600.0
DT = 30.0


def create_initial_state() -> OrbitalState:
    """Create a circular 700 km orbit with the requested inclination."""

    radius = EARTH.radius + ALTITUDE
    inclination = np.deg2rad(INCLINATION_DEG)
    speed = np.sqrt(EARTH.mu / radius)

    # Start on +x. The velocity lies in the yz-plane, so the orbit
    # has the requested inclination relative to Earth's equator.
    position = np.array([
        radius,
        0.0,
        0.0,
    ])

    velocity = np.array([
        0.0,
        speed * np.cos(inclination),
        speed * np.sin(inclination),
    ])

    return OrbitalState(
        position=position,
        velocity=velocity,
    )


def propagate_two_body(initial_state: OrbitalState):
    mission = Mission(
        initial_state=initial_state,
        body=EARTH,
        dt=DT,
    )

    return mission.coast(
        DURATION,
        label="Two-body propagation",
    )


def propagate_with_j2(initial_state: OrbitalState):
    j2_model = make_j2_perturbation(
        EARTH_J2
    )

    derivative = build_state_derivative(
        body=EARTH,
        perturbations=[
            j2_model,
        ],
    )

    mission = Mission(
        initial_state=initial_state,
        body=EARTH,
        dt=DT,
        derivative=derivative,
    )

    return mission.coast(
        DURATION,
        label="J2 propagation",
    )


def print_report(
    initial_state: OrbitalState,
    classical_result,
    j2_result,
):
    position_difference = (
        j2_result.positions
        - classical_result.positions
    )

    velocity_difference = (
        j2_result.velocities
        - classical_result.velocities
    )

    position_separation = np.linalg.norm(
        position_difference,
        axis=1,
    )

    velocity_separation = np.linalg.norm(
        velocity_difference,
        axis=1,
    )

    initial_j2 = j2_acceleration(
        initial_state,
        EARTH,
        EARTH_J2,
    )

    radius = np.linalg.norm(
        initial_state.position
    )

    central_gravity = (
        EARTH.mu / radius**2
    )

    initial_elements = compute_orbital_elements(
        initial_state,
        EARTH,
    )

    classical_final_state = OrbitalState(
        position=classical_result.positions[-1],
        velocity=classical_result.velocities[-1],
    )

    j2_final_state = OrbitalState(
        position=j2_result.positions[-1],
        velocity=j2_result.velocities[-1],
    )

    classical_elements = compute_orbital_elements(
        classical_final_state,
        EARTH,
    )

    j2_elements = compute_orbital_elements(
        j2_final_state,
        EARTH,
    )

    print()
    print("=" * 72)
    print("TWO-BODY vs J2 — 24 HOUR COMPARISON")
    print("=" * 72)
    print(f"Altitude                  : {ALTITUDE / 1e3:.3f} km")
    print(f"Initial inclination       : {INCLINATION_DEG:.3f} deg")
    print(f"J2                        : {EARTH_J2:.10e}")
    print(f"Propagation time          : {DURATION / 3600:.2f} h")
    print(f"Time step                 : {DT:.1f} s")
    print()
    print(f"Initial |a_J2|            : {np.linalg.norm(initial_j2):.8e} m/s²")
    print(f"Initial |g|               : {central_gravity:.8e} m/s²")
    print(
        "Initial |a_J2| / |g|     : "
        f"{np.linalg.norm(initial_j2) / central_gravity:.8e}"
    )
    print()
    print(f"Final position separation : {position_separation[-1] / 1e3:.3f} km")
    print(f"Maximum separation        : {np.max(position_separation) / 1e3:.3f} km")
    print(f"Final velocity separation : {velocity_separation[-1]:.6f} m/s")
    print()
    print("OSCULATING ELEMENTS")
    print("-" * 72)
    print(
        f"Initial a                  : {initial_elements.semi_major_axis / 1e3:.3f} km"
    )
    print(
        f"Two-body final a           : {classical_elements.semi_major_axis / 1e3:.3f} km"
    )
    print(
        f"J2 final a                 : {j2_elements.semi_major_axis / 1e3:.3f} km"
    )
    print(
        f"Initial e                  : {initial_elements.eccentricity:.8e}"
    )
    print(
        f"Two-body final e           : {classical_elements.eccentricity:.8e}"
    )
    print(
        f"J2 final e                 : {j2_elements.eccentricity:.8e}"
    )
    print(
        f"Initial i                  : {np.rad2deg(initial_elements.inclination):.6f} deg"
    )
    print(
        f"Two-body final i           : {np.rad2deg(classical_elements.inclination):.6f} deg"
    )
    print(
        f"J2 final i                 : {np.rad2deg(j2_elements.inclination):.6f} deg"
    )
    print("=" * 72)

    return position_separation, velocity_separation


def plot_separation(
    times: np.ndarray,
    position_separation: np.ndarray,
):
    time_hours = times / 3600.0

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        time_hours,
        position_separation / 1e3,
    )

    ax.set_title(
        "Position divergence caused by Earth J2"
    )
    ax.set_xlabel(
        "Time [h]"
    )
    ax.set_ylabel(
        "|r(J2) - r(2-body)| [km]"
    )
    ax.grid(True)

    fig.tight_layout()

    return fig, ax


def plot_velocity_separation(
    times: np.ndarray,
    velocity_separation: np.ndarray,
):
    time_hours = times / 3600.0

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        time_hours,
        velocity_separation,
    )

    ax.set_title(
        "Velocity divergence caused by Earth J2"
    )
    ax.set_xlabel(
        "Time [h]"
    )
    ax.set_ylabel(
        "|v(J2) - v(2-body)| [m/s]"
    )
    ax.grid(True)

    fig.tight_layout()

    return fig, ax


def main():
    initial_state = create_initial_state()

    classical_result = propagate_two_body(
        initial_state
    )

    j2_result = propagate_with_j2(
        initial_state
    )

    if not np.allclose(
        classical_result.times,
        j2_result.times,
    ):
        raise RuntimeError(
            "The two propagations do not share the same time grid."
        )

    (
        position_separation,
        velocity_separation,
    ) = print_report(
        initial_state,
        classical_result,
        j2_result,
    )

    plot_separation(
        classical_result.times,
        position_separation,
    )

    plot_velocity_separation(
        classical_result.times,
        velocity_separation,
    )

    plt.show()


if __name__ == "__main__":
    main()
