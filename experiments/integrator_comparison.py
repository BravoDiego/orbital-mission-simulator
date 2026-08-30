"""Compare Euler and RK4 on a circular Earth orbit."""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle

from src.integrators.euler import Euler
from src.integrators.rk4 import RK4
from src.orbital.conservation import (
    angular_momentum_relative_drift,
    energy_relative_drift,
)
from src.orbital.elements import orbital_elements
from src.orbital.state import OrbitalState
from src.physics.bodies import EARTH
from src.physics.two_body import state_derivative
from src.simulation.simulator import Simulator


ALTITUDE = 500_000.0       # m
DT = 10.0                  # s


def create_initial_state() -> OrbitalState:
    """Create a circular orbit at 500 km altitude."""

    orbital_radius = (
        EARTH.radius
        + ALTITUDE
    )

    circular_velocity = np.sqrt(
        EARTH.mu / orbital_radius
    )

    return OrbitalState(
        position=np.array([
            orbital_radius,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            0.0,
            circular_velocity,
            0.0,
        ]),
    )


def theoretical_period(
    state: OrbitalState,
) -> float:
    """Return the theoretical period of the circular orbit."""

    r = np.linalg.norm(
        state.position
    )

    return (
        2.0
        * np.pi
        * np.sqrt(
            r**3 / EARTH.mu
        )
    )


def run_simulation(
    integrator,
    initial_state: OrbitalState,
    duration: float,
):
    """Run a simulation with the chosen integrator."""

    def derivative(state):
        return state_derivative(
            state,
            EARTH,
        )

    simulator = Simulator(
        integrator=integrator,
        derivative=derivative,
        dt=DT,
    )

    return simulator.run(
        initial_state=initial_state,
        duration=duration,
    )


def compute_metrics(
    result,
    initial_state: OrbitalState,
):
    """Compute quantitative accuracy metrics."""

    energy_drift = energy_relative_drift(
        result,
        EARTH,
    )

    momentum_drift = (
        angular_momentum_relative_drift(
            result
        )
    )

    final_state = result.final_state

    position_error = np.linalg.norm(
        final_state.position
        - initial_state.position
    )

    velocity_error = np.linalg.norm(
        final_state.velocity
        - initial_state.velocity
    )

    elements = orbital_elements(
        final_state,
        EARTH,
    )

    return {
        "position_error": position_error,
        "velocity_error": velocity_error,
        "max_energy_drift": np.max(
            np.abs(energy_drift)
        ),
        "max_momentum_drift": np.max(
            np.abs(momentum_drift)
        ),
        "semi_major_axis": (
            elements.semi_major_axis
        ),
        "eccentricity": (
            elements.eccentricity
        ),
    }


def print_metrics(
    name: str,
    metrics: dict,
):
    """Print accuracy metrics."""

    print()
    print(name)
    print("-" * len(name))

    print(
        f"Position closure error : "
        f"{metrics['position_error'] / 1000:.6f} km"
    )

    print(
        f"Velocity closure error : "
        f"{metrics['velocity_error']:.6e} m/s"
    )

    print(
        f"Maximum |ΔE/E0|        : "
        f"{metrics['max_energy_drift']:.6e}"
    )

    print(
        f"Maximum |Δh/h0|        : "
        f"{metrics['max_momentum_drift']:.6e}"
    )

    print(
        f"Final semi-major axis   : "
        f"{metrics['semi_major_axis'] / 1000:.6f} km"
    )

    print(
        f"Final eccentricity      : "
        f"{metrics['eccentricity']:.6e}"
    )


def plot_trajectories(
    euler_result,
    rk4_result,
):
    """Compare Euler and RK4 trajectories."""

    fig, ax = plt.subplots()

    euler_positions = (
        euler_result.positions
        / 1000.0
    )

    rk4_positions = (
        rk4_result.positions
        / 1000.0
    )

    ax.plot(
        euler_positions[:, 0],
        euler_positions[:, 1],
        label="Euler",
    )

    ax.plot(
        rk4_positions[:, 0],
        rk4_positions[:, 1],
        label="RK4",
    )

    earth = Circle(
        (0.0, 0.0),
        EARTH.radius / 1000.0,
        alpha=0.3,
        label="Earth",
    )

    ax.add_patch(earth)

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")

    ax.set_title(
        "500 km circular orbit — Euler vs RK4"
    )

    ax.grid(True)
    ax.legend()

    return fig, ax


def plot_energy_drift(
    euler_result,
    rk4_result,
):
    """Plot relative orbital-energy drift."""

    euler_error = energy_relative_drift(
        euler_result,
        EARTH,
    )

    rk4_error = energy_relative_drift(
        rk4_result,
        EARTH,
    )

    fig, ax = plt.subplots()

    ax.plot(
        euler_result.times / 60.0,
        euler_error,
        label="Euler",
    )

    ax.plot(
        rk4_result.times / 60.0,
        rk4_error,
        label="RK4",
    )

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("ΔE / |E₀|")

    ax.set_title(
        "Specific orbital energy drift"
    )

    ax.grid(True)
    ax.legend()

    return fig, ax


def plot_momentum_drift(
    euler_result,
    rk4_result,
):
    """Plot relative angular-momentum drift."""

    euler_error = (
        angular_momentum_relative_drift(
            euler_result
        )
    )

    rk4_error = (
        angular_momentum_relative_drift(
            rk4_result
        )
    )

    fig, ax = plt.subplots()

    ax.plot(
        euler_result.times / 60.0,
        euler_error,
        label="Euler",
    )

    ax.plot(
        rk4_result.times / 60.0,
        rk4_error,
        label="RK4",
    )

    ax.set_xlabel("Time [min]")
    ax.set_ylabel("Δh / h₀")

    ax.set_title(
        "Specific angular momentum drift"
    )

    ax.grid(True)
    ax.legend()

    return fig, ax


def main():
    initial_state = create_initial_state()

    period = theoretical_period(
        initial_state
    )

    circular_velocity = np.linalg.norm(
        initial_state.velocity
    )

    print(
        "ORBITAL INTEGRATOR COMPARISON"
    )

    print(
        "============================="
    )

    print(
        f"Altitude             : "
        f"{ALTITUDE / 1000:.0f} km"
    )

    print(
        f"Time step            : "
        f"{DT:.1f} s"
    )

    print(
        f"Circular velocity    : "
        f"{circular_velocity / 1000:.6f} km/s"
    )

    print(
        f"Theoretical period   : "
        f"{period:.3f} s"
    )

    print(
        f"                     : "
        f"{period / 60:.3f} min"
    )

    euler_result = run_simulation(
        Euler(),
        initial_state,
        period,
    )

    rk4_result = run_simulation(
        RK4(),
        initial_state,
        period,
    )

    euler_metrics = compute_metrics(
        euler_result,
        initial_state,
    )

    rk4_metrics = compute_metrics(
        rk4_result,
        initial_state,
    )

    print_metrics(
        "EULER",
        euler_metrics,
    )

    print_metrics(
        "RK4",
        rk4_metrics,
    )

    plot_trajectories(
        euler_result,
        rk4_result,
    )

    plot_energy_drift(
        euler_result,
        rk4_result,
    )

    plot_momentum_drift(
        euler_result,
        rk4_result,
    )

    plt.show()


if __name__ == "__main__":
    main()