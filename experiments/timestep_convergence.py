"""Study numerical convergence of Euler and RK4.

The experiment propagates a 500 km circular Earth orbit over one
theoretical orbital period using several time steps.

The numerical closure error is then measured as a function of dt.
"""

import matplotlib.pyplot as plt
import numpy as np

from src.integrators.euler import Euler
from src.integrators.rk4 import RK4

from src.orbital.conservation import (
    energy_relative_drift,
)

from src.orbital.state import OrbitalState

from src.physics.bodies import EARTH
from src.physics.two_body import state_derivative

from src.simulation.simulator import Simulator


ALTITUDE = 500_000.0  # m

TIME_STEPS = np.array([
    1.0,
    2.0,
    5.0,
    10.0,
    20.0,
    30.0,
    60.0,
])


def create_initial_state() -> OrbitalState:
    """Return a circular orbit at 500 km altitude."""

    radius = (
        EARTH.radius
        + ALTITUDE
    )

    circular_velocity = np.sqrt(
        EARTH.mu / radius
    )

    return OrbitalState(
        position=np.array([
            radius,
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
    """Return the theoretical Keplerian orbital period."""

    radius = np.linalg.norm(
        state.position
    )

    return (
        2.0
        * np.pi
        * np.sqrt(
            radius**3 / EARTH.mu
        )
    )


def run_simulation(
    integrator,
    initial_state: OrbitalState,
    duration: float,
    dt: float,
):
    """Run one orbital simulation."""

    def derivative(state):
        return state_derivative(
            state,
            EARTH,
        )

    simulator = Simulator(
        integrator=integrator,
        derivative=derivative,
        dt=dt,
    )

    return simulator.run(
        initial_state=initial_state,
        duration=duration,
    )


def compute_errors(
    result,
    initial_state: OrbitalState,
):
    """Return closure and conservation errors."""

    final_state = result.final_state

    position_error = np.linalg.norm(
        final_state.position
        - initial_state.position
    )

    velocity_error = np.linalg.norm(
        final_state.velocity
        - initial_state.velocity
    )

    energy_drift = energy_relative_drift(
        result,
        EARTH,
    )

    max_energy_drift = np.max(
        np.abs(energy_drift)
    )

    return (
        position_error,
        velocity_error,
        max_energy_drift,
    )


def run_convergence_study(
    integrator_class,
    initial_state,
    period,
):
    """Run one integrator for every requested time step."""

    position_errors = []
    velocity_errors = []
    energy_errors = []

    for dt in TIME_STEPS:

        result = run_simulation(
            integrator=integrator_class(),
            initial_state=initial_state,
            duration=period,
            dt=dt,
        )

        (
            position_error,
            velocity_error,
            energy_error,
        ) = compute_errors(
            result,
            initial_state,
        )

        position_errors.append(
            position_error
        )

        velocity_errors.append(
            velocity_error
        )

        energy_errors.append(
            energy_error
        )

    return {
        "position": np.array(
            position_errors
        ),
        "velocity": np.array(
            velocity_errors
        ),
        "energy": np.array(
            energy_errors
        ),
    }


def estimate_order(
    time_steps,
    errors,
):
    """Estimate convergence order from a log-log linear fit."""

    time_steps = np.asarray(
        time_steps,
        dtype=float,
    )

    errors = np.asarray(
        errors,
        dtype=float,
    )

    valid = (
        np.isfinite(errors)
        & (errors > 0)
    )

    log_dt = np.log(
        time_steps[valid]
    )

    log_error = np.log(
        errors[valid]
    )

    slope, intercept = np.polyfit(
        log_dt,
        log_error,
        1,
    )

    return slope, intercept


def print_results(
    euler_errors,
    rk4_errors,
):
    """Print convergence results as a table."""

    print()
    print(
        "NUMERICAL CONVERGENCE STUDY"
    )
    print(
        "=" * 74
    )

    print(
        f"{'dt [s]':>8}"
        f"{'Euler error [km]':>20}"
        f"{'RK4 error [km]':>20}"
        f"{'Euler ΔE/E0':>13}"
        f"{'RK4 ΔE/E0':>13}"
    )

    print(
        "-" * 74
    )

    for i, dt in enumerate(
        TIME_STEPS
    ):

        print(
            f"{dt:8.1f}"
            f"{euler_errors['position'][i] / 1000:20.8e}"
            f"{rk4_errors['position'][i] / 1000:20.8e}"
            f"{euler_errors['energy'][i]:13.3e}"
            f"{rk4_errors['energy'][i]:13.3e}"
        )


def plot_position_convergence(
    euler_errors,
    rk4_errors,
):
    """Plot position error as a function of dt."""

    fig, ax = plt.subplots()

    ax.loglog(
        TIME_STEPS,
        euler_errors["position"],
        marker="o",
        label="Euler",
    )

    ax.loglog(
        TIME_STEPS,
        rk4_errors["position"],
        marker="o",
        label="RK4",
    )

    ax.set_xlabel(
        "Time step dt [s]"
    )

    ax.set_ylabel(
        "Position closure error [m]"
    )

    ax.set_title(
        "Numerical convergence — position error"
    )

    ax.grid(
        True,
        which="both",
    )

    ax.legend()

    return fig, ax


def plot_energy_convergence(
    euler_errors,
    rk4_errors,
):
    """Plot maximum energy drift as a function of dt."""

    fig, ax = plt.subplots()

    ax.loglog(
        TIME_STEPS,
        euler_errors["energy"],
        marker="o",
        label="Euler",
    )

    ax.loglog(
        TIME_STEPS,
        rk4_errors["energy"],
        marker="o",
        label="RK4",
    )

    ax.set_xlabel(
        "Time step dt [s]"
    )

    ax.set_ylabel(
        "Maximum |ΔE / E₀|"
    )

    ax.set_title(
        "Numerical convergence — energy conservation"
    )

    ax.grid(
        True,
        which="both",
    )

    ax.legend()

    return fig, ax


def main():
    initial_state = (
        create_initial_state()
    )

    period = theoretical_period(
        initial_state
    )

    print(
        f"Theoretical orbital period: "
        f"{period:.6f} s "
        f"({period / 60:.6f} min)"
    )

    print(
        f"Altitude: "
        f"{ALTITUDE / 1000:.0f} km"
    )

    print(
        f"Time steps: "
        f"{TIME_STEPS} s"
    )

    # -----------------------------------------
    # Euler
    # -----------------------------------------

    euler_errors = (
        run_convergence_study(
            Euler,
            initial_state,
            period,
        )
    )

    # -----------------------------------------
    # RK4
    # -----------------------------------------

    rk4_errors = (
        run_convergence_study(
            RK4,
            initial_state,
            period,
        )
    )

    print_results(
        euler_errors,
        rk4_errors,
    )

    # -----------------------------------------
    # Experimental convergence orders
    # -----------------------------------------

    euler_order, _ = estimate_order(
        TIME_STEPS,
        euler_errors["position"],
    )

    rk4_order, _ = estimate_order(
        TIME_STEPS,
        rk4_errors["position"],
    )

    print()
    print(
        "ESTIMATED CONVERGENCE ORDER"
    )

    print(
        "---------------------------"
    )

    print(
        f"Euler : p = "
        f"{euler_order:.4f}"
    )

    print(
        f"RK4   : p = "
        f"{rk4_order:.4f}"
    )

    # -----------------------------------------
    # Figures
    # -----------------------------------------

    plot_position_convergence(
        euler_errors,
        rk4_errors,
    )

    plot_energy_convergence(
        euler_errors,
        rk4_errors,
    )

    plt.show()


if __name__ == "__main__":
    main()