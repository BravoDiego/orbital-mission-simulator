"""Compare two-body propagation with a simplified lunar third-body model."""

import numpy as np
import matplotlib.pyplot as plt

from src.orbital.conversions import (
    state_from_classical_elements,
)
from src.orbital.history import (
    compute_orbital_elements_history,
)
from src.physics.bodies import EARTH, MOON
from src.physics.dynamics import (
    build_state_derivative,
)
from src.physics.ephemeris import (
    moon_circular_ephemeris,
)
from src.physics.perturbations.third_body import (
    make_third_body_perturbation,
)
from src.mission.mission import Mission
from src.visualization.orbital_elements_plot import (
    plot_orbital_elements_comparison,
)


SEMI_MAJOR_AXIS = 70_000_000.0
ECCENTRICITY = 0.30
INCLINATION = np.deg2rad(30.0)
RAAN = np.deg2rad(40.0)
ARGUMENT_OF_PERIAPSIS = np.deg2rad(20.0)
TRUE_ANOMALY = 0.0

DURATION_DAYS = 30.0
DT = 300.0


def create_initial_state():
    return state_from_classical_elements(
        semi_major_axis=SEMI_MAJOR_AXIS,
        eccentricity=ECCENTRICITY,
        inclination=INCLINATION,
        raan=RAAN,
        argument_of_periapsis=(
            ARGUMENT_OF_PERIAPSIS
        ),
        true_anomaly=TRUE_ANOMALY,
        body=EARTH,
    )


def main():
    initial_state = create_initial_state()

    moon_ephemeris = moon_circular_ephemeris(
        phase_at_epoch=np.deg2rad(60.0),
    )

    moon_perturbation = (
        make_third_body_perturbation(
            third_body=MOON,
            position_function=moon_ephemeris,
        )
    )

    lunar_derivative = build_state_derivative(
        body=EARTH,
        perturbations=[
            moon_perturbation,
        ],
    )

    duration = (
        DURATION_DAYS
        * 86400.0
    )

    # --------------------------------------------------
    # Two-body reference
    # --------------------------------------------------

    reference_mission = Mission(
        initial_state=initial_state,
        body=EARTH,
        dt=DT,
    )

    reference_result = reference_mission.coast(
        duration,
        label="Two-body propagation",
    )

    # --------------------------------------------------
    # Lunar third-body propagation
    # --------------------------------------------------

    lunar_mission = Mission(
        initial_state=initial_state,
        body=EARTH,
        dt=DT,
        derivative=lunar_derivative,
    )

    lunar_result = lunar_mission.coast(
        duration,
        label="Moon third-body propagation",
    )

    # --------------------------------------------------
    # Element histories
    # --------------------------------------------------

    reference_history = (
        compute_orbital_elements_history(
            reference_result,
            EARTH,
        )
    )

    lunar_history = (
        compute_orbital_elements_history(
            lunar_result,
            EARTH,
        )
    )

    final_position_separation = np.linalg.norm(
        lunar_result.final_state.position
        - reference_result.final_state.position
    )

    final_velocity_separation = np.linalg.norm(
        lunar_result.final_state.velocity
        - reference_result.final_state.velocity
    )

    delta_a = (
        lunar_history.semi_major_axis[-1]
        - reference_history.semi_major_axis[-1]
    )

    delta_e = (
        lunar_history.eccentricity[-1]
        - reference_history.eccentricity[-1]
    )

    delta_i = (
        lunar_history.inclination[-1]
        - reference_history.inclination[-1]
    )

    print()
    print("=" * 76)
    print("LUNAR THIRD-BODY PERTURBATION — SIMPLIFIED CIRCULAR EPHEMERIS")
    print("=" * 76)

    print(
        f"Initial semi-major axis   : "
        f"{SEMI_MAJOR_AXIS / 1e3:.3f} km"
    )

    print(
        f"Initial eccentricity      : "
        f"{ECCENTRICITY:.4f}"
    )

    print(
        f"Initial inclination       : "
        f"{np.rad2deg(INCLINATION):.3f} deg"
    )

    print(
        f"Propagation duration      : "
        f"{DURATION_DAYS:.1f} days"
    )

    print(
        f"Time step                 : "
        f"{DT:.1f} s"
    )

    print()
    print("FINAL DIFFERENCE FROM TWO-BODY REFERENCE")
    print("-" * 76)

    print(
        f"Position separation       : "
        f"{final_position_separation / 1e3:.3f} km"
    )

    print(
        f"Velocity separation       : "
        f"{final_velocity_separation:.6f} m/s"
    )

    print(
        f"Semi-major-axis difference: "
        f"{delta_a / 1e3:.3f} km"
    )

    print(
        f"Eccentricity difference   : "
        f"{delta_e:+.8e}"
    )

    print(
        f"Inclination difference    : "
        f"{np.rad2deg(delta_i):+.6f} deg"
    )

    print("=" * 76)

    # --------------------------------------------------
    # Full orbital-element dashboard
    # --------------------------------------------------

    plot_orbital_elements_comparison(
        reference=reference_history,
        perturbed=lunar_history,
        reference_label="Two-body",
        perturbed_label="Moon third-body",
        show=False,
    )

    # --------------------------------------------------
    # Direct trajectory separation over time
    # --------------------------------------------------

    position_separation = np.linalg.norm(
        lunar_result.positions
        - reference_result.positions,
        axis=1,
    )

    velocity_separation = np.linalg.norm(
        lunar_result.velocities
        - reference_result.velocities,
        axis=1,
    )

    time_days = (
        lunar_result.times
        / 86400.0
    )

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(10, 8),
        sharex=True,
    )

    axes[0].plot(
        time_days,
        position_separation / 1e3,
    )
    axes[0].set_ylabel(
        "|Δr| [km]"
    )
    axes[0].set_title(
        "Accumulated Lunar Third-Body Effect"
    )
    axes[0].grid(True)

    axes[1].plot(
        time_days,
        velocity_separation,
    )
    axes[1].set_ylabel(
        "|Δv| [m/s]"
    )
    axes[1].set_xlabel(
        "Time [days]"
    )
    axes[1].grid(True)

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
