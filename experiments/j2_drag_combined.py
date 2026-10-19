"""Demonstrate simultaneous J2 and atmospheric-drag perturbations in LEO."""

import numpy as np
import matplotlib.pyplot as plt

from src.orbital.conversions import (
    state_from_classical_elements,
)
from src.orbital.conservation import (
    energy_history,
)
from src.orbital.history import (
    compute_orbital_elements_history,
    linear_rate,
    secular_rate,
)
from src.orbital.precession import (
    j2_nodal_precession_rate_from_elements,
)
from src.physics.bodies import EARTH
from src.physics.constants import (
    EARTH_J2,
    EARTH_ROTATION_RATE,
)
from src.physics.dynamics import (
    build_state_derivative,
)
from src.physics.perturbations.j2 import (
    make_j2_perturbation,
)
from src.physics.perturbations.drag import (
    DragProperties,
    make_drag_perturbation,
)
from src.mission.mission import Mission


SEMI_MAJOR_AXIS = (
    EARTH.radius
    + 350e3
)

ECCENTRICITY = 0.005
INCLINATION = np.deg2rad(51.6)
RAAN = np.deg2rad(25.0)
ARGUMENT_OF_PERIAPSIS = np.deg2rad(40.0)
TRUE_ANOMALY = np.deg2rad(10.0)

DURATION_DAYS = 1.0
DT = 120.0

# Deliberately low ballistic coefficient so the drag signature is easy to
# see over one day. This is a demonstration spacecraft, not a lifetime model.
SPACECRAFT = DragProperties(
    mass=100.0,
    cross_section_area=6.0,
    drag_coefficient=2.2,
)


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


def propagate(
    initial_state,
    perturbations,
    label,
):
    derivative = build_state_derivative(
        body=EARTH,
        perturbations=perturbations,
    )

    mission = Mission(
        initial_state=initial_state,
        body=EARTH,
        dt=DT,
        derivative=derivative,
    )

    return mission.coast(
        DURATION_DAYS * 86400.0,
        label=label,
    )


def deg_per_day(rate):
    return float(
        np.rad2deg(rate)
        * 86400.0
    )


def km_per_day(rate):
    return float(
        rate
        * 86400.0
        / 1e3
    )


def instantaneous_nodal_rate(history):
    return np.array([
        j2_nodal_precession_rate_from_elements(
            semi_major_axis=a,
            eccentricity=e,
            inclination=i,
            body=EARTH,
            j2=EARTH_J2,
        )
        for a, e, i in zip(
            history.semi_major_axis,
            history.eccentricity,
            history.inclination,
        )
    ])


def main():
    initial_state = create_initial_state()

    j2_model = make_j2_perturbation(
        EARTH_J2
    )

    drag_model = make_drag_perturbation(
        properties=SPACECRAFT,
        rotation_rate=(
            EARTH_ROTATION_RATE
        ),
    )

    # ==================================================
    # Three controlled experiments
    # ==================================================

    j2_result = propagate(
        initial_state,
        [j2_model],
        "J2 only",
    )

    drag_result = propagate(
        initial_state,
        [drag_model],
        "Drag only",
    )

    combined_result = propagate(
        initial_state,
        [
            j2_model,
            drag_model,
        ],
        "J2 + drag",
    )

    j2_history = compute_orbital_elements_history(
        j2_result,
        EARTH,
    )

    drag_history = compute_orbital_elements_history(
        drag_result,
        EARTH,
    )

    combined_history = compute_orbital_elements_history(
        combined_result,
        EARTH,
    )

    j2_energy = energy_history(
        j2_result,
        EARTH,
    )

    drag_energy = energy_history(
        drag_result,
        EARTH,
    )

    combined_energy = energy_history(
        combined_result,
        EARTH,
    )

    # ==================================================
    # Secular diagnostics
    # ==================================================

    j2_raan_rate = secular_rate(
        j2_history.times,
        j2_history.raan,
    )

    drag_raan_rate = secular_rate(
        drag_history.times,
        drag_history.raan,
    )

    combined_raan_rate = secular_rate(
        combined_history.times,
        combined_history.raan,
    )

    # A raw J2 osculating semi-major axis oscillates. To isolate the
    # dissipative contribution, compare the combined run against J2 alone.
    isolated_drag_a = (
        combined_history.semi_major_axis
        - j2_history.semi_major_axis
    )

    isolated_drag_energy = (
        combined_energy
        - j2_energy
    )

    isolated_drag_a_rate = linear_rate(
        combined_history.times,
        isolated_drag_a,
    )

    initial_theoretical_raan_rate = (
        j2_nodal_precession_rate_from_elements(
            semi_major_axis=(
                combined_history.semi_major_axis[0]
            ),
            eccentricity=(
                combined_history.eccentricity[0]
            ),
            inclination=(
                combined_history.inclination[0]
            ),
            body=EARTH,
            j2=EARTH_J2,
        )
    )

    final_theoretical_raan_rate = (
        j2_nodal_precession_rate_from_elements(
            semi_major_axis=(
                combined_history.semi_major_axis[-1]
            ),
            eccentricity=(
                combined_history.eccentricity[-1]
            ),
            inclination=(
                combined_history.inclination[-1]
            ),
            body=EARTH,
            j2=EARTH_J2,
        )
    )

    # ==================================================
    # Report
    # ==================================================

    print()
    print("=" * 76)
    print("COUPLED LEO PERTURBATIONS — J2 + ATMOSPHERIC DRAG")
    print("=" * 76)

    print(
        f"Initial a altitude         : "
        f"{(SEMI_MAJOR_AXIS - EARTH.radius) / 1e3:.3f} km"
    )

    print(
        f"Initial eccentricity       : "
        f"{ECCENTRICITY:.5f}"
    )

    print(
        f"Inclination                : "
        f"{np.rad2deg(INCLINATION):.3f} deg"
    )

    print(
        f"Propagation duration       : "
        f"{DURATION_DAYS:.3f} day"
    )

    print(
        f"Time step                  : "
        f"{DT:.1f} s"
    )

    print()

    print(
        f"Ballistic coefficient      : "
        f"{SPACECRAFT.ballistic_coefficient:.3f} kg/m^2"
    )

    print()
    print("ISOLATED DRAG SIGNATURE INSIDE THE COMBINED MODEL")
    print("-" * 76)

    print(
        f"Final a(combined)-a(J2)    : "
        f"{isolated_drag_a[-1] / 1e3:.3f} km"
    )

    print(
        f"Fitted isolated da/dt      : "
        f"{km_per_day(isolated_drag_a_rate):.3f} km/day"
    )

    print(
        f"Final energy difference    : "
        f"{isolated_drag_energy[-1] / 1e3:.3f} kJ/kg"
    )

    print()
    print("NODAL PRECESSION")
    print("-" * 76)

    print(
        f"Drag-only RAAN rate        : "
        f"{deg_per_day(drag_raan_rate):.6f} deg/day"
    )

    print(
        f"J2-only RAAN rate          : "
        f"{deg_per_day(j2_raan_rate):.6f} deg/day"
    )

    print(
        f"Combined RAAN rate         : "
        f"{deg_per_day(combined_raan_rate):.6f} deg/day"
    )

    print(
        f"Theory at initial orbit    : "
        f"{deg_per_day(initial_theoretical_raan_rate):.6f} deg/day"
    )

    print(
        f"Theory at final orbit      : "
        f"{deg_per_day(final_theoretical_raan_rate):.6f} deg/day"
    )

    print("=" * 76)

    # ==================================================
    # Plots
    # ==================================================

    time_days = (
        combined_result.times
        / 86400.0
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 13),
        sharex=True,
    )

    # --------------------------------------------------
    # Osculating semi-major axis
    # --------------------------------------------------

    for history, label in (
        (j2_history, "J2 only"),
        (drag_history, "Drag only"),
        (combined_history, "J2 + drag"),
    ):
        axes[0].plot(
            time_days,
            (
                history.semi_major_axis
                - EARTH.radius
            ) / 1e3,
            label=label,
        )

    axes[0].set_ylabel(
        "a - R_E [km]"
    )
    axes[0].set_title(
        "Coupled J2 + Atmospheric Drag"
    )

    # --------------------------------------------------
    # Isolated dissipative signature
    # --------------------------------------------------

    axes[1].plot(
        time_days,
        isolated_drag_a / 1e3,
        label="a(J2 + drag) - a(J2)",
    )

    axes[1].set_ylabel(
        "Isolated Δa [km]"
    )

    # --------------------------------------------------
    # RAAN evolution
    # --------------------------------------------------

    for history, label in (
        (drag_history, "Drag only"),
        (j2_history, "J2 only"),
        (combined_history, "J2 + drag"),
    ):
        axes[2].plot(
            time_days,
            np.rad2deg(
                history.raan_unwrapped
                - history.raan_unwrapped[0]
            ),
            label=label,
        )

    axes[2].set_ylabel(
        "ΔΩ [deg]"
    )

    # --------------------------------------------------
    # Isolated energy loss
    # --------------------------------------------------

    axes[3].plot(
        time_days,
        isolated_drag_energy / 1e3,
        label="epsilon(J2 + drag) - epsilon(J2)",
    )

    axes[3].set_ylabel(
        "Isolated Δenergy [kJ/kg]"
    )

    axes[3].set_xlabel(
        "Time [days]"
    )

    for ax in axes:
        ax.grid(True)
        ax.legend()

    fig.tight_layout()

    # Optional second figure: analytical nodal rate as the orbit decays.
    combined_theoretical_rate = instantaneous_nodal_rate(
        combined_history
    )

    fig2, ax2 = plt.subplots(
        figsize=(10, 5)
    )

    ax2.plot(
        time_days,
        np.rad2deg(
            combined_theoretical_rate
        ) * 86400.0,
    )

    ax2.set_xlabel(
        "Time [days]"
    )

    ax2.set_ylabel(
        "Theoretical dOmega/dt [deg/day]"
    )

    ax2.set_title(
        "Drag Slowly Modifies the J2 Nodal Precession Rate"
    )

    ax2.grid(True)
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
