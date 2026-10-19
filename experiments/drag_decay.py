"""Compare a two-body orbit with atmospheric-drag orbital decay."""

import numpy as np
import matplotlib.pyplot as plt

from src.orbital.state import OrbitalState
from src.orbital.history import (
    compute_orbital_elements_history,
)
from src.orbital.conservation import (
    energy_history,
)

from src.physics.bodies import EARTH
from src.physics.constants import (
    EARTH_ROTATION_RATE,
)
from src.physics.atmosphere import (
    earth_exponential_density,
)
from src.physics.dynamics import (
    build_state_derivative,
)
from src.physics.perturbations.drag import (
    DragProperties,
    drag_acceleration,
    make_drag_perturbation,
)

from src.mission.mission import Mission


INITIAL_ALTITUDE = 250e3
DURATION_DAYS = 2.0
DT = 20.0

SPACECRAFT = DragProperties(
    mass=200.0,
    cross_section_area=2.0,
    drag_coefficient=2.2,
)


def create_initial_state() -> OrbitalState:

    radius = (
        EARTH.radius
        + INITIAL_ALTITUDE
    )

    speed = np.sqrt(
        EARTH.mu
        / radius
    )

    return OrbitalState(
        position=np.array([
            radius,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            0.0,
            speed,
            0.0,
        ]),
    )


def main():

    initial_state = (
        create_initial_state()
    )

    duration = (
        DURATION_DAYS
        * 86400.0
    )

    # ==================================================
    # Two-body reference
    # ==================================================

    reference_mission = Mission(
        initial_state=initial_state,
        body=EARTH,
        dt=DT,
    )

    reference_result = (
        reference_mission.coast(
            duration,
            label="Two-body reference",
        )
    )

    # ==================================================
    # Atmospheric drag
    # ==================================================

    drag_model = make_drag_perturbation(
        properties=SPACECRAFT,
        rotation_rate=(
            EARTH_ROTATION_RATE
        ),
    )

    drag_derivative = (
        build_state_derivative(
            body=EARTH,
            perturbations=[
                drag_model
            ],
        )
    )

    drag_mission = Mission(
        initial_state=initial_state,
        body=EARTH,
        dt=DT,
        derivative=drag_derivative,
    )

    drag_result = drag_mission.coast(
        duration,
        label="Atmospheric drag",
    )

    # ==================================================
    # Diagnostics
    # ==================================================

    reference_history = (
        compute_orbital_elements_history(
            reference_result,
            EARTH,
        )
    )

    drag_history = (
        compute_orbital_elements_history(
            drag_result,
            EARTH,
        )
    )

    reference_energy = energy_history(
        reference_result,
        EARTH,
    )

    drag_energy = energy_history(
        drag_result,
        EARTH,
    )

    initial_density = (
        earth_exponential_density(
            INITIAL_ALTITUDE
        )
    )

    initial_drag = drag_acceleration(
        state=initial_state,
        body=EARTH,
        properties=SPACECRAFT,
        rotation_rate=(
            EARTH_ROTATION_RATE
        ),
    )

    initial_a = (
        drag_history.semi_major_axis[0]
    )

    final_a = (
        drag_history.semi_major_axis[-1]
    )

    initial_periapsis = (
        drag_history.periapsis_radius[0]
    )

    final_periapsis = (
        drag_history.periapsis_radius[-1]
    )

    # ==================================================
    # Report
    # ==================================================

    print()
    print("=" * 72)
    print("ATMOSPHERIC DRAG ORBITAL DECAY")
    print("=" * 72)

    print(
        f"Initial altitude          : "
        f"{INITIAL_ALTITUDE / 1e3:.3f} km"
    )

    print(
        f"Propagation duration      : "
        f"{DURATION_DAYS:.3f} days"
    )

    print(
        f"Time step                 : "
        f"{DT:.1f} s"
    )

    print()

    print(
        f"Spacecraft mass           : "
        f"{SPACECRAFT.mass:.3f} kg"
    )

    print(
        f"Cross-sectional area      : "
        f"{SPACECRAFT.cross_section_area:.3f} m^2"
    )

    print(
        f"Drag coefficient          : "
        f"{SPACECRAFT.drag_coefficient:.3f}"
    )

    print(
        f"Ballistic coefficient     : "
        f"{SPACECRAFT.ballistic_coefficient:.3f} kg/m^2"
    )

    print()

    print(
        f"Initial density           : "
        f"{initial_density:.8e} kg/m^3"
    )

    print(
        f"Initial |a_drag|          : "
        f"{np.linalg.norm(initial_drag):.8e} m/s^2"
    )

    print()

    print(
        f"Initial semi-major axis   : "
        f"{initial_a / 1e3:.3f} km"
    )

    print(
        f"Final semi-major axis     : "
        f"{final_a / 1e3:.3f} km"
    )

    print(
        f"Semi-major-axis decay     : "
        f"{(final_a - initial_a) / 1e3:.3f} km"
    )

    print(
        f"Periapsis-altitude change : "
        f"{(final_periapsis - initial_periapsis) / 1e3:.3f} km"
    )

    print(
        f"Specific-energy change    : "
        f"{(drag_energy[-1] - drag_energy[0]) / 1e3:.3f} kJ/kg"
    )

    print("=" * 72)

    # ==================================================
    # Plots
    # ==================================================

    time_days = (
        drag_result.times
        / 86400.0
    )

    reference_time_days = (
        reference_result.times
        / 86400.0
    )

    fig, axes = plt.subplots(
        4,
        1,
        figsize=(11, 12),
        sharex=True,
    )

    # Semi-major-axis altitude
    axes[0].plot(
        reference_time_days,
        (
            reference_history.semi_major_axis
            - EARTH.radius
        ) / 1e3,
        label="Two-body",
    )

    axes[0].plot(
        time_days,
        (
            drag_history.semi_major_axis
            - EARTH.radius
        ) / 1e3,
        label="Drag",
    )

    axes[0].set_ylabel(
        "a - R_E [km]"
    )
    axes[0].set_title(
        "Atmospheric Drag Orbital Decay"
    )

    # Eccentricity
    axes[1].plot(
        reference_time_days,
        reference_history.eccentricity,
        label="Two-body",
    )

    axes[1].plot(
        time_days,
        drag_history.eccentricity,
        label="Drag",
    )

    axes[1].set_ylabel("e")

    # Periapsis and apoapsis altitude
    axes[2].plot(
        time_days,
        (
            drag_history.periapsis_radius
            - EARTH.radius
        ) / 1e3,
        label="Periapsis altitude",
    )

    axes[2].plot(
        time_days,
        (
            drag_history.apoapsis_radius
            - EARTH.radius
        ) / 1e3,
        label="Apoapsis altitude",
    )

    axes[2].set_ylabel(
        "Altitude [km]"
    )

    # Specific orbital energy relative to initial value
    axes[3].plot(
        reference_time_days,
        (
            reference_energy
            - reference_energy[0]
        ) / 1e3,
        label="Two-body",
    )

    axes[3].plot(
        time_days,
        (
            drag_energy
            - drag_energy[0]
        ) / 1e3,
        label="Drag",
    )

    axes[3].set_ylabel(
        "Delta energy [kJ/kg]"
    )
    axes[3].set_xlabel(
        "Time [days]"
    )

    for ax in axes:
        ax.grid(True)
        ax.legend()

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
