import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import EARTH
from src.physics.constants import EARTH_RADIUS

from src.orbital.transfers import (
    circular_velocity,
)

from src.mission.mission import (
    Mission,
    orbital_period_from_state,
)

from src.visualization.generic_mission_plot import (
    plot_mission,
)


def create_initial_state():
    """
    Create a circular orbit at 400 km altitude,
    starting at an arbitrary angular position.
    """

    altitude = 400e3

    radius = (
        EARTH_RADIUS
        + altitude
    )

    angle = np.deg2rad(
        135.0
    )

    speed = circular_velocity(
        radius,
        EARTH.mu,
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


def print_mission_report(
    result,
):

    print()
    print("=" * 65)
    print("GENERIC ORBITAL MISSION")
    print("=" * 65)

    print()

    print(
        f"Mission duration : "
        f"{result.elapsed_time / 3600:.3f} h"
    )

    print(
        f"Number of phases : "
        f"{len(result.phases)}"
    )

    print(
        f"Number of burns  : "
        f"{len(result.burn_phases)}"
    )

    print(
        f"Total delta-v    : "
        f"{result.total_delta_v:.3f} m/s"
    )

    print()
    print("PHASES")
    print("-" * 65)

    for index, phase in enumerate(
        result.phases,
        start=1,
    ):

        print()

        print(
            f"{index}. {phase.label}"
        )

        print(
            f"   Type       : "
            f"{phase.kind}"
        )

        print(
            f"   Start time : "
            f"{phase.start_time:.2f} s"
        )

        print(
            f"   End time   : "
            f"{phase.end_time:.2f} s"
        )

        if phase.maneuver_result is not None:

            print(
                f"   Delta-v    : "
                f"{phase.maneuver_result.delta_v_magnitude:.3f} m/s"
            )

    print()
    print("=" * 65)


def main():

    # --------------------------------------------------
    # Initial state
    # --------------------------------------------------

    initial_state = (
        create_initial_state()
    )

    # --------------------------------------------------
    # Mission
    # --------------------------------------------------

    mission = Mission(
        initial_state=initial_state,
        body=EARTH,
        dt=10.0,
    )

    # --------------------------------------------------
    # Phase 1
    # Initial coast
    # --------------------------------------------------

    mission.coast(
        duration=600.0,
        label="Initial coast",
    )

    # --------------------------------------------------
    # Phase 2
    # Raise apoapsis to 2500 km altitude
    # --------------------------------------------------

    target_apoapsis_radius = (
        EARTH_RADIUS
        + 2500e3
    )

    mission.set_apoapsis_to(
        target_radius=target_apoapsis_radius,
        label="Raise apoapsis to 2500 km",
    )

    # --------------------------------------------------
    # Phase 3
    # Coast automatically to apoapsis
    # --------------------------------------------------

    mission.coast_until_apoapsis(
        label="Coast to apoapsis",
    )

    # --------------------------------------------------
    # Record actual apoapsis
    # --------------------------------------------------

    apoapsis_state = (
        mission.current_state
    )

    apoapsis_radius = np.linalg.norm(
        apoapsis_state.position
    )

    # --------------------------------------------------
    # Phase 4
    # Circularize at apoapsis
    # --------------------------------------------------

    mission.circularize(
        label="Circularization",
    )

    # --------------------------------------------------
    # Phase 5
    # One complete final orbit
    # --------------------------------------------------

    final_period = orbital_period_from_state(
        mission.current_state,
        EARTH,
    )

    mission.coast(
        duration=final_period,
        label="Final circular orbit",
    )

    # --------------------------------------------------
    # Mission result
    # --------------------------------------------------

    result = mission.result()

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print()

    print(
        f"Target apoapsis altitude  : "
        f"{(target_apoapsis_radius - EARTH_RADIUS) / 1e3:.3f} km"
    )

    print(
        f"Detected apoapsis altitude: "
        f"{(apoapsis_radius - EARTH_RADIUS) / 1e3:.3f} km"
    )

    print_mission_report(
        result
    )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    plot_mission(
        mission=result,
        body=EARTH,
    )


if __name__ == "__main__":
    main()