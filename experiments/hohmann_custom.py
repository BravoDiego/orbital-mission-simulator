import numpy as np

from src.physics.bodies import EARTH
from src.physics.constants import EARTH_RADIUS

from src.mission.hohmann import (
    simulate_hohmann_mission,
)

from src.visualization.mission_plot import (
    plot_hohmann_mission,
)


def print_mission_report(
    mission,
):
    """
    Print a summary of a custom Hohmann transfer mission.
    """

    transfer = mission.setup.transfer

    # --------------------------------------------------
    # Numerical arrival state
    # --------------------------------------------------

    arrival_state = (
        mission
        .transfer_result
        .final_state
    )

    arrival_radius = np.linalg.norm(
        arrival_state.position
    )

    arrival_speed = np.linalg.norm(
        arrival_state.velocity
    )

    # --------------------------------------------------
    # Final state after circularization
    # --------------------------------------------------

    final_speed = np.linalg.norm(
        mission
        .burn2
        .state_after
        .velocity
    )

    # --------------------------------------------------
    # Errors
    # --------------------------------------------------

    radius_error = (
        arrival_radius
        - transfer.r2
    )

    relative_radius_error = (
        abs(radius_error)
        / transfer.r2
    )

    velocity_error = (
        arrival_speed
        - transfer.v_transfer_2
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print()
    print("=" * 65)
    print("CUSTOM HOHMANN MISSION")
    print("=" * 65)

    print()
    print("ORBITS")
    print("-" * 65)

    print(
        f"Initial radius         : "
        f"{transfer.r1 / 1e3:12.3f} km"
    )

    print(
        f"Initial altitude       : "
        f"{(transfer.r1 - EARTH_RADIUS) / 1e3:12.3f} km"
    )

    print(
        f"Final radius           : "
        f"{transfer.r2 / 1e3:12.3f} km"
    )

    print(
        f"Final altitude         : "
        f"{(transfer.r2 - EARTH_RADIUS) / 1e3:12.3f} km"
    )

    print()
    print("MANEUVERS")
    print("-" * 65)

    print(
        f"Burn 1 signed delta-v  : "
        f"{transfer.delta_v1:12.3f} m/s"
    )

    print(
        f"Burn 2 signed delta-v  : "
        f"{transfer.delta_v2:12.3f} m/s"
    )

    print(
        f"Total delta-v          : "
        f"{transfer.delta_v_total:12.3f} m/s"
    )

    print()
    print("TRANSFER")
    print("-" * 65)

    print(
        f"Transfer time          : "
        f"{transfer.transfer_time / 3600:12.4f} h"
    )

    print(
        f"Numerical arrival      : "
        f"{arrival_radius / 1e3:12.3f} km"
    )

    print(
        f"Target radius          : "
        f"{transfer.r2 / 1e3:12.3f} km"
    )

    print(
        f"Radius error           : "
        f"{radius_error:12.3f} m"
    )

    print(
        f"Relative radius error  : "
        f"{relative_radius_error:12.3e}"
    )

    print()
    print("VELOCITY")
    print("-" * 65)

    print(
        f"Arrival speed RK4      : "
        f"{arrival_speed:12.3f} m/s"
    )

    print(
        f"Theoretical speed      : "
        f"{transfer.v_transfer_2:12.3f} m/s"
    )

    print(
        f"Velocity error         : "
        f"{velocity_error:12.6f} m/s"
    )

    print(
        f"Speed after Burn 2     : "
        f"{final_speed:12.3f} m/s"
    )

    print()
    print("=" * 65)


def get_orbital_radius(
    name: str,
) -> float:
    """
    Ask the user for an orbital radius in km
    and return it in meters.
    """

    while True:

        try:

            radius_km = float(
                input(
                    f"{name} [km from Earth's center]: "
                )
            )

        except ValueError:

            print(
                "Please enter a valid number."
            )

            continue

        radius_m = radius_km * 1e3

        if radius_m <= EARTH_RADIUS:

            print()
            print(
                "Invalid orbit: this radius is "
                "inside the Earth."
            )

            print(
                f"Earth radius = "
                f"{EARTH_RADIUS / 1e3:.3f} km"
            )

            print()

            continue

        return radius_m


def main():

    print()
    print("=" * 65)
    print("HOHMANN TRANSFER SIMULATOR")
    print("=" * 65)

    print()
    print(
        "Enter orbital radii measured from "
        "the center of the Earth."
    )

    print(
        f"Earth radius: "
        f"{EARTH_RADIUS / 1e3:.3f} km"
    )

    print()

    # --------------------------------------------------
    # User-defined orbits
    # --------------------------------------------------

    r1 = get_orbital_radius(
        "Initial radius r1"
    )

    r2 = get_orbital_radius(
        "Final radius r2"
    )

    initial_angle_deg = float(
        input(
            "Initial orbital angle [deg]: "
        )
    )

    initial_angle = np.deg2rad(
        initial_angle_deg
    )

    if np.isclose(
        r1,
        r2,
    ):
        print()
        print(
            "r1 and r2 are identical."
        )

        print(
            "No Hohmann transfer is required."
        )

        return

    # --------------------------------------------------
    # Numerical settings
    # --------------------------------------------------

    dt = 10.0

    # --------------------------------------------------
    # Mission simulation
    # --------------------------------------------------

    mission = simulate_hohmann_mission(
        r1=r1,
        r2=r2,
        body=EARTH,
        dt=dt,
        initial_angle=initial_angle,
    )

    # --------------------------------------------------
    # Report
    # --------------------------------------------------

    print_mission_report(
        mission
    )

    # --------------------------------------------------
    # Visualization
    # --------------------------------------------------

    plot_hohmann_mission(
        mission=mission,
        body=EARTH,
    )


if __name__ == "__main__":
    main()