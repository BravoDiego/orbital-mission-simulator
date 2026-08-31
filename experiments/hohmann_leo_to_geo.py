import numpy as np

from src.physics.bodies import EARTH
from src.physics.constants import EARTH_RADIUS

from src.mission.hohmann import (
    simulate_hohmann_mission,
)

from src.visualization.mission_plot import (
    plot_hohmann_mission,
)


def print_mission_report(mission):
    """
    Print a summary of the Hohmann mission.
    """

    transfer = mission.setup.transfer

    # --------------------------------------------------
    # Arrival data
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
    # Post-burn data
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

    radius_relative_error = (
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
    print("=" * 60)
    print("HOHMANN MISSION REPORT")
    print("=" * 60)

    print()
    print("ORBITS")
    print("-" * 60)

    print(
        f"Initial radius       : "
        f"{transfer.r1 / 1e3:12.3f} km"
    )

    print(
        f"Initial altitude     : "
        f"{(transfer.r1 - EARTH_RADIUS) / 1e3:12.3f} km"
    )

    print(
        f"Final radius         : "
        f"{transfer.r2 / 1e3:12.3f} km"
    )

    print(
        f"Final altitude       : "
        f"{(transfer.r2 - EARTH_RADIUS) / 1e3:12.3f} km"
    )

    print()
    print("MANEUVERS")
    print("-" * 60)

    print(
        f"Burn 1 delta-v       : "
        f"{transfer.delta_v1:12.3f} m/s"
    )

    print(
        f"Burn 2 delta-v       : "
        f"{transfer.delta_v2:12.3f} m/s"
    )

    print(
        f"Total delta-v        : "
        f"{transfer.delta_v_total:12.3f} m/s"
    )

    print()
    print("TRANSFER")
    print("-" * 60)

    print(
        f"Transfer time        : "
        f"{transfer.transfer_time / 3600:12.4f} h"
    )

    print(
        f"Arrival radius RK4   : "
        f"{arrival_radius / 1e3:12.3f} km"
    )

    print(
        f"Target radius        : "
        f"{transfer.r2 / 1e3:12.3f} km"
    )

    print(
        f"Radius error         : "
        f"{radius_error:12.3f} m"
    )

    print(
        f"Relative radius error: "
        f"{radius_relative_error:12.3e}"
    )

    print()
    print("VELOCITY")
    print("-" * 60)

    print(
        f"Arrival speed RK4    : "
        f"{arrival_speed:12.3f} m/s"
    )

    print(
        f"Theoretical speed    : "
        f"{transfer.v_transfer_2:12.3f} m/s"
    )

    print(
        f"Velocity error       : "
        f"{velocity_error:12.6f} m/s"
    )

    print(
        f"After Burn 2         : "
        f"{final_speed:12.3f} m/s"
    )

    print()
    print("=" * 60)


def main():

    # --------------------------------------------------
    # Mission definition
    # --------------------------------------------------

    initial_altitude = 300e3

    r1 = (
        EARTH_RADIUS
        + initial_altitude
    )

    # Geostationary orbital radius
    r2 = 42164e3

    # Numerical integration step
    dt = 10.0

    # --------------------------------------------------
    # Simulation
    # --------------------------------------------------

    mission = simulate_hohmann_mission(
        r1=r1,
        r2=r2,
        body=EARTH,
        dt=dt,
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