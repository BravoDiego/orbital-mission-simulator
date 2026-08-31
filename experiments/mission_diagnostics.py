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

from src.mission.diagnostics import (
    compute_mission_diagnostics,
)

from src.visualization.generic_mission_plot import (
    plot_mission,
)

from src.visualization.mission_diagnostics_plot import (
    plot_mission_diagnostics,
)


def create_initial_state():

    altitude = 400e3

    radius = (
        EARTH_RADIUS
        + altitude
    )

    angle = np.deg2rad(
        45.0
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


def main():

    # --------------------------------------------------
    # Initial orbit
    # --------------------------------------------------

    initial_state = (
        create_initial_state()
    )

    mission = Mission(
        initial_state=initial_state,
        body=EARTH,
        dt=10.0,
    )

    # --------------------------------------------------
    # Initial coast
    # --------------------------------------------------

    mission.coast(
        300.0,
        label="Initial coast",
    )

    # --------------------------------------------------
    # Raise apoapsis
    # --------------------------------------------------

    target_radius = (
        EARTH_RADIUS
        + 2500e3
    )

    mission.set_apoapsis_to(
        target_radius,
        label="Raise apoapsis",
    )

    # --------------------------------------------------
    # Transfer
    # --------------------------------------------------

    mission.coast_until_apoapsis(
        label="Transfer to apoapsis",
    )

    # --------------------------------------------------
    # Circularization
    # --------------------------------------------------

    mission.circularize(
        label="Circularization",
    )

    # --------------------------------------------------
    # Final circular coast
    # --------------------------------------------------

    period = orbital_period_from_state(
        mission.current_state,
        EARTH,
    )

    mission.coast(
        period,
        label="Final circular orbit",
    )

    # --------------------------------------------------
    # Results
    # --------------------------------------------------

    result = mission.result()

    diagnostics = (
        compute_mission_diagnostics(
            result,
            EARTH,
        )
    )

    # --------------------------------------------------
    # Terminal report
    # --------------------------------------------------

    print()
    print("=" * 60)
    print("MISSION DIAGNOSTICS")
    print("=" * 60)

    print(
        f"Samples          : "
        f"{len(diagnostics.times)}"
    )

    print(
        f"Mission duration : "
        f"{result.elapsed_time / 3600:.3f} h"
    )

    print(
        f"Burns            : "
        f"{len(result.burn_phases)}"
    )

    print(
        f"Total delta-v    : "
        f"{result.total_delta_v:.3f} m/s"
    )

    print()

    print(
        f"Initial a        : "
        f"{diagnostics.semi_major_axis[0] / 1e3:.3f} km"
    )

    print(
        f"Final a          : "
        f"{diagnostics.semi_major_axis[-1] / 1e3:.3f} km"
    )

    print(
        f"Initial e        : "
        f"{diagnostics.eccentricity[0]:.6e}"
    )

    print(
        f"Final e          : "
        f"{diagnostics.eccentricity[-1]:.6e}"
    )

    print("=" * 60)

    # --------------------------------------------------
    # Spatial trajectory
    # --------------------------------------------------

    plot_mission(
        mission=result,
        body=EARTH,
    )

    # --------------------------------------------------
    # Scientific diagnostics
    # --------------------------------------------------

    plot_mission_diagnostics(
        diagnostics,
    )


if __name__ == "__main__":
    main()