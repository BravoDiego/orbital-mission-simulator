import numpy as np
import matplotlib.pyplot as plt

from src.physics.bodies import CelestialBody
from src.mission.mission import MissionResult


def _draw_burn_arrow(
    ax,
    position: np.ndarray,
    delta_v_vector: np.ndarray,
    reference_radius: float,
):
    """
    Draw a visual arrow showing the direction of a burn.

    The arrow length is purely visual and is not proportional
    to the physical delta-v magnitude.
    """

    delta_v_norm = np.linalg.norm(
        delta_v_vector
    )

    if delta_v_norm == 0.0:
        return

    position_km = (
        position[:2]
        / 1e3
    )

    direction = (
        delta_v_vector[:2]
        / delta_v_norm
    )

    arrow_length_km = (
        0.06
        * reference_radius
        / 1e3
    )

    end = (
        position_km
        + arrow_length_km
        * direction
    )

    ax.annotate(
        "",
        xy=end,
        xytext=position_km,
        arrowprops={
            "arrowstyle": "->",
            "linewidth": 2.0,
        },
    )


def _mission_reference_radius(
    mission: MissionResult,
) -> float:
    """
    Find a useful characteristic radius for plotting.
    """

    radii = [
        np.linalg.norm(
            mission.initial_state.position
        ),
        np.linalg.norm(
            mission.final_state.position
        ),
    ]

    for phase in mission.coast_phases:

        phase_radii = np.linalg.norm(
            phase.simulation_result.positions,
            axis=1,
        )

        radii.append(
            np.max(phase_radii)
        )

    for phase in mission.burn_phases:

        radii.append(
            np.linalg.norm(
                phase.state_after.position
            )
        )

    return float(
        max(radii)
    )


def plot_mission(
    mission: MissionResult,
    body: CelestialBody,
    show: bool = True,
):
    """
    Plot a generic orbital mission.

    Every coast phase is drawn from its numerical RK4
    trajectory. Every burn is automatically marked and
    annotated.

    Parameters
    ----------
    mission
        Generic MissionResult.

    body
        Central celestial body.

    show
        Display the figure immediately when True.

    Returns
    -------
    tuple
        matplotlib Figure and Axes.
    """

    fig, ax = plt.subplots(
        figsize=(10, 10)
    )

    reference_radius = (
        _mission_reference_radius(
            mission
        )
    )

    # --------------------------------------------------
    # Central body
    # --------------------------------------------------

    body_circle = plt.Circle(
        (0.0, 0.0),
        body.radius / 1e3,
        alpha=0.5,
        label=body.name,
    )

    ax.add_patch(
        body_circle
    )

    # --------------------------------------------------
    # Coast phases
    # --------------------------------------------------

    for index, phase in enumerate(
        mission.coast_phases,
        start=1,
    ):

        positions = (
            phase
            .simulation_result
            .positions
            / 1e3
        )

        ax.plot(
            positions[:, 0],
            positions[:, 1],
            linewidth=2.0,
            label=phase.label,
        )

    # --------------------------------------------------
    # Burns
    # --------------------------------------------------

    for phase in mission.burn_phases:

        maneuver = (
            phase.maneuver_result
        )

        position = (
            maneuver
            .state_after
            .position
        )

        position_km = (
            position
            / 1e3
        )

        ax.scatter(
            position_km[0],
            position_km[1],
            s=80,
            zorder=5,
        )

        _draw_burn_arrow(
            ax=ax,
            position=position,
            delta_v_vector=(
                maneuver.delta_v_vector
            ),
            reference_radius=reference_radius,
        )

        ax.annotate(
            (
                f"{phase.label}\n"
                f"Δv = "
                f"{maneuver.delta_v_magnitude:.1f} m/s\n"
                f"t = "
                f"{phase.start_time / 60:.1f} min"
            ),
            xy=position_km[:2],
            xytext=(12, 12),
            textcoords="offset points",
        )

    # --------------------------------------------------
    # Initial state
    # --------------------------------------------------

    initial_position = (
        mission
        .initial_state
        .position
        / 1e3
    )

    ax.scatter(
        initial_position[0],
        initial_position[1],
        marker="o",
        s=50,
        zorder=5,
        label="Mission start",
    )

    # --------------------------------------------------
    # Final state
    # --------------------------------------------------

    final_position = (
        mission
        .final_state
        .position
        / 1e3
    )

    ax.scatter(
        final_position[0],
        final_position[1],
        marker="x",
        s=80,
        zorder=5,
        label="Mission end",
    )

    # --------------------------------------------------
    # Mission summary
    # --------------------------------------------------

    info = (
        f"{body.name} mission\n"
        f"Coast phases: "
        f"{len(mission.coast_phases)}\n"
        f"Burns: "
        f"{len(mission.burn_phases)}\n"
        f"Mission time: "
        f"{mission.elapsed_time / 3600:.2f} h\n"
        f"Total Δv: "
        f"{mission.total_delta_v:.1f} m/s"
    )

    ax.text(
        0.02,
        0.98,
        info,
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={
            "boxstyle": "round",
            "alpha": 0.8,
        },
    )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    ax.set_title(
        f"Orbital Mission — {body.name}"
    )

    ax.set_xlabel(
        "x [km]"
    )

    ax.set_ylabel(
        "y [km]"
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.grid(True)

    ax.legend()

    limit = (
        1.15
        * reference_radius
        / 1e3
    )

    ax.set_xlim(
        -limit,
        limit,
    )

    ax.set_ylim(
        -limit,
        limit,
    )

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax