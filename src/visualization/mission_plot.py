import numpy as np
import matplotlib.pyplot as plt

from src.physics.bodies import CelestialBody
from src.mission.hohmann import HohmannMissionResult


def _draw_burn_arrow(
    ax,
    position: np.ndarray,
    delta_v_vector: np.ndarray,
    reference_radius: float,
):
    """
    Draw a direction arrow for an impulsive maneuver.

    The arrow length is visual only and is not proportional
    to the physical delta-v magnitude.
    """

    position_km = position[:2] / 1e3

    delta_v_norm = np.linalg.norm(delta_v_vector)

    if delta_v_norm == 0.0:
        return

    direction = (
        delta_v_vector[:2]
        / delta_v_norm
    )

    arrow_length = (
        0.08
        * reference_radius
        / 1e3
    )

    end = (
        position_km
        + arrow_length * direction
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


def plot_hohmann_mission(
    mission: HohmannMissionResult,
    body: CelestialBody,
    show: bool = True,
):
    """
    Plot a complete numerical Hohmann mission.

    The transfer trajectory and final orbit come directly
    from the numerical RK4 propagation.

    The initial circular orbit is shown analytically as a
    reference orbit because it is not propagated before burn 1.

    Parameters
    ----------
    mission
        Complete Hohmann mission result.

    body
        Central celestial body.

    show
        If True, display the figure immediately.

    Returns
    -------
    tuple
        matplotlib Figure and Axes.
    """

    transfer = mission.setup.transfer

    r1 = transfer.r1
    r2 = transfer.r2

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------

    fig, ax = plt.subplots(
        figsize=(10, 10)
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

    ax.add_patch(body_circle)

    # --------------------------------------------------
    # Initial circular orbit
    # --------------------------------------------------

    theta = np.linspace(
        0.0,
        2.0 * np.pi,
        500,
    )

    initial_x = (
        r1
        * np.cos(theta)
        / 1e3
    )

    initial_y = (
        r1
        * np.sin(theta)
        / 1e3
    )

    ax.plot(
        initial_x,
        initial_y,
        linestyle="--",
        label="Initial circular orbit",
    )

    # --------------------------------------------------
    # Numerical transfer trajectory
    # --------------------------------------------------

    transfer_positions = (
        mission
        .transfer_result
        .positions
        / 1e3
    )

    ax.plot(
        transfer_positions[:, 0],
        transfer_positions[:, 1],
        linewidth=2.0,
        label="RK4 transfer trajectory",
    )

    # --------------------------------------------------
    # Numerical final orbit
    # --------------------------------------------------

    final_positions = (
        mission
        .final_orbit_result
        .positions
        / 1e3
    )

    ax.plot(
        final_positions[:, 0],
        final_positions[:, 1],
        linewidth=2.0,
        label="Final numerical orbit",
    )

    # --------------------------------------------------
    # Burn positions
    # --------------------------------------------------

    burn1_position = (
        mission
        .setup
        .burn1
        .state_after
        .position
    )

    burn2_position = (
        mission
        .burn2
        .state_after
        .position
    )

    burn1_km = burn1_position / 1e3
    burn2_km = burn2_position / 1e3

    ax.scatter(
        burn1_km[0],
        burn1_km[1],
        s=80,
        zorder=5,
        label="Burn 1",
    )

    ax.scatter(
        burn2_km[0],
        burn2_km[1],
        s=80,
        zorder=5,
        label="Burn 2",
    )

    # --------------------------------------------------
    # Burn arrows
    # --------------------------------------------------

    reference_radius = max(
        r1,
        r2,
    )

    _draw_burn_arrow(
        ax=ax,
        position=burn1_position,
        delta_v_vector=(
            mission
            .setup
            .burn1
            .delta_v_vector
        ),
        reference_radius=reference_radius,
    )

    _draw_burn_arrow(
        ax=ax,
        position=burn2_position,
        delta_v_vector=(
            mission
            .burn2
            .delta_v_vector
        ),
        reference_radius=reference_radius,
    )

    # --------------------------------------------------
    # Burn labels
    # --------------------------------------------------

    dv1 = (
        mission
        .setup
        .burn1
        .delta_v_magnitude
    )

    dv2 = (
        mission
        .burn2
        .delta_v_magnitude
    )

    ax.annotate(
        f"Burn 1\nΔv = {dv1:.1f} m/s",
        xy=burn1_km[:2],
        xytext=(15, 15),
        textcoords="offset points",
    )

    ax.annotate(
        f"Burn 2\nΔv = {dv2:.1f} m/s",
        xy=burn2_km[:2],
        xytext=(15, 15),
        textcoords="offset points",
    )

    # --------------------------------------------------
    # Mission information
    # --------------------------------------------------

    transfer_time_hours = (
        transfer.transfer_time
        / 3600.0
    )

    info = (
        f"{body.name} Hohmann transfer\n"
        f"r₁ = {r1 / 1e3:.1f} km\n"
        f"r₂ = {r2 / 1e3:.1f} km\n"
        f"Δv₁ = {transfer.delta_v1:.1f} m/s\n"
        f"Δv₂ = {transfer.delta_v2:.1f} m/s\n"
        f"Δv total = {transfer.delta_v_total:.1f} m/s\n"
        f"Transfer time = {transfer_time_hours:.2f} h"
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
        f"Hohmann Mission — {body.name}"
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

    # Symmetric limits around the central body
    max_radius_km = (
        1.15
        * max(r1, r2)
        / 1e3
    )

    ax.set_xlim(
        -max_radius_km,
        max_radius_km,
    )

    ax.set_ylim(
        -max_radius_km,
        max_radius_km,
    )

    fig.tight_layout()

    if show:
        plt.show()

    return fig, ax