import matplotlib.pyplot as plt

from src.mission.diagnostics import (
    MissionDiagnostics,
)


def _draw_burn_lines(
    axes,
    diagnostics: MissionDiagnostics,
):
    """
    Draw vertical markers at every burn time.
    """

    burn_indices = diagnostics.burn_indices

    for index in burn_indices:

        burn_time_hours = (
            diagnostics.times[index]
            / 3600.0
        )

        label = (
            diagnostics
            .phase_labels[index]
        )

        for ax in axes:

            ax.axvline(
                burn_time_hours,
                linestyle="--",
                alpha=0.5,
            )

        axes[0].annotate(
            label,
            xy=(
                burn_time_hours,
                diagnostics.radii[index]
                / 1e3,
            ),
            xytext=(5, 5),
            textcoords="offset points",
            rotation=90,
            verticalalignment="bottom",
        )


def plot_mission_diagnostics(
    diagnostics: MissionDiagnostics,
    show: bool = True,
):
    """
    Plot the main orbital diagnostics of a mission.

    Plots:
    - orbital radius
    - speed
    - specific orbital energy
    - semi-major axis
    - eccentricity
    """

    time_hours = (
        diagnostics.times
        / 3600.0
    )

    radius_km = (
        diagnostics.radii
        / 1e3
    )

    speed_km_s = (
        diagnostics.speeds
        / 1e3
    )

    energy_mj_kg = (
        diagnostics.specific_energy
        / 1e6
    )

    semi_major_axis_km = (
        diagnostics.semi_major_axis
        / 1e3
    )

    # --------------------------------------------------
    # Figure
    # --------------------------------------------------

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(11, 14),
        sharex=True,
    )

    # --------------------------------------------------
    # Radius
    # --------------------------------------------------

    axes[0].plot(
        time_hours,
        radius_km,
        linewidth=1.8,
    )

    axes[0].set_ylabel(
        "Radius [km]"
    )

    axes[0].set_title(
        "Orbital Mission Diagnostics"
    )

    # --------------------------------------------------
    # Speed
    # --------------------------------------------------

    axes[1].plot(
        time_hours,
        speed_km_s,
        linewidth=1.8,
    )

    axes[1].set_ylabel(
        "Speed [km/s]"
    )

    # --------------------------------------------------
    # Energy
    # --------------------------------------------------

    axes[2].plot(
        time_hours,
        energy_mj_kg,
        linewidth=1.8,
    )

    axes[2].set_ylabel(
        "Energy [MJ/kg]"
    )

    # --------------------------------------------------
    # Semi-major axis
    # --------------------------------------------------

    axes[3].plot(
        time_hours,
        semi_major_axis_km,
        linewidth=1.8,
    )

    axes[3].set_ylabel(
        "a [km]"
    )

    # --------------------------------------------------
    # Eccentricity
    # --------------------------------------------------

    axes[4].plot(
        time_hours,
        diagnostics.eccentricity,
        linewidth=1.8,
    )

    axes[4].set_ylabel(
        "Eccentricity"
    )

    axes[4].set_xlabel(
        "Mission time [h]"
    )

    # --------------------------------------------------
    # Burn markers
    # --------------------------------------------------

    _draw_burn_lines(
        axes,
        diagnostics,
    )

    # --------------------------------------------------
    # Formatting
    # --------------------------------------------------

    for ax in axes:
        ax.grid(True)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes