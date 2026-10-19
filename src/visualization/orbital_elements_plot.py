"""Visualization of osculating orbital-element histories."""

import numpy as np
import matplotlib.pyplot as plt

from src.orbital.history import OrbitalElementsHistory


def _relative_angle_degrees(
    angles: np.ndarray,
) -> np.ndarray:
    """Convert unwrapped radians to degrees relative to first finite sample."""

    values = np.asarray(
        angles,
        dtype=float,
    )

    result = np.full_like(
        values,
        np.nan,
        dtype=float,
    )

    finite = np.isfinite(
        values
    )

    if not np.any(finite):
        return result

    reference = values[
        np.flatnonzero(finite)[0]
    ]

    result[finite] = np.rad2deg(
        values[finite]
        - reference
    )

    return result


def plot_orbital_elements_history(
    history: OrbitalElementsHistory,
    show: bool = True,
):
    """Plot the main osculating orbital elements of one simulation."""

    time_days = (
        history.times
        / 86400.0
    )

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(11, 14),
        sharex=True,
    )

    axes[0].plot(
        time_days,
        history.semi_major_axis / 1e3,
    )
    axes[0].set_ylabel("a [km]")
    axes[0].set_title(
        "Osculating Orbital Elements"
    )

    axes[1].plot(
        time_days,
        history.eccentricity,
    )
    axes[1].set_ylabel("e")

    axes[2].plot(
        time_days,
        np.rad2deg(
            history.inclination
        ),
    )
    axes[2].set_ylabel("i [deg]")

    axes[3].plot(
        time_days,
        _relative_angle_degrees(
            history.raan_unwrapped
        ),
    )
    axes[3].set_ylabel("ΔΩ [deg]")

    axes[4].plot(
        time_days,
        _relative_angle_degrees(
            history.argument_of_periapsis_unwrapped
        ),
    )
    axes[4].set_ylabel("Δω [deg]")
    axes[4].set_xlabel("Time [days]")

    for ax in axes:
        ax.grid(True)

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes


def plot_orbital_elements_comparison(
    reference: OrbitalElementsHistory,
    perturbed: OrbitalElementsHistory,
    reference_label: str = "Two-body",
    perturbed_label: str = "Perturbed",
    show: bool = True,
):
    """Compare two osculating-element histories on one dashboard."""

    reference_time_days = (
        reference.times
        / 86400.0
    )

    perturbed_time_days = (
        perturbed.times
        / 86400.0
    )

    fig, axes = plt.subplots(
        5,
        1,
        figsize=(11, 14),
        sharex=True,
    )

    axes[0].plot(
        reference_time_days,
        reference.semi_major_axis / 1e3,
        label=reference_label,
    )
    axes[0].plot(
        perturbed_time_days,
        perturbed.semi_major_axis / 1e3,
        label=perturbed_label,
    )
    axes[0].set_ylabel("a [km]")
    axes[0].set_title(
        "Osculating Orbital Elements Comparison"
    )

    axes[1].plot(
        reference_time_days,
        reference.eccentricity,
        label=reference_label,
    )
    axes[1].plot(
        perturbed_time_days,
        perturbed.eccentricity,
        label=perturbed_label,
    )
    axes[1].set_ylabel("e")

    axes[2].plot(
        reference_time_days,
        np.rad2deg(
            reference.inclination
        ),
        label=reference_label,
    )
    axes[2].plot(
        perturbed_time_days,
        np.rad2deg(
            perturbed.inclination
        ),
        label=perturbed_label,
    )
    axes[2].set_ylabel("i [deg]")

    axes[3].plot(
        reference_time_days,
        _relative_angle_degrees(
            reference.raan_unwrapped
        ),
        label=reference_label,
    )
    axes[3].plot(
        perturbed_time_days,
        _relative_angle_degrees(
            perturbed.raan_unwrapped
        ),
        label=perturbed_label,
    )
    axes[3].set_ylabel("ΔΩ [deg]")

    axes[4].plot(
        reference_time_days,
        _relative_angle_degrees(
            reference.argument_of_periapsis_unwrapped
        ),
        label=reference_label,
    )
    axes[4].plot(
        perturbed_time_days,
        _relative_angle_degrees(
            perturbed.argument_of_periapsis_unwrapped
        ),
        label=perturbed_label,
    )
    axes[4].set_ylabel("Δω [deg]")
    axes[4].set_xlabel("Time [days]")

    for ax in axes:
        ax.grid(True)
        ax.legend()

    fig.tight_layout()

    if show:
        plt.show()

    return fig, axes
