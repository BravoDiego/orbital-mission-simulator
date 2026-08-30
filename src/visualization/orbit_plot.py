"""Orbital trajectory visualization."""

import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from src.physics.bodies import CelestialBody
from src.simulation.result import SimulationResult


def plot_orbit(
    result: SimulationResult,
    body: CelestialBody,
    show: bool = True,
):
    """Plot the XY projection of an orbital trajectory.

    Parameters
    ----------
    result : SimulationResult
        Simulation trajectory.
    body : CelestialBody
        Central celestial body.
    show : bool
        Whether to immediately display the figure.

    Returns
    -------
    tuple
        Matplotlib figure and axes.
    """

    positions_km = (
        result.positions / 1000.0
    )

    body_radius_km = (
        body.radius / 1000.0
    )

    x = positions_km[:, 0]
    y = positions_km[:, 1]

    fig, ax = plt.subplots()

    # Orbital trajectory
    ax.plot(
        x,
        y,
        label="Trajectory",
    )

    # Central body
    body_circle = Circle(
        (0.0, 0.0),
        body_radius_km,
        alpha=0.3,
        label=body.name,
    )

    ax.add_patch(body_circle)

    # Initial and final positions
    ax.scatter(
        x[0],
        y[0],
        marker="o",
        label="Initial position",
    )

    ax.scatter(
        x[-1],
        y[-1],
        marker="x",
        label="Final position",
    )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")

    ax.set_title(
        f"Orbit around {body.name}"
    )

    ax.grid(True)
    ax.legend()

    if show:
        plt.show()

    return fig, ax