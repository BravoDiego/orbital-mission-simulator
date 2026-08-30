"""Regime diagram for tangential launches.

We vary the initial radius r0 and the initial tangential speed v0,
then classify the resulting motion into:
- fall / impact
- elliptical orbit
- circular orbit
- very elliptical orbit
- escape
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Circle

from src.integrators.rk4 import RK4
from src.orbital.elements import orbital_elements
from src.orbital.state import OrbitalState
from src.physics.bodies import EARTH
from src.physics.two_body import state_derivative
from src.simulation.simulator import Simulator


# ==========================================================
# CONFIGURATION
# ==========================================================

ECCENTRICITY_THRESHOLD = 0.25

ALTITUDE_EXAMPLE = 500_000.0   # m
DT = 5.0                       # s

RADIUS_MIN = EARTH.radius + 100_000.0      # 100 km altitude
RADIUS_MAX = EARTH.radius + 40_000_000.0   # 40,000 km altitude

N_RADIUS = 300
N_SPEED = 300

SPEED_MIN = 0.0
SPEED_MAX = 12_000.0   # m/s


# ==========================================================
# REGIME LABELS
# ==========================================================

FALL = 0
ELLIPTIC = 1
CIRCULAR = 2
VERY_ELLIPTIC = 3
ESCAPE = 4


REGIME_NAMES = {
    FALL: "Fall / impact",
    ELLIPTIC: "Elliptical orbit",
    CIRCULAR: "Circular orbit",
    VERY_ELLIPTIC: "Very elliptical orbit",
    ESCAPE: "Escape",
}


# ==========================================================
# BASIC ORBIT SETUP
# ==========================================================

def make_state(radius: float, speed: float) -> OrbitalState:
    """Create a purely tangential initial state."""
    return OrbitalState(
        position=np.array([radius, 0.0, 0.0]),
        velocity=np.array([0.0, speed, 0.0]),
    )


def circular_speed(radius: float) -> float:
    """Return circular speed at radius r."""
    return np.sqrt(EARTH.mu / radius)


def escape_speed(radius: float) -> float:
    """Return escape speed at radius r."""
    return np.sqrt(2.0 * EARTH.mu / radius)


def specific_energy(radius: float, speed: float) -> float:
    """Specific orbital energy."""
    return 0.5 * speed**2 - EARTH.mu / radius


def tangential_eccentricity(radius: float, speed: float) -> float:
    """Eccentricity for a purely tangential launch.

    For r = (r, 0, 0) and v = (0, v, 0), one gets:

        e = |r v^2 / mu - 1|
    """
    return abs(radius * speed**2 / EARTH.mu - 1.0)


# ==========================================================
# REGIME CLASSIFICATION
# ==========================================================

def classify_regime(
    radius: float,
    speed: float,
    e_threshold: float = ECCENTRICITY_THRESHOLD,
) -> int:
    """Classify a tangential launch regime."""

    vc = circular_speed(radius)
    vesc = escape_speed(radius)

    # Radial fall / trivial impact
    if speed <= 1e-12:
        return FALL

    # Close enough to circular
    if abs(speed - vc) / vc < 1e-3:
        return CIRCULAR

    # Escape family: parabolic or hyperbolic
    if speed >= vesc * (1.0 - 1e-6):
        return ESCAPE

    # Bound case: compute orbital elements
    state = make_state(radius, speed)
    elements = orbital_elements(state, EARTH)

    # If the perigee goes below Earth's radius,
    # the orbit intersects Earth -> impact / fall
    if elements.periapsis_radius <= EARTH.radius:
        return FALL

    # Pedagogical split inside the ellipse family
    if elements.eccentricity < e_threshold:
        return ELLIPTIC

    return VERY_ELLIPTIC


# ==========================================================
# DIAGRAM COMPUTATION
# ==========================================================

def compute_regime_grid():
    """Compute the regime map on a (radius, speed) grid."""

    radii = np.linspace(
        RADIUS_MIN,
        RADIUS_MAX,
        N_RADIUS,
    )

    speeds = np.linspace(
        SPEED_MIN,
        SPEED_MAX,
        N_SPEED,
    )

    grid = np.zeros(
        (N_SPEED, N_RADIUS),
        dtype=int,
    )

    for i, speed in enumerate(speeds):
        for j, radius in enumerate(radii):
            grid[i, j] = classify_regime(
                radius,
                speed,
            )

    return radii, speeds, grid


# ==========================================================
# PLOT: REGIME DIAGRAM
# ==========================================================

def plot_regime_diagram(
    radii,
    speeds,
    grid,
):
    """Plot the regime diagram in the (r0, v0) plane."""

    cmap = ListedColormap([
        "#c0392b",  # fall
        "#3498db",  # elliptic
        "#27ae60",  # circular
        "#8e44ad",  # very elliptic
        "#f39c12",  # escape
    ])

    norm = BoundaryNorm(
        boundaries=np.arange(-0.5, 5.5, 1.0),
        ncolors=cmap.N,
    )

    fig, ax = plt.subplots()

    image = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        cmap=cmap,
        norm=norm,
        extent=[
            radii[0] / 1000.0,
            radii[-1] / 1000.0,
            speeds[0] / 1000.0,
            speeds[-1] / 1000.0,
        ],
    )

    ax.set_xlabel("Initial radius r₀ [km]")
    ax.set_ylabel("Initial tangential speed v₀ [km/s]")

    ax.set_title("Orbital regime diagram in the (r₀, v₀) plane")

    colorbar = fig.colorbar(
        image,
        ticks=[0, 1, 2, 3, 4],
    )

    colorbar.ax.set_yticklabels([
        REGIME_NAMES[FALL],
        REGIME_NAMES[ELLIPTIC],
        REGIME_NAMES[CIRCULAR],
        REGIME_NAMES[VERY_ELLIPTIC],
        REGIME_NAMES[ESCAPE],
    ])

    # Overlay theoretical boundary curves
    vc = np.sqrt(EARTH.mu / radii)
    vesc = np.sqrt(2.0 * EARTH.mu / radii)

    ax.plot(
        radii / 1000.0,
        vc / 1000.0,
        linestyle="--",
        linewidth=2,
        label="Circular speed",
    )

    ax.plot(
        radii / 1000.0,
        vesc / 1000.0,
        linestyle="--",
        linewidth=2,
        label="Escape speed",
    )

    ax.grid(True)
    ax.legend()

    return fig, ax


# ==========================================================
# SIMULATION HELPERS
# ==========================================================

def simulate_trajectory(
    radius: float,
    speed: float,
):
    """Simulate one trajectory with RK4."""

    state0 = make_state(radius, speed)

    vc = circular_speed(radius)
    circular_period = (
        2.0
        * np.pi
        * np.sqrt(radius**3 / EARTH.mu)
    )

    # Choose a duration adapted to the regime
    if speed <= 1e-12:
        duration = 0.5 * circular_period

    else:
        energy = specific_energy(radius, speed)

        if energy < 0.0:
            try:
                elements = orbital_elements(state0, EARTH)
                if elements.period is not None:
                    duration = elements.period
                else:
                    duration = 1.5 * circular_period
            except Exception:
                duration = 1.5 * circular_period

        else:
            duration = 1.5 * circular_period

    def derivative(state):
        return state_derivative(
            state,
            EARTH,
        )

    simulator = Simulator(
        integrator=RK4(),
        derivative=derivative,
        dt=DT,
    )

    result = simulator.run(
        initial_state=state0,
        duration=duration,
    )

    return result


def truncate_at_impact(result):
    """Truncate the trajectory at first Earth impact, if any."""
    radii = np.linalg.norm(
        result.positions,
        axis=1,
    )

    impact_indices = np.where(
        radii <= EARTH.radius
    )[0]

    if len(impact_indices) == 0:
        return result

    impact_index = impact_indices[0]

    from src.simulation.result import SimulationResult

    return SimulationResult(
        times=result.times[: impact_index + 1],
        positions=result.positions[: impact_index + 1],
        velocities=result.velocities[: impact_index + 1],
    )


# ==========================================================
# PLOT: EXAMPLE TRAJECTORIES
# ==========================================================

def plot_example_trajectories():
    """Plot representative trajectories for one chosen altitude."""

    radius = EARTH.radius + ALTITUDE_EXAMPLE
    vc = circular_speed(radius)

    examples = [
        ("Fall / impact", 0.95 * vc),
        ("Elliptical orbit", 0.99 * vc),
        ("Circular orbit", 1.00 * vc),
        ("Very elliptical orbit", 1.35 * vc),
        ("Escape", 1.45 * vc),
    ]

    fig, ax = plt.subplots()

    earth = Circle(
        (0.0, 0.0),
        EARTH.radius / 1000.0,
        alpha=0.3,
        label="Earth",
    )
    ax.add_patch(earth)

    for label, speed in examples:
        result = simulate_trajectory(
            radius,
            speed,
        )

        result = truncate_at_impact(result)

        positions_km = result.positions / 1000.0

        ax.plot(
            positions_km[:, 0],
            positions_km[:, 1],
            label=f"{label} ({speed / 1000:.2f} km/s)",
        )

    ax.set_aspect(
        "equal",
        adjustable="box",
    )

    ax.set_xlabel("x [km]")
    ax.set_ylabel("y [km]")

    ax.set_title(
        "Representative trajectories at 500 km altitude"
    )

    ax.grid(True)
    ax.legend()

    return fig, ax


# ==========================================================
# MAIN
# ==========================================================

def main():
    print("Computing regime diagram...")
    radii, speeds, grid = compute_regime_grid()

    print("Plotting regime diagram...")
    plot_regime_diagram(
        radii,
        speeds,
        grid,
    )

    print("Plotting example trajectories...")
    plot_example_trajectories()

    plt.show()


if __name__ == "__main__":
    main()