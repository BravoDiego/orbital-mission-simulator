"""Entry point for the Orbital Mission Simulator."""

import numpy as np

from src.integrators.rk4 import RK4
from src.orbital.state import OrbitalState
from src.physics.bodies import EARTH
from src.physics.two_body import state_derivative
from src.simulation.simulator import Simulator
from src.visualization.orbit_plot import plot_orbit


def main():
    altitude = 500_000.0

    orbital_radius = (
        EARTH.radius
        + altitude
    )

    circular_velocity = np.sqrt(
        EARTH.mu / orbital_radius
    )

    initial_state = OrbitalState(
        position=np.array([
            orbital_radius,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            0.0,
            circular_velocity,
            0.0,
        ]),
    )

    derivative = lambda state: state_derivative(
        state,
        EARTH,
    )

    simulator = Simulator(
        integrator=RK4(),
        derivative=derivative,
        dt=10.0,
    )

    result = simulator.run(
        initial_state=initial_state,
        duration=9000.0,
    )
    plot_orbit(
        result,
        EARTH,
    )

    print("Simulation completed")
    print("--------------------")

    print(
        "Samples:",
        len(result.times),
    )

    print(
        "Duration:",
        result.duration,
        "s",
    )

    print(
        "Initial position:",
        result.initial_state.position,
    )

    print(
        "Final position:",
        result.final_state.position,
    )

    print(
        "Final velocity:",
        result.final_state.velocity,
    )

    print(result.times.shape)
    print(result.positions.shape)
    print(result.velocities.shape)


if __name__ == "__main__":
    main()