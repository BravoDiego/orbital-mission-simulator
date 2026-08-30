"""Entry point for the Orbital Mission Simulator."""

import numpy as np

from src.integrators.euler import Euler
from src.integrators.rk4 import RK4
from src.orbital.state import OrbitalState
from src.physics.bodies import EARTH
from src.physics.two_body import state_derivative


def main():
    altitude = 500_000.0

    r = EARTH.radius + altitude

    circular_velocity = np.sqrt(
        EARTH.mu / r
    )

    initial_state = OrbitalState(
        position=np.array([
            r,
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

    dt = 10.0

    euler = Euler()
    rk4 = RK4()

    state_euler = euler.step(
        initial_state,
        dt,
        derivative,
    )

    state_rk4 = rk4.step(
        initial_state,
        dt,
        derivative,
    )

    print("Initial position:")
    print(initial_state.position)

    print("\nEuler after 10 s:")
    print(state_euler.position)
    print(state_euler.velocity)

    print("\nRK4 after 10 s:")
    print(state_rk4.position)
    print(state_rk4.velocity)


if __name__ == "__main__":
    main()