"""Validate J2 nodal precession against first-order secular theory."""

import numpy as np
import matplotlib.pyplot as plt

from src.orbital.state import OrbitalState
from src.orbital.elements import (
    right_ascension_of_ascending_node,
)
from src.orbital.precession import (
    j2_nodal_precession_rate,
)
from src.physics.bodies import EARTH
from src.physics.constants import EARTH_J2
from src.physics.dynamics import build_state_derivative
from src.physics.perturbations.j2 import make_j2_perturbation
from src.mission.mission import Mission


ALTITUDE = 700e3
INCLINATION_DEG = 63.4
DURATION_DAYS = 3.0
DT = 30.0


def create_initial_state() -> OrbitalState:
    """Create the inclined circular reference orbit."""

    radius = EARTH.radius + ALTITUDE
    inclination = np.deg2rad(INCLINATION_DEG)
    speed = np.sqrt(EARTH.mu / radius)

    return OrbitalState(
        position=np.array([
            radius,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            0.0,
            speed * np.cos(inclination),
            speed * np.sin(inclination),
        ]),
    )


def propagate_two_body(
    initial_state: OrbitalState,
):
    mission = Mission(
        initial_state=initial_state,
        body=EARTH,
        dt=DT,
    )

    return mission.coast(
        DURATION_DAYS * 86400.0,
        label="Two-body propagation",
    )


def propagate_with_j2(
    initial_state: OrbitalState,
):
    derivative = build_state_derivative(
        body=EARTH,
        perturbations=[
            make_j2_perturbation(
                EARTH_J2
            )
        ],
    )

    mission = Mission(
        initial_state=initial_state,
        body=EARTH,
        dt=DT,
        derivative=derivative,
    )

    return mission.coast(
        DURATION_DAYS * 86400.0,
        label="J2 propagation",
    )


def compute_raan_history(
    positions: np.ndarray,
    velocities: np.ndarray,
) -> np.ndarray:
    """Compute and unwrap the RAAN history."""

    values = []

    for position, velocity in zip(
        positions,
        velocities,
    ):
        state = OrbitalState(
            position=position,
            velocity=velocity,
        )

        raan = right_ascension_of_ascending_node(
            state
        )

        if raan is None:
            raise RuntimeError(
                "RAAN became undefined during propagation."
            )

        values.append(
            raan
        )

    return np.unwrap(
        np.asarray(
            values,
            dtype=float,
        )
    )


def main():
    initial_state = create_initial_state()

    two_body_result = propagate_two_body(
        initial_state
    )

    j2_result = propagate_with_j2(
        initial_state
    )

    if not np.allclose(
        two_body_result.times,
        j2_result.times,
    ):
        raise RuntimeError(
            "The propagations do not share the same time grid."
        )

    times = j2_result.times
    time_days = times / 86400.0

    two_body_raan = compute_raan_history(
        two_body_result.positions,
        two_body_result.velocities,
    )

    j2_raan = compute_raan_history(
        j2_result.positions,
        j2_result.velocities,
    )

    theoretical_rate = j2_nodal_precession_rate(
        initial_state,
        EARTH,
        EARTH_J2,
    )

    theoretical_raan = (
        j2_raan[0]
        + theoretical_rate * times
    )

    numerical_rate = np.polyfit(
        times,
        j2_raan,
        deg=1,
    )[0]

    two_body_rate = np.polyfit(
        times,
        two_body_raan,
        deg=1,
    )[0]

    numerical_deg_day = (
        np.rad2deg(numerical_rate)
        * 86400.0
    )

    theoretical_deg_day = (
        np.rad2deg(theoretical_rate)
        * 86400.0
    )

    two_body_deg_day = (
        np.rad2deg(two_body_rate)
        * 86400.0
    )

    relative_error = abs(
        numerical_rate
        - theoretical_rate
    ) / abs(
        theoretical_rate
    )

    print()
    print("=" * 72)
    print("J2 NODAL PRECESSION VALIDATION")
    print("=" * 72)
    print(f"Altitude                  : {ALTITUDE / 1e3:.3f} km")
    print(f"Inclination               : {INCLINATION_DEG:.3f} deg")
    print(f"Propagation duration      : {DURATION_DAYS:.3f} days")
    print(f"Time step                 : {DT:.1f} s")
    print()
    print(f"Two-body RAAN rate        : {two_body_deg_day:.6f} deg/day")
    print(f"Numerical J2 RAAN rate    : {numerical_deg_day:.6f} deg/day")
    print(f"Theoretical J2 RAAN rate  : {theoretical_deg_day:.6f} deg/day")
    print(f"Relative rate error       : {100.0 * relative_error:.4f} %")
    print()
    print(
        "Numerical RAAN change     : "
        f"{np.rad2deg(j2_raan[-1] - j2_raan[0]):.6f} deg"
    )
    print(
        "Theoretical RAAN change   : "
        f"{np.rad2deg(theoretical_raan[-1] - theoretical_raan[0]):.6f} deg"
    )
    print("=" * 72)

    fig, ax = plt.subplots(
        figsize=(10, 6)
    )

    ax.plot(
        time_days,
        np.rad2deg(
            two_body_raan
            - two_body_raan[0]
        ),
        label="Two-body",
    )

    ax.plot(
        time_days,
        np.rad2deg(
            j2_raan
            - j2_raan[0]
        ),
        label="J2 numerical",
    )

    ax.plot(
        time_days,
        np.rad2deg(
            theoretical_raan
            - theoretical_raan[0]
        ),
        linestyle="--",
        label="J2 secular theory",
    )

    ax.set_title(
        "Earth J2 nodal precession"
    )
    ax.set_xlabel(
        "Time [days]"
    )
    ax.set_ylabel(
        "RAAN change ΔΩ [deg]"
    )
    ax.grid(True)
    ax.legend()

    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
