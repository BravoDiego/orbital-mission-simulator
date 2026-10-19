"""Validate J2 apsidal precession and the critical inclination."""

import numpy as np
import matplotlib.pyplot as plt

from src.orbital.state import OrbitalState
from src.orbital.elements import argument_of_periapsis
from src.orbital.conversions import state_from_classical_elements
from src.orbital.precession import (
    j2_apsidal_precession_rate,
    j2_apsidal_precession_rate_from_elements,
    j2_critical_inclinations,
)
from src.physics.bodies import EARTH
from src.physics.constants import EARTH_J2
from src.physics.dynamics import build_state_derivative
from src.physics.perturbations.j2 import make_j2_perturbation
from src.mission.mission import Mission


SEMI_MAJOR_AXIS = (
    EARTH.radius
    + 1500e3
)
ECCENTRICITY = 0.5
RAAN = np.deg2rad(25.0)
ARGUMENT_OF_PERIAPSIS = np.deg2rad(40.0)
TRUE_ANOMALY = np.deg2rad(10.0)
DURATION_DAYS = 3.0
DT = 30.0


def create_initial_state(
    inclination_deg: float,
) -> OrbitalState:
    return state_from_classical_elements(
        semi_major_axis=SEMI_MAJOR_AXIS,
        eccentricity=ECCENTRICITY,
        inclination=np.deg2rad(inclination_deg),
        raan=RAAN,
        argument_of_periapsis=ARGUMENT_OF_PERIAPSIS,
        true_anomaly=TRUE_ANOMALY,
        body=EARTH,
    )


def propagate_omega_history(
    inclination_deg: float,
):
    state = create_initial_state(
        inclination_deg
    )

    derivative = build_state_derivative(
        body=EARTH,
        perturbations=[
            make_j2_perturbation(
                EARTH_J2
            )
        ],
    )

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=DT,
        derivative=derivative,
    )

    result = mission.coast(
        DURATION_DAYS * 86400.0,
        label="J2 propagation",
    )

    omega_history = []

    for position, velocity in zip(
        result.positions,
        result.velocities,
    ):
        sample_state = OrbitalState(
            position=position,
            velocity=velocity,
        )

        omega = argument_of_periapsis(
            sample_state,
            EARTH,
        )

        if omega is None:
            raise RuntimeError(
                "Argument of periapsis became undefined."
            )

        omega_history.append(omega)

    omega_unwrapped = np.unwrap(
        np.asarray(omega_history)
    )

    numerical_rate = np.polyfit(
        result.times,
        omega_unwrapped,
        deg=1,
    )[0]

    return (
        result.times,
        omega_unwrapped,
        numerical_rate,
    )


def radians_per_second_to_degrees_per_day(
    rate: float,
) -> float:
    return float(
        np.rad2deg(rate)
        * 86400.0
    )


def main():
    critical_prograde, critical_retrograde = (
        j2_critical_inclinations()
    )

    critical_deg = np.rad2deg(
        critical_prograde
    )

    validation_inclination = 45.0
    state = create_initial_state(
        validation_inclination
    )

    times, omega, numerical_rate = (
        propagate_omega_history(
            validation_inclination
        )
    )

    theoretical_rate = j2_apsidal_precession_rate(
        state,
        EARTH,
        EARTH_J2,
    )

    relative_error = (
        abs(
            numerical_rate
            - theoretical_rate
        )
        / abs(theoretical_rate)
    )

    theoretical_omega = (
        omega[0]
        + theoretical_rate * times
    )

    print()
    print("=" * 72)
    print("J2 APSIDAL PRECESSION VALIDATION")
    print("=" * 72)
    print(
        f"Semi-major axis           : "
        f"{SEMI_MAJOR_AXIS / 1e3:.3f} km"
    )
    print(
        f"Eccentricity              : "
        f"{ECCENTRICITY:.4f}"
    )
    print(
        f"Validation inclination    : "
        f"{validation_inclination:.3f} deg"
    )
    print(
        f"Propagation duration      : "
        f"{DURATION_DAYS:.3f} days"
    )
    print()
    print(
        f"Numerical apsidal rate    : "
        f"{radians_per_second_to_degrees_per_day(numerical_rate):.6f} deg/day"
    )
    print(
        f"Theoretical apsidal rate  : "
        f"{radians_per_second_to_degrees_per_day(theoretical_rate):.6f} deg/day"
    )
    print(
        f"Relative rate error       : "
        f"{100.0 * relative_error:.4f} %"
    )
    print()
    print(
        f"Critical inclination      : "
        f"{critical_deg:.8f} deg"
    )
    print(
        f"Supplementary critical i  : "
        f"{np.rad2deg(critical_retrograde):.8f} deg"
    )
    print("=" * 72)

    time_days = times / 86400.0
    omega_change_deg = np.rad2deg(
        omega - omega[0]
    )
    theoretical_change_deg = np.rad2deg(
        theoretical_omega
        - theoretical_omega[0]
    )

    fig1, ax1 = plt.subplots(
        figsize=(10, 6)
    )
    ax1.plot(
        time_days,
        omega_change_deg,
        label="J2 numerical",
    )
    ax1.plot(
        time_days,
        theoretical_change_deg,
        linestyle="--",
        label="J2 secular theory",
    )
    ax1.set_xlabel("Time [days]")
    ax1.set_ylabel(
        "Argument of periapsis change Δω [deg]"
    )
    ax1.set_title("J2 Apsidal Precession")
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()

    inclination_deg = np.linspace(
        1.0,
        179.0,
        500,
    )

    rates = np.array([
        j2_apsidal_precession_rate_from_elements(
            semi_major_axis=SEMI_MAJOR_AXIS,
            eccentricity=ECCENTRICITY,
            inclination=np.deg2rad(value),
            body=EARTH,
            j2=EARTH_J2,
        )
        for value in inclination_deg
    ])

    rates_deg_day = (
        np.rad2deg(rates)
        * 86400.0
    )

    fig2, ax2 = plt.subplots(
        figsize=(10, 6)
    )
    ax2.plot(
        inclination_deg,
        rates_deg_day,
        label="First-order J2 theory",
    )
    ax2.axhline(
        0.0,
        linewidth=1.0,
    )
    ax2.axvline(
        np.rad2deg(critical_prograde),
        linestyle="--",
        label="Critical inclinations",
    )
    ax2.axvline(
        np.rad2deg(critical_retrograde),
        linestyle="--",
    )
    ax2.set_xlabel("Inclination [deg]")
    ax2.set_ylabel(
        "Apsidal precession rate dω/dt [deg/day]"
    )
    ax2.set_title("J2 Critical Inclination")
    ax2.grid(True)
    ax2.legend()
    fig2.tight_layout()

    plt.show()


if __name__ == "__main__":
    main()
