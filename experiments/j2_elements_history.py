"""Compare complete osculating-element histories with and without J2."""

import numpy as np
import matplotlib.pyplot as plt

from src.orbital.conversions import (
    state_from_classical_elements,
)
from src.orbital.history import (
    compute_orbital_elements_history,
    secular_rate,
)
from src.orbital.precession import (
    j2_nodal_precession_rate,
    j2_apsidal_precession_rate,
)
from src.physics.bodies import EARTH
from src.physics.constants import EARTH_J2
from src.physics.dynamics import build_state_derivative
from src.physics.perturbations.j2 import make_j2_perturbation
from src.mission.mission import Mission
from src.visualization.orbital_elements_plot import (
    plot_orbital_elements_comparison,
)


SEMI_MAJOR_AXIS = EARTH.radius + 1500e3
ECCENTRICITY = 0.05
INCLINATION = np.deg2rad(45.0)
RAAN = np.deg2rad(25.0)
ARGUMENT_OF_PERIAPSIS = np.deg2rad(40.0)
TRUE_ANOMALY = np.deg2rad(10.0)

DURATION_DAYS = 3.0
DT = 30.0


def create_initial_state():

    return state_from_classical_elements(
        semi_major_axis=SEMI_MAJOR_AXIS,
        eccentricity=ECCENTRICITY,
        inclination=INCLINATION,
        raan=RAAN,
        argument_of_periapsis=ARGUMENT_OF_PERIAPSIS,
        true_anomaly=TRUE_ANOMALY,
        body=EARTH,
    )


def propagate_two_body(initial_state):

    mission = Mission(
        initial_state=initial_state,
        body=EARTH,
        dt=DT,
    )

    return mission.coast(
        DURATION_DAYS * 86400.0,
        label="Two-body propagation",
    )


def propagate_j2(initial_state):

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


def to_deg_per_day(rate):

    return float(
        np.rad2deg(rate)
        * 86400.0
    )


def main():

    initial_state = create_initial_state()

    two_body_result = propagate_two_body(
        initial_state
    )

    j2_result = propagate_j2(
        initial_state
    )

    two_body_history = (
        compute_orbital_elements_history(
            two_body_result,
            EARTH,
        )
    )

    j2_history = (
        compute_orbital_elements_history(
            j2_result,
            EARTH,
        )
    )

    numerical_raan_rate = secular_rate(
        j2_history.times,
        j2_history.raan,
    )

    numerical_omega_rate = secular_rate(
        j2_history.times,
        j2_history.argument_of_periapsis,
    )

    theoretical_raan_rate = (
        j2_nodal_precession_rate(
            initial_state,
            EARTH,
            EARTH_J2,
        )
    )

    theoretical_omega_rate = (
        j2_apsidal_precession_rate(
            initial_state,
            EARTH,
            EARTH_J2,
        )
    )

    a_variation = (
        np.max(j2_history.semi_major_axis)
        - np.min(j2_history.semi_major_axis)
    )

    e_variation = (
        np.max(j2_history.eccentricity)
        - np.min(j2_history.eccentricity)
    )

    i_variation_deg = np.rad2deg(
        np.max(j2_history.inclination)
        - np.min(j2_history.inclination)
    )

    print()
    print("=" * 72)
    print("J2 OSCULATING ELEMENTS HISTORY")
    print("=" * 72)
    print(
        f"Propagation duration      : {DURATION_DAYS:.3f} days"
    )
    print(
        f"Time step                 : {DT:.1f} s"
    )
    print()
    print(
        f"a peak-to-peak variation  : {a_variation:.3f} m"
    )
    print(
        f"e peak-to-peak variation  : {e_variation:.8e}"
    )
    print(
        f"i peak-to-peak variation  : {i_variation_deg:.6f} deg"
    )
    print()
    print(
        f"RAAN numerical rate       : {to_deg_per_day(numerical_raan_rate):.6f} deg/day"
    )
    print(
        f"RAAN theoretical rate     : {to_deg_per_day(theoretical_raan_rate):.6f} deg/day"
    )
    print(
        f"omega numerical rate      : {to_deg_per_day(numerical_omega_rate):.6f} deg/day"
    )
    print(
        f"omega theoretical rate    : {to_deg_per_day(theoretical_omega_rate):.6f} deg/day"
    )
    print("=" * 72)

    plot_orbital_elements_comparison(
        reference=two_body_history,
        perturbed=j2_history,
        reference_label="Two-body",
        perturbed_label="J2",
        show=False,
    )

    plt.show()


if __name__ == "__main__":
    main()
