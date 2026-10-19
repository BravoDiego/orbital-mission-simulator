import numpy as np

from src.orbital.conversions import (
    state_from_classical_elements,
)
from src.orbital.conservation import (
    specific_orbital_energy,
)
from src.orbital.history import (
    compute_orbital_elements_history,
)
from src.physics.bodies import EARTH
from src.physics.constants import (
    EARTH_J2,
    EARTH_ROTATION_RATE,
)
from src.physics.dynamics import (
    build_state_derivative,
)
from src.physics.two_body import (
    state_derivative,
)
from src.physics.perturbations.j2 import (
    j2_acceleration,
    make_j2_perturbation,
)
from src.physics.perturbations.drag import (
    DragProperties,
    drag_acceleration,
    make_drag_perturbation,
)
from src.mission.mission import Mission


SPACECRAFT = DragProperties(
    mass=100.0,
    cross_section_area=6.0,
    drag_coefficient=2.2,
)


def create_leo_state():
    return state_from_classical_elements(
        semi_major_axis=(
            EARTH.radius
            + 350e3
        ),
        eccentricity=0.005,
        inclination=np.deg2rad(
            51.6
        ),
        raan=np.deg2rad(
            25.0
        ),
        argument_of_periapsis=np.deg2rad(
            40.0
        ),
        true_anomaly=np.deg2rad(
            10.0
        ),
        body=EARTH,
    )


def build_models():
    j2 = make_j2_perturbation(
        EARTH_J2
    )

    drag = make_drag_perturbation(
        properties=SPACECRAFT,
        rotation_rate=(
            EARTH_ROTATION_RATE
        ),
    )

    return j2, drag


def test_combined_derivative_equals_sum_of_perturbations():
    state = create_leo_state()

    j2, drag = build_models()

    derivative = build_state_derivative(
        body=EARTH,
        perturbations=[
            j2,
            drag,
        ],
    )

    result = derivative(
        state
    )

    expected = np.asarray(
        state_derivative(
            state,
            EARTH,
        ),
        dtype=float,
    ).copy()

    expected[3:] += j2_acceleration(
        state,
        EARTH,
        EARTH_J2,
    )

    expected[3:] += drag_acceleration(
        state=state,
        body=EARTH,
        properties=SPACECRAFT,
        rotation_rate=(
            EARTH_ROTATION_RATE
        ),
    )

    assert np.allclose(
        result,
        expected,
    )


def test_drag_dissipation_remains_visible_on_top_of_j2():
    state = create_leo_state()

    j2, drag = build_models()

    j2_derivative = build_state_derivative(
        body=EARTH,
        perturbations=[
            j2
        ],
    )

    combined_derivative = build_state_derivative(
        body=EARTH,
        perturbations=[
            j2,
            drag,
        ],
    )

    j2_mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=60.0,
        derivative=j2_derivative,
    )

    combined_mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=60.0,
        derivative=combined_derivative,
    )

    j2_result = j2_mission.coast(
        1800.0
    )

    combined_result = combined_mission.coast(
        1800.0
    )

    j2_energy = specific_orbital_energy(
        j2_result.final_state,
        EARTH,
    )

    combined_energy = specific_orbital_energy(
        combined_result.final_state,
        EARTH,
    )

    assert combined_energy < j2_energy


def test_combined_model_keeps_j2_nodal_signature():
    state = create_leo_state()

    j2, drag = build_models()

    drag_derivative = build_state_derivative(
        body=EARTH,
        perturbations=[
            drag
        ],
    )

    combined_derivative = build_state_derivative(
        body=EARTH,
        perturbations=[
            j2,
            drag,
        ],
    )

    drag_mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=60.0,
        derivative=drag_derivative,
    )

    combined_mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=60.0,
        derivative=combined_derivative,
    )

    drag_result = drag_mission.coast(
        7200.0
    )

    combined_result = combined_mission.coast(
        7200.0
    )

    drag_history = compute_orbital_elements_history(
        drag_result,
        EARTH,
    )

    combined_history = compute_orbital_elements_history(
        combined_result,
        EARTH,
    )

    drag_raan_change = (
        drag_history.raan_unwrapped[-1]
        - drag_history.raan_unwrapped[0]
    )

    combined_raan_change = (
        combined_history.raan_unwrapped[-1]
        - combined_history.raan_unwrapped[0]
    )

    assert abs(
        combined_raan_change
    ) > 100.0 * abs(
        drag_raan_change
    )

    assert combined_raan_change < 0.0
