import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import EARTH
from src.physics.two_body import state_derivative
from src.physics.dynamics import build_state_derivative


def create_test_state():
    return OrbitalState(
        position=np.array([
            EARTH.radius + 500e3,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            0.0,
            7600.0,
            0.0,
        ]),
    )


def test_no_perturbations_matches_two_body():
    state = create_test_state()

    classical = state_derivative(
        state,
        EARTH,
    )

    derivative = build_state_derivative(
        EARTH
    )

    composed = derivative(state)

    assert composed.shape == (6,)
    assert np.allclose(
        composed,
        classical,
    )


def test_single_perturbation_is_added():
    state = create_test_state()

    perturbation_vector = np.array([
        0.1,
        -0.2,
        0.3,
    ])

    def fake_perturbation(state, body):
        return perturbation_vector

    classical = state_derivative(
        state,
        EARTH,
    )

    derivative = build_state_derivative(
        body=EARTH,
        perturbations=[
            fake_perturbation,
        ],
    )

    result = derivative(state)

    expected = classical.copy()
    expected[3:] += perturbation_vector

    assert np.allclose(
        result,
        expected,
    )


def test_multiple_perturbations_are_summed():
    state = create_test_state()

    def perturbation_1(state, body):
        return np.array([
            1.0,
            0.0,
            0.0,
        ])

    def perturbation_2(state, body):
        return np.array([
            0.0,
            2.0,
            0.0,
        ])

    derivative = build_state_derivative(
        EARTH,
        perturbations=[
            perturbation_1,
            perturbation_2,
        ],
    )

    classical = state_derivative(
        state,
        EARTH,
    )

    result = derivative(state)

    expected = classical.copy()
    expected[3:] += np.array([
        1.0,
        2.0,
        0.0,
    ])

    assert np.allclose(
        result,
        expected,
    )
