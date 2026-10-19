import numpy as np
import pytest

from src.orbital.state import OrbitalState
from src.orbital.conservation import (
    specific_orbital_energy,
)
from src.orbital.elements import (
    semi_major_axis,
)

from src.physics.bodies import EARTH
from src.physics.constants import (
    EARTH_ROTATION_RATE,
)
from src.physics.dynamics import (
    build_state_derivative,
)
from src.physics.perturbations.drag import (
    DragProperties,
    atmospheric_relative_velocity,
    drag_acceleration,
    make_drag_perturbation,
)

from src.mission.mission import Mission


def create_circular_state(
    altitude=250e3,
):

    radius = (
        EARTH.radius
        + altitude
    )

    speed = np.sqrt(
        EARTH.mu
        / radius
    )

    return OrbitalState(
        position=np.array([
            radius,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            0.0,
            speed,
            0.0,
        ]),
    )


def test_drag_points_opposite_relative_velocity():

    state = create_circular_state()

    properties = DragProperties(
        mass=100.0,
        cross_section_area=1.0,
        drag_coefficient=2.2,
    )

    relative_velocity = (
        atmospheric_relative_velocity(
            state,
            EARTH_ROTATION_RATE,
        )
    )

    acceleration = drag_acceleration(
        state=state,
        body=EARTH,
        properties=properties,
        rotation_rate=EARTH_ROTATION_RATE,
    )

    assert np.dot(
        acceleration,
        relative_velocity,
    ) < 0.0


def test_corotating_spacecraft_has_zero_drag():

    radius = (
        EARTH.radius
        + 250e3
    )

    state = OrbitalState(
        position=np.array([
            radius,
            0.0,
            0.0,
        ]),
        velocity=np.array([
            0.0,
            EARTH_ROTATION_RATE * radius,
            0.0,
        ]),
    )

    properties = DragProperties(
        mass=100.0,
        cross_section_area=1.0,
    )

    acceleration = drag_acceleration(
        state=state,
        body=EARTH,
        properties=properties,
        rotation_rate=EARTH_ROTATION_RATE,
    )

    assert np.allclose(
        acceleration,
        np.zeros(3),
        atol=1e-18,
    )


def test_drag_scales_with_area_to_mass_ratio():

    state = create_circular_state()

    small_ratio = DragProperties(
        mass=200.0,
        cross_section_area=1.0,
    )

    large_ratio = DragProperties(
        mass=100.0,
        cross_section_area=1.0,
    )

    a_small = np.linalg.norm(
        drag_acceleration(
            state,
            EARTH,
            small_ratio,
            rotation_rate=EARTH_ROTATION_RATE,
        )
    )

    a_large = np.linalg.norm(
        drag_acceleration(
            state,
            EARTH,
            large_ratio,
            rotation_rate=EARTH_ROTATION_RATE,
        )
    )

    assert a_large == pytest.approx(
        2.0 * a_small,
        rel=1e-12,
    )


def test_drag_reduces_orbital_energy_and_semi_major_axis():

    state = create_circular_state(
        altitude=250e3
    )

    properties = DragProperties(
        mass=100.0,
        cross_section_area=4.0,
        drag_coefficient=2.2,
    )

    derivative = build_state_derivative(
        body=EARTH,
        perturbations=[
            make_drag_perturbation(
                properties=properties,
                rotation_rate=(
                    EARTH_ROTATION_RATE
                ),
            )
        ],
    )

    initial_energy = (
        specific_orbital_energy(
            state,
            EARTH,
        )
    )

    initial_a = semi_major_axis(
        state,
        EARTH,
    )

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=5.0,
        derivative=derivative,
    )

    mission.coast(
        3600.0
    )

    final_state = (
        mission.current_state
    )

    final_energy = (
        specific_orbital_energy(
            final_state,
            EARTH,
        )
    )

    final_a = semi_major_axis(
        final_state,
        EARTH,
    )

    assert final_energy < initial_energy
    assert final_a < initial_a
