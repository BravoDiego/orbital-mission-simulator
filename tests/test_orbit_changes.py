import numpy as np
import pytest

from src.orbital.state import OrbitalState
from src.physics.bodies import EARTH
from src.physics.constants import EARTH_RADIUS

from src.orbital.transfers import (
    circular_velocity,
)

from src.mission.mission import (
    Mission,
    orbital_eccentricity,
)

from src.mission.orbit_changes import (
    apsis_radii,
    plan_circularization,
    plan_set_apoapsis,
    plan_set_periapsis,
)


def create_circular_state(
    altitude: float = 400e3,
):

    radius = (
        EARTH_RADIUS
        + altitude
    )

    speed = circular_velocity(
        radius,
        EARTH.mu,
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


def test_apsis_radii_of_circular_orbit():

    state = create_circular_state()

    radius = np.linalg.norm(
        state.position
    )

    periapsis, apoapsis = apsis_radii(
        state,
        EARTH,
    )

    assert periapsis == pytest.approx(
        radius,
        rel=1e-12,
    )

    assert apoapsis == pytest.approx(
        radius,
        rel=1e-12,
    )


def test_plan_set_apoapsis_requires_prograde_burn():

    state = create_circular_state()

    current_radius = np.linalg.norm(
        state.position
    )

    target_radius = (
        EARTH_RADIUS
        + 2000e3
    )

    plan = plan_set_apoapsis(
        state=state,
        target_radius=target_radius,
        body=EARTH,
    )

    delta_v_tangential = np.dot(
        plan.delta_v_vector,
        state.velocity
        / np.linalg.norm(
            state.velocity
        ),
    )

    assert target_radius > current_radius

    assert delta_v_tangential > 0.0


def test_set_apoapsis_reaches_target():

    state = create_circular_state()

    target_radius = (
        EARTH_RADIUS
        + 2000e3
    )

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
    )

    mission.set_apoapsis_to(
        target_radius
    )

    mission.coast_until_apoapsis()

    numerical_radius = np.linalg.norm(
        mission.current_state.position
    )

    assert numerical_radius == pytest.approx(
        target_radius,
        rel=1e-5,
    )


def test_circularize_at_apoapsis():

    state = create_circular_state()

    target_radius = (
        EARTH_RADIUS
        + 2000e3
    )

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
    )

    mission.set_apoapsis_to(
        target_radius
    )

    mission.coast_until_apoapsis()

    eccentricity_before = orbital_eccentricity(
        mission.current_state,
        EARTH,
    )

    mission.circularize()

    eccentricity_after = orbital_eccentricity(
        mission.current_state,
        EARTH,
    )

    assert eccentricity_before > 0.0

    assert eccentricity_after == pytest.approx(
        0.0,
        abs=1e-5,
    )


def test_circularization_speed_matches_theory():

    state = create_circular_state()

    target_radius = (
        EARTH_RADIUS
        + 2000e3
    )

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
    )

    mission.set_apoapsis_to(
        target_radius
    )

    mission.coast_until_apoapsis()

    radius = np.linalg.norm(
        mission.current_state.position
    )

    mission.circularize()

    speed = np.linalg.norm(
        mission.current_state.velocity
    )

    expected_speed = circular_velocity(
        radius,
        EARTH.mu,
    )

    assert speed == pytest.approx(
        expected_speed,
        rel=1e-6,
    )


def test_set_periapsis_reaches_target():

    state = create_circular_state(
        altitude=3000e3
    )

    target_radius = (
        EARTH_RADIUS
        + 500e3
    )

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
    )

    mission.set_periapsis_to(
        target_radius
    )

    mission.coast_until_periapsis()

    numerical_radius = np.linalg.norm(
        mission.current_state.position
    )

    assert numerical_radius == pytest.approx(
        target_radius,
        rel=1e-5,
    )


def test_set_periapsis_is_retrograde():

    state = create_circular_state(
        altitude=3000e3
    )

    target_radius = (
        EARTH_RADIUS
        + 500e3
    )

    plan = plan_set_periapsis(
        state=state,
        target_radius=target_radius,
        body=EARTH,
    )

    velocity_direction = (
        state.velocity
        / np.linalg.norm(
            state.velocity
        )
    )

    tangential_delta_v = np.dot(
        plan.delta_v_vector,
        velocity_direction,
    )

    assert tangential_delta_v < 0.0


def test_high_level_orbit_raise_and_circularization():

    state = create_circular_state()

    target_radius = (
        EARTH_RADIUS
        + 2500e3
    )

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
    )

    mission.set_apoapsis_to(
        target_radius,
        label="Raise apoapsis",
    )

    mission.coast_until_apoapsis()

    mission.circularize(
        label="Circularize",
    )

    result = mission.result()

    final_radius = np.linalg.norm(
        result.final_state.position
    )

    final_eccentricity = orbital_eccentricity(
        result.final_state,
        EARTH,
    )

    assert final_radius == pytest.approx(
        target_radius,
        rel=1e-5,
    )

    assert final_eccentricity == pytest.approx(
        0.0,
        abs=1e-5,
    )

    assert len(
        result.burn_phases
    ) == 2

    assert result.total_delta_v > 0.0