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
    radial_velocity,
    orbital_eccentricity,
    orbital_period_from_state,
)


def create_circular_test_state():

    radius = (
        EARTH_RADIUS
        + 400e3
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


def test_radial_velocity_is_zero_on_circular_orbit():

    state = create_circular_test_state()

    vr = radial_velocity(
        state
    )

    assert vr == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_circular_orbit_eccentricity_is_zero():

    state = create_circular_test_state()

    eccentricity = orbital_eccentricity(
        state,
        EARTH,
    )

    assert eccentricity == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_orbital_period_from_state():

    state = create_circular_test_state()

    radius = np.linalg.norm(
        state.position
    )

    expected_period = (
        2.0
        * np.pi
        * np.sqrt(
            radius**3
            / EARTH.mu
        )
    )

    period = orbital_period_from_state(
        state,
        EARTH,
    )

    assert period == pytest.approx(
        expected_period,
        rel=1e-12,
    )


def test_circular_orbit_has_no_unique_apoapsis():

    state = create_circular_test_state()

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
    )

    with pytest.raises(
        ValueError
    ):
        mission.coast_until_apoapsis()


def test_prograde_burn_creates_elliptical_orbit():

    state = create_circular_test_state()

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
    )

    mission.prograde_burn(
        500.0
    )

    eccentricity = orbital_eccentricity(
        mission.current_state,
        EARTH,
    )

    assert eccentricity > 0.0

    assert eccentricity < 1.0


def test_coast_until_apoapsis():

    state = create_circular_test_state()

    initial_radius = np.linalg.norm(
        state.position
    )

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
    )

    # Starting point becomes the periapsis of
    # a new elliptical orbit.
    mission.prograde_burn(
        500.0,
        label="Raise apoapsis",
    )

    mission.coast_until_apoapsis()

    final_state = (
        mission.current_state
    )

    final_radius = np.linalg.norm(
        final_state.position
    )

    final_radial_velocity = radial_velocity(
        final_state
    )

    assert final_radius > initial_radius

    assert abs(
        final_radial_velocity
    ) < 1.0


def test_detected_apoapsis_matches_theory():

    state = create_circular_test_state()

    periapsis_radius = np.linalg.norm(
        state.position
    )

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
    )

    mission.prograde_burn(
        500.0
    )

    transfer_state = (
        mission.current_state
    )

    r = np.linalg.norm(
        transfer_state.position
    )

    v = np.linalg.norm(
        transfer_state.velocity
    )

    energy = (
        0.5 * v**2
        - EARTH.mu / r
    )

    semi_major_axis = (
        -EARTH.mu
        / (2.0 * energy)
    )

    expected_apoapsis = (
        2.0 * semi_major_axis
        - periapsis_radius
    )

    mission.coast_until_apoapsis()

    numerical_apoapsis = np.linalg.norm(
        mission.current_state.position
    )

    assert numerical_apoapsis == pytest.approx(
        expected_apoapsis,
        rel=1e-5,
    )


def test_coast_from_apoapsis_to_periapsis():

    state = create_circular_test_state()

    original_radius = np.linalg.norm(
        state.position
    )

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
    )

    mission.prograde_burn(
        500.0
    )

    mission.coast_until_apoapsis()

    apoapsis_radius = np.linalg.norm(
        mission.current_state.position
    )

    mission.coast_until_periapsis()

    periapsis_radius = np.linalg.norm(
        mission.current_state.position
    )

    assert apoapsis_radius > original_radius

    assert periapsis_radius == pytest.approx(
        original_radius,
        rel=1e-5,
    )


def test_apsis_coasts_are_recorded():

    state = create_circular_test_state()

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
    )

    mission.prograde_burn(
        500.0,
        label="Departure burn",
    )

    mission.coast_until_apoapsis()

    result = mission.result()

    assert len(
        result.phases
    ) == 2

    assert result.phases[0].kind == "burn"

    assert result.phases[1].kind == "coast"

    assert (
        result.phases[1].label
        == "Coast to apoapsis"
    )