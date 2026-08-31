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
)

from src.mission.report import (
    summarize_orbital_state,
    build_mission_report,
    format_mission_report,
)


def create_circular_state(
    altitude=400e3,
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


def test_circular_state_summary():

    state = create_circular_state()

    radius = np.linalg.norm(
        state.position
    )

    summary = summarize_orbital_state(
        state,
        EARTH,
    )

    assert summary.radius == pytest.approx(
        radius
    )

    assert summary.semi_major_axis == pytest.approx(
        radius,
        rel=1e-12,
    )

    assert summary.eccentricity == pytest.approx(
        0.0,
        abs=1e-12,
    )

    assert summary.periapsis_radius == pytest.approx(
        radius,
        rel=1e-12,
    )

    assert summary.apoapsis_radius == pytest.approx(
        radius,
        rel=1e-12,
    )


def test_report_counts_phases_and_burns():

    state = create_circular_state()

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
    )

    mission.coast(
        100.0
    )

    mission.prograde_burn(
        100.0
    )

    mission.coast(
        100.0
    )

    result = mission.result()

    report = build_mission_report(
        result,
        EARTH,
    )

    assert report.phase_count == 3

    assert report.burn_count == 1

    assert report.total_delta_v == pytest.approx(
        100.0
    )


def test_report_detects_prograde_burn():

    state = create_circular_state()

    mission = Mission(
        state,
        EARTH,
    )

    mission.prograde_burn(
        100.0,
        label="Raise orbit",
    )

    report = build_mission_report(
        mission.result(),
        EARTH,
    )

    assert len(
        report.burns
    ) == 1

    assert (
        report.burns[0].direction
        == "prograde"
    )


def test_report_detects_retrograde_burn():

    state = create_circular_state()

    mission = Mission(
        state,
        EARTH,
    )

    mission.retrograde_burn(
        100.0
    )

    report = build_mission_report(
        mission.result(),
        EARTH,
    )

    assert (
        report.burns[0].direction
        == "retrograde"
    )


def test_report_validation_passes():

    state = create_circular_state()

    mission = Mission(
        state,
        EARTH,
        dt=10.0,
    )

    mission.prograde_burn(
        200.0
    )

    mission.coast(
        100.0
    )

    report = build_mission_report(
        mission.result(),
        EARTH,
    )

    assert all(
        validation.passed
        for validation
        in report.validations
    )


def test_formatted_report_contains_sections():

    state = create_circular_state()

    mission = Mission(
        state,
        EARTH,
    )

    mission.prograde_burn(
        100.0,
        label="Test burn",
    )

    report = build_mission_report(
        mission.result(),
        EARTH,
    )

    text = format_mission_report(
        report,
        EARTH,
    )

    assert "ORBITAL MISSION REPORT" in text
    assert "INITIAL ORBIT" in text
    assert "FINAL ORBIT" in text
    assert "VALIDATION" in text
    assert "Test burn" in text