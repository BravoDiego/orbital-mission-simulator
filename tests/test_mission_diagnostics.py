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

from src.mission.diagnostics import (
    compute_mission_diagnostics,
)

import matplotlib.pyplot as plt

from src.visualization.mission_diagnostics_plot import (
    plot_mission_diagnostics,
)


def create_circular_state():

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


def test_diagnostics_arrays_have_same_length():

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

    diagnostics = compute_mission_diagnostics(
        mission.result(),
        EARTH,
    )

    n = len(
        diagnostics.times
    )

    assert len(
        diagnostics.positions
    ) == n

    assert len(
        diagnostics.velocities
    ) == n

    assert len(
        diagnostics.radii
    ) == n

    assert len(
        diagnostics.speeds
    ) == n

    assert len(
        diagnostics.specific_energy
    ) == n

    assert len(
        diagnostics.semi_major_axis
    ) == n

    assert len(
        diagnostics.eccentricity
    ) == n


def test_initial_circular_orbit_diagnostics():

    state = create_circular_state()

    radius = np.linalg.norm(
        state.position
    )

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
    )

    mission.coast(
        100.0
    )

    diagnostics = compute_mission_diagnostics(
        mission.result(),
        EARTH,
    )

    assert diagnostics.radii[0] == pytest.approx(
        radius,
        rel=1e-12,
    )

    assert diagnostics.semi_major_axis[0] == pytest.approx(
        radius,
        rel=1e-12,
    )

    assert diagnostics.eccentricity[0] == pytest.approx(
        0.0,
        abs=1e-12,
    )


def test_burn_creates_duplicate_time():

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
        100.0,
        label="Test burn",
    )

    diagnostics = compute_mission_diagnostics(
        mission.result(),
        EARTH,
    )

    indices = np.where(
        np.isclose(
            diagnostics.times,
            100.0,
        )
    )[0]

    # Before and after the instantaneous burn
    assert len(indices) >= 2


def test_prograde_burn_increases_energy():

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
        100.0,
        label="Energy burn",
    )

    diagnostics = compute_mission_diagnostics(
        mission.result(),
        EARTH,
    )

    burn_index = (
        diagnostics.burn_indices[0]
    )

    energy_before = (
        diagnostics
        .specific_energy[
            burn_index - 1
        ]
    )

    energy_after = (
        diagnostics
        .specific_energy[
            burn_index
        ]
    )

    assert energy_after > energy_before


def test_prograde_burn_increases_semi_major_axis():

    state = create_circular_state()

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=10.0,
    )

    mission.prograde_burn(
        200.0
    )

    diagnostics = compute_mission_diagnostics(
        mission.result(),
        EARTH,
    )

    burn_index = (
        diagnostics.burn_indices[0]
    )

    a_before = (
        diagnostics
        .semi_major_axis[
            burn_index - 1
        ]
    )

    a_after = (
        diagnostics
        .semi_major_axis[
            burn_index
        ]
    )

    assert a_after > a_before


def test_orbit_raise_then_circularization():

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
    )

    mission.coast_until_apoapsis()

    mission.circularize()

    diagnostics = compute_mission_diagnostics(
        mission.result(),
        EARTH,
    )

    final_eccentricity = (
        diagnostics
        .eccentricity[-1]
    )

    final_a = (
        diagnostics
        .semi_major_axis[-1]
    )

    assert final_eccentricity == pytest.approx(
        0.0,
        abs=1e-5,
    )

    assert final_a == pytest.approx(
        target_radius,
        rel=1e-5,
    )

def test_plot_mission_diagnostics():

    state = create_circular_state()

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=30.0,
    )

    mission.prograde_burn(
        200.0
    )

    mission.coast_until_apoapsis()

    diagnostics = compute_mission_diagnostics(
        mission.result(),
        EARTH,
    )

    fig, axes = plot_mission_diagnostics(
        diagnostics,
        show=False,
    )

    assert fig is not None

    assert len(
        axes
    ) == 5

    plt.close(
        fig
    )