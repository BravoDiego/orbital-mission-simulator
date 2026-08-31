import matplotlib.pyplot as plt
import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import EARTH
from src.physics.constants import EARTH_RADIUS

from src.orbital.transfers import (
    circular_velocity,
)

from src.mission.mission import (
    Mission,
)

from src.visualization.generic_mission_plot import (
    plot_mission,
)


def create_test_state():

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


def test_plot_generic_mission():

    state = create_test_state()

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=30.0,
    )

    mission.coast(
        300.0,
        label="Initial coast",
    )

    mission.prograde_burn(
        100.0,
        label="Test burn",
    )

    mission.coast(
        300.0,
        label="Final coast",
    )

    result = mission.result()

    fig, ax = plot_mission(
        mission=result,
        body=EARTH,
        show=False,
    )

    assert fig is not None
    assert ax is not None

    plt.close(fig)