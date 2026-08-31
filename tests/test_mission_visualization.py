import matplotlib.pyplot as plt

from src.physics.bodies import EARTH
from src.physics.constants import EARTH_RADIUS

from src.mission.hohmann import (
    simulate_hohmann_mission,
)

from src.visualization.mission_plot import (
    plot_hohmann_mission,
)


def test_plot_hohmann_mission():

    r1 = EARTH_RADIUS + 300e3
    r2 = 42164e3

    mission = simulate_hohmann_mission(
        r1=r1,
        r2=r2,
        body=EARTH,
        dt=30.0,
    )

    fig, ax = plot_hohmann_mission(
        mission=mission,
        body=EARTH,
        show=False,
    )

    assert fig is not None
    assert ax is not None

    plt.close(fig)