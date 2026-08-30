import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from src.physics.bodies import EARTH
from src.simulation.result import SimulationResult
from src.visualization.orbit_plot import plot_orbit


def test_plot_orbit():
    result = SimulationResult(
        times=np.array([
            0.0,
            1.0,
        ]),
        positions=np.array([
            [
                EARTH.radius + 500_000.0,
                0.0,
                0.0,
            ],
            [
                EARTH.radius + 499_000.0,
                100_000.0,
                0.0,
            ],
        ]),
        velocities=np.zeros(
            (2, 3)
        ),
    )

    fig, ax = plot_orbit(
        result,
        EARTH,
        show=False,
    )

    assert fig is not None
    assert ax is not None

    plt.close(fig)