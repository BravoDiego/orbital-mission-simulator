import numpy as np
import pytest
import matplotlib.pyplot as plt

from src.orbital.conversions import (
    state_from_classical_elements,
)
from src.orbital.history import (
    compute_orbital_elements_history,
    secular_rate,
    unwrap_defined_angles,
)
from src.physics.bodies import EARTH
from src.mission.mission import Mission
from src.visualization.orbital_elements_plot import (
    plot_orbital_elements_history,
)


def create_elliptical_state():

    return state_from_classical_elements(
        semi_major_axis=(
            EARTH.radius
            + 1500e3
        ),
        eccentricity=0.05,
        inclination=np.deg2rad(
            45.0
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


def test_unwrap_defined_angles_crosses_zero_cleanly():

    angles = np.deg2rad(
        np.array([
            350.0,
            355.0,
            1.0,
            5.0,
        ])
    )

    result = unwrap_defined_angles(
        angles
    )

    assert np.rad2deg(
        result
    ) == pytest.approx([
        350.0,
        355.0,
        361.0,
        365.0,
    ])


def test_orbital_elements_history_lengths_match_simulation():

    state = create_elliptical_state()

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=30.0,
    )

    result = mission.coast(
        3600.0
    )

    history = compute_orbital_elements_history(
        result,
        EARTH,
    )

    n = len(
        result.times
    )

    assert len(history.times) == n
    assert len(history.semi_major_axis) == n
    assert len(history.eccentricity) == n
    assert len(history.inclination) == n
    assert len(history.raan) == n
    assert len(history.argument_of_periapsis) == n
    assert len(history.true_anomaly) == n


def test_two_body_osculating_elements_remain_constant():

    state = create_elliptical_state()

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=20.0,
    )

    result = mission.coast(
        3.0 * 3600.0
    )

    history = compute_orbital_elements_history(
        result,
        EARTH,
    )

    initial_a = history.semi_major_axis[0]
    initial_e = history.eccentricity[0]
    initial_i = history.inclination[0]

    assert np.max(
        np.abs(
            history.semi_major_axis
            - initial_a
        )
    ) / initial_a < 1e-7

    assert np.max(
        np.abs(
            history.eccentricity
            - initial_e
        )
    ) < 1e-8

    assert np.max(
        np.abs(
            history.inclination
            - initial_i
        )
    ) < 1e-10

    assert abs(
        secular_rate(
            history.times,
            history.raan,
        )
    ) < 1e-10

    assert abs(
        secular_rate(
            history.times,
            history.argument_of_periapsis,
        )
    ) < 1e-10


def test_plot_orbital_elements_history():

    state = create_elliptical_state()

    mission = Mission(
        initial_state=state,
        body=EARTH,
        dt=30.0,
    )

    result = mission.coast(
        600.0
    )

    history = compute_orbital_elements_history(
        result,
        EARTH,
    )

    fig, axes = plot_orbital_elements_history(
        history,
        show=False,
    )

    assert fig is not None
    assert len(axes) == 5

    plt.close(fig)
