# tests/test_transfers.py

import pytest

from src.orbital.transfers import hohmann_transfer
from src.physics.constants import *


def test_leo_to_geo_hohmann():

    r1 = EARTH_RADIUS + 300.0e3
    r2 = 42164.0e3

    transfer = hohmann_transfer(
        r1=r1,
        r2=r2,
        mu=EARTH_MU,
    )

    assert transfer.delta_v1 == pytest.approx(
        2.42573e3,
        rel=1e-4
    )

    assert transfer.delta_v2 == pytest.approx(
        1.46682e3,
        rel=1e-4
    )

    assert transfer.delta_v_total == pytest.approx(
        3.89255e3,
        rel=1e-4
    )

    assert transfer.transfer_time / 3600 == pytest.approx(
        5.27504,
        rel=1e-4
    )