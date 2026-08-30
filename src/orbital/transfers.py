from dataclasses import dataclass
from math import sqrt, pi


@dataclass(frozen=True)
class HohmannTransfer:
    r1: float
    r2: float

    semi_major_axis: float

    v_circular_1: float
    v_circular_2: float

    v_transfer_1: float
    v_transfer_2: float

    delta_v1: float
    delta_v2: float
    delta_v_total: float

    transfer_time: float


def circular_velocity(r: float, mu: float) -> float:
    """
    Circular orbital velocity at radius r.

    Parameters
    ----------
    r : float
        Orbital radius.
    mu : float
        Standard gravitational parameter.

    Returns
    -------
    float
        Circular orbital velocity.
    """

    if r <= 0:
        raise ValueError("Orbital radius must be positive.")

    if mu <= 0:
        raise ValueError("mu must be positive.")

    return sqrt(mu / r)


def vis_viva_velocity(r: float, a: float, mu: float) -> float:
    """
    Orbital velocity obtained from the vis-viva equation.
    """

    if r <= 0:
        raise ValueError("Orbital radius must be positive.")

    if a <= 0:
        raise ValueError("Semi-major axis must be positive.")

    if mu <= 0:
        raise ValueError("mu must be positive.")

    return sqrt(
        mu * (2.0 / r - 1.0 / a)
    )


def hohmann_transfer(
    r1: float,
    r2: float,
    mu: float
) -> HohmannTransfer:
    """
    Compute an ideal coplanar Hohmann transfer between
    two circular orbits.

    Radii must be measured from the center of the central body.

    delta_v1 and delta_v2 are signed:
        positive -> prograde burn
        negative -> retrograde burn
    """

    if r1 <= 0 or r2 <= 0:
        raise ValueError("Orbital radii must be positive.")

    if mu <= 0:
        raise ValueError("mu must be positive.")

    # No transfer needed
    if r1 == r2:
        v = circular_velocity(r1, mu)

        return HohmannTransfer(
            r1=r1,
            r2=r2,
            semi_major_axis=r1,
            v_circular_1=v,
            v_circular_2=v,
            v_transfer_1=v,
            v_transfer_2=v,
            delta_v1=0.0,
            delta_v2=0.0,
            delta_v_total=0.0,
            transfer_time=0.0,
        )

    # Transfer ellipse
    a_transfer = (r1 + r2) / 2.0

    # Circular velocities
    v1 = circular_velocity(r1, mu)
    v2 = circular_velocity(r2, mu)

    # Velocities on transfer ellipse
    vt1 = vis_viva_velocity(
        r=r1,
        a=a_transfer,
        mu=mu
    )

    vt2 = vis_viva_velocity(
        r=r2,
        a=a_transfer,
        mu=mu
    )

    # Signed impulsive burns
    delta_v1 = vt1 - v1
    delta_v2 = v2 - vt2

    delta_v_total = (
        abs(delta_v1)
        + abs(delta_v2)
    )

    # Half-period of transfer ellipse
    transfer_time = (
        pi
        * sqrt(a_transfer**3 / mu)
    )

    return HohmannTransfer(
        r1=r1,
        r2=r2,
        semi_major_axis=a_transfer,
        v_circular_1=v1,
        v_circular_2=v2,
        v_transfer_1=vt1,
        v_transfer_2=vt2,
        delta_v1=delta_v1,
        delta_v2=delta_v2,
        delta_v_total=delta_v_total,
        transfer_time=transfer_time,
    )