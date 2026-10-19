import numpy as np


# Piecewise exponential reference atmosphere.
# Altitudes are in km, densities in kg/m^3 and scale heights in km.
# The model is intentionally simple: it does not include solar activity,
# geomagnetic activity, latitude, local solar time or seasonal effects.
_REFERENCE_ALTITUDES_KM = np.array([
    0.0,
    25.0,
    30.0,
    40.0,
    50.0,
    60.0,
    70.0,
    80.0,
    90.0,
    100.0,
    110.0,
    120.0,
    130.0,
    140.0,
    150.0,
    180.0,
    200.0,
    250.0,
    300.0,
    350.0,
    400.0,
    450.0,
    500.0,
    600.0,
    700.0,
    800.0,
    900.0,
    1000.0,
])

_REFERENCE_DENSITIES = np.array([
    1.225,
    3.899e-2,
    1.774e-2,
    3.972e-3,
    1.057e-3,
    3.206e-4,
    8.770e-5,
    1.905e-5,
    3.396e-6,
    5.297e-7,
    9.661e-8,
    2.438e-8,
    8.484e-9,
    3.845e-9,
    2.070e-9,
    5.464e-10,
    2.789e-10,
    7.248e-11,
    2.418e-11,
    9.518e-12,
    3.725e-12,
    1.585e-12,
    6.967e-13,
    1.454e-13,
    3.614e-14,
    1.170e-14,
    5.245e-15,
    3.019e-15,
])

_SCALE_HEIGHTS_KM = np.array([
    7.249,
    6.349,
    6.682,
    7.554,
    8.382,
    7.714,
    6.549,
    5.799,
    5.382,
    5.877,
    7.263,
    9.473,
    12.636,
    16.149,
    22.523,
    29.740,
    37.105,
    45.546,
    53.628,
    53.298,
    58.515,
    60.828,
    63.822,
    71.835,
    88.667,
    124.640,
    181.050,
    268.000,
])


def earth_exponential_density(
    altitude: float,
) -> float:
    """Return a simplified Earth atmospheric density in kg/m^3.

    Parameters
    ----------
    altitude
        Geometric altitude above the reference Earth radius in meters.

    Notes
    -----
    A piecewise exponential model is used between tabulated reference
    altitudes. Above 1000 km the last exponential segment is extrapolated.
    This model is suitable for educational orbit-decay experiments, not for
    operational lifetime prediction.
    """

    altitude = float(altitude)

    if not np.isfinite(altitude):
        raise ValueError(
            "Altitude must be finite."
        )

    if altitude < 0.0:
        raise ValueError(
            "Atmospheric altitude cannot be negative."
        )

    altitude_km = (
        altitude / 1e3
    )

    # Use the last reference layer for extrapolation above 1000 km.
    index = int(
        np.searchsorted(
            _REFERENCE_ALTITUDES_KM,
            altitude_km,
            side="right",
        )
        - 1
    )

    index = min(
        max(index, 0),
        len(_REFERENCE_ALTITUDES_KM) - 1,
    )

    reference_altitude = (
        _REFERENCE_ALTITUDES_KM[index]
    )

    reference_density = (
        _REFERENCE_DENSITIES[index]
    )

    scale_height = (
        _SCALE_HEIGHTS_KM[index]
    )

    density = (
        reference_density
        * np.exp(
            -(
                altitude_km
                - reference_altitude
            )
            / scale_height
        )
    )

    return float(density)