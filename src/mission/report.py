"""Structured scientific reporting for orbital missions."""

from dataclasses import dataclass

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody

from src.orbital.conservation import (
    specific_orbital_energy,
)

from src.orbital.elements import (
    compute_orbital_elements,
)

from src.mission.mission import (
    MissionResult,
)


@dataclass(frozen=True)
class OrbitalStateSummary:
    """
    Orbital quantities derived from one spacecraft state.
    """

    radius: float
    altitude: float
    speed: float

    specific_energy: float

    semi_major_axis: float | None
    eccentricity: float

    periapsis_radius: float | None
    apoapsis_radius: float | None


@dataclass(frozen=True)
class BurnSummary:
    """
    Summary of one impulsive maneuver.
    """

    label: str
    time: float

    delta_v: float
    direction: str

    speed_before: float
    speed_after: float

    energy_before: float
    energy_after: float


@dataclass(frozen=True)
class ValidationCheck:
    """
    One mission validation result.
    """

    name: str
    passed: bool
    message: str


@dataclass(frozen=True)
class MissionReport:
    """
    Complete mission report.
    """

    initial_orbit: OrbitalStateSummary
    final_orbit: OrbitalStateSummary

    burns: tuple[
        BurnSummary,
        ...
    ]

    elapsed_time: float

    phase_count: int
    burn_count: int

    total_delta_v: float

    validations: tuple[
        ValidationCheck,
        ...
    ]


def summarize_orbital_state(
    state: OrbitalState,
    body: CelestialBody,
) -> OrbitalStateSummary:
    """
    Build an orbital summary using canonical orbital modules.
    """

    radius = float(
        np.linalg.norm(
            state.position
        )
    )

    if radius == 0.0:
        raise ValueError(
            "Orbital state cannot be summarized "
            "at the body's center."
        )

    speed = float(
        np.linalg.norm(
            state.velocity
        )
    )

    energy = (
        specific_orbital_energy(
            state,
            body,
        )
    )

    elements = (
        compute_orbital_elements(
            state,
            body,
        )
    )

    semi_major = (
        elements.semi_major_axis
    )

    if not np.isfinite(
        semi_major
    ):
        semi_major_summary = None

    else:
        semi_major_summary = (
            float(
                semi_major
            )
        )

    return OrbitalStateSummary(
        radius=radius,
        altitude=(
            radius
            - body.radius
        ),
        speed=speed,
        specific_energy=energy,
        semi_major_axis=(
            semi_major_summary
        ),
        eccentricity=(
            elements.eccentricity
        ),
        periapsis_radius=(
            elements.periapsis_radius
        ),
        apoapsis_radius=(
            elements.apoapsis_radius
        ),
    )


def _burn_direction(
    maneuver,
) -> str:
    """
    Classify a burn relative to pre-burn velocity.
    """

    velocity = np.asarray(
        maneuver
        .state_before
        .velocity,
        dtype=float,
    )

    delta_v = np.asarray(
        maneuver.delta_v_vector,
        dtype=float,
    )

    speed = np.linalg.norm(
        velocity
    )

    delta_v_magnitude = (
        np.linalg.norm(
            delta_v
        )
    )

    if delta_v_magnitude == 0.0:
        return "zero"

    if speed == 0.0:
        return "arbitrary"

    velocity_direction = (
        velocity
        / speed
    )

    projection = float(
        np.dot(
            delta_v,
            velocity_direction,
        )
    )

    transverse_part = (
        delta_v
        - projection
        * velocity_direction
    )

    transverse_magnitude = (
        np.linalg.norm(
            transverse_part
        )
    )

    tolerance = (
        1e-8
        * max(
            delta_v_magnitude,
            1.0,
        )
    )

    if (
        transverse_magnitude
        <= tolerance
    ):

        if projection > 0.0:
            return "prograde"

        if projection < 0.0:
            return "retrograde"

    return "vector"


def _build_burn_summaries(
    mission: MissionResult,
    body: CelestialBody,
) -> tuple[
    BurnSummary,
    ...
]:

    summaries = []

    for phase in mission.burn_phases:

        maneuver = (
            phase.maneuver_result
        )

        speed_before = float(
            np.linalg.norm(
                maneuver
                .state_before
                .velocity
            )
        )

        speed_after = float(
            np.linalg.norm(
                maneuver
                .state_after
                .velocity
            )
        )

        energy_before = (
            specific_orbital_energy(
                maneuver.state_before,
                body,
            )
        )

        energy_after = (
            specific_orbital_energy(
                maneuver.state_after,
                body,
            )
        )

        summaries.append(
            BurnSummary(
                label=phase.label,
                time=phase.start_time,
                delta_v=(
                    maneuver
                    .delta_v_magnitude
                ),
                direction=(
                    _burn_direction(
                        maneuver
                    )
                ),
                speed_before=(
                    speed_before
                ),
                speed_after=(
                    speed_after
                ),
                energy_before=(
                    energy_before
                ),
                energy_after=(
                    energy_after
                ),
            )
        )

    return tuple(
        summaries
    )


def _validate_mission(
    mission: MissionResult,
) -> tuple[
    ValidationCheck,
    ...
]:

    checks = []

    # --------------------------------------------------
    # Timeline
    # --------------------------------------------------

    timeline_valid = True

    previous_end = 0.0

    for phase in mission.phases:

        if (
            phase.start_time
            < previous_end - 1e-9
        ):
            timeline_valid = False
            break

        if (
            phase.end_time
            < phase.start_time - 1e-9
        ):
            timeline_valid = False
            break

        previous_end = (
            phase.end_time
        )

    if mission.phases:

        timeline_valid = (
            timeline_valid
            and np.isclose(
                mission
                .phases[-1]
                .end_time,
                mission.elapsed_time,
            )
        )

    checks.append(
        ValidationCheck(
            name="Mission timeline",
            passed=bool(
                timeline_valid
            ),
            message=(
                "Phase times are consistent."
                if timeline_valid
                else
                "Mission phase timing is inconsistent."
            ),
        )
    )

    # --------------------------------------------------
    # Delta-v budget
    # --------------------------------------------------

    manual_delta_v = sum(
        phase
        .maneuver_result
        .delta_v_magnitude

        for phase
        in mission.burn_phases
    )

    delta_v_valid = (
        np.isclose(
            manual_delta_v,
            mission.total_delta_v,
        )
    )

    checks.append(
        ValidationCheck(
            name="Delta-v budget",
            passed=bool(
                delta_v_valid
            ),
            message=(
                "Burn magnitudes match mission total."
                if delta_v_valid
                else
                "Mission delta-v budget is inconsistent."
            ),
        )
    )

    # --------------------------------------------------
    # Finite final state
    # --------------------------------------------------

    finite_state = (
        np.all(
            np.isfinite(
                mission
                .final_state
                .position
            )
        )
        and
        np.all(
            np.isfinite(
                mission
                .final_state
                .velocity
            )
        )
    )

    checks.append(
        ValidationCheck(
            name="Finite final state",
            passed=bool(
                finite_state
            ),
            message=(
                "Final position and velocity are finite."
                if finite_state
                else
                "Final state contains NaN or infinity."
            ),
        )
    )

    # --------------------------------------------------
    # Final-state history consistency
    # --------------------------------------------------

    if mission.phases:

        last_state = (
            mission
            .phases[-1]
            .state_after
        )

        final_state_valid = (
            np.allclose(
                last_state.position,
                mission
                .final_state
                .position,
            )
            and
            np.allclose(
                last_state.velocity,
                mission
                .final_state
                .velocity,
            )
        )

    else:

        final_state_valid = (
            np.allclose(
                mission
                .initial_state
                .position,
                mission
                .final_state
                .position,
            )
            and
            np.allclose(
                mission
                .initial_state
                .velocity,
                mission
                .final_state
                .velocity,
            )
        )

    checks.append(
        ValidationCheck(
            name="Final state consistency",
            passed=bool(
                final_state_valid
            ),
            message=(
                "Mission final state matches phase history."
                if final_state_valid
                else
                "Mission final state does not match history."
            ),
        )
    )

    return tuple(
        checks
    )


def build_mission_report(
    mission: MissionResult,
    body: CelestialBody,
) -> MissionReport:
    """
    Build a structured mission report.
    """

    return MissionReport(
        initial_orbit=(
            summarize_orbital_state(
                mission.initial_state,
                body,
            )
        ),
        final_orbit=(
            summarize_orbital_state(
                mission.final_state,
                body,
            )
        ),
        burns=(
            _build_burn_summaries(
                mission,
                body,
            )
        ),
        elapsed_time=(
            mission.elapsed_time
        ),
        phase_count=len(
            mission.phases
        ),
        burn_count=len(
            mission.burn_phases
        ),
        total_delta_v=(
            mission.total_delta_v
        ),
        validations=(
            _validate_mission(
                mission
            )
        ),
    )


def _format_optional_distance(
    value: float | None,
) -> str:

    if value is None:
        return "N/A"

    return (
        f"{value / 1e3:.3f} km"
    )


def _format_orbit_section(
    title: str,
    orbit: OrbitalStateSummary,
    body: CelestialBody,
) -> list[str]:

    lines = [
        title,
        "-" * 65,
        (
            f"Radius             : "
            f"{orbit.radius / 1e3:.3f} km"
        ),
        (
            f"Altitude           : "
            f"{orbit.altitude / 1e3:.3f} km"
        ),
        (
            f"Speed              : "
            f"{orbit.speed / 1e3:.6f} km/s"
        ),
        (
            f"Specific energy    : "
            f"{orbit.specific_energy / 1e6:.6f} MJ/kg"
        ),
        (
            f"Semi-major axis    : "
            f"{_format_optional_distance(orbit.semi_major_axis)}"
        ),
        (
            f"Eccentricity       : "
            f"{orbit.eccentricity:.8f}"
        ),
    ]

    if (
        orbit.periapsis_radius
        is not None
    ):

        lines.append(
            f"Periapsis radius   : "
            f"{orbit.periapsis_radius / 1e3:.3f} km"
        )

        lines.append(
            f"Periapsis altitude : "
            f"{(orbit.periapsis_radius - body.radius) / 1e3:.3f} km"
        )

    else:

        lines.append(
            "Periapsis          : N/A"
        )

    if (
        orbit.apoapsis_radius
        is not None
    ):

        lines.append(
            f"Apoapsis radius    : "
            f"{orbit.apoapsis_radius / 1e3:.3f} km"
        )

        lines.append(
            f"Apoapsis altitude  : "
            f"{(orbit.apoapsis_radius - body.radius) / 1e3:.3f} km"
        )

    else:

        lines.append(
            "Apoapsis           : N/A"
        )

    return lines


def format_mission_report(
    report: MissionReport,
    body: CelestialBody,
) -> str:

    lines = [
        "=" * 65,
        "ORBITAL MISSION REPORT",
        "=" * 65,
        "",
        f"Central body       : {body.name}",
        (
            f"Mission duration   : "
            f"{report.elapsed_time:.3f} s "
            f"({report.elapsed_time / 3600:.4f} h)"
        ),
        (
            f"Number of phases   : "
            f"{report.phase_count}"
        ),
        (
            f"Number of burns    : "
            f"{report.burn_count}"
        ),
        (
            f"Total delta-v      : "
            f"{report.total_delta_v:.3f} m/s"
        ),
        "",
    ]

    lines.extend(
        _format_orbit_section(
            "INITIAL ORBIT",
            report.initial_orbit,
            body,
        )
    )

    for index, burn in enumerate(
        report.burns,
        start=1,
    ):

        lines.extend([
            "",
            (
                f"BURN {index} — "
                f"{burn.label}"
            ),
            "-" * 65,
            (
                f"Mission time       : "
                f"{burn.time:.3f} s"
            ),
            (
                f"Delta-v            : "
                f"{burn.delta_v:.3f} m/s"
            ),
            (
                f"Direction          : "
                f"{burn.direction}"
            ),
            (
                f"Speed before       : "
                f"{burn.speed_before:.3f} m/s"
            ),
            (
                f"Speed after        : "
                f"{burn.speed_after:.3f} m/s"
            ),
            (
                f"Energy before      : "
                f"{burn.energy_before / 1e6:.6f} MJ/kg"
            ),
            (
                f"Energy after       : "
                f"{burn.energy_after / 1e6:.6f} MJ/kg"
            ),
            (
                f"Energy change      : "
                f"{(burn.energy_after - burn.energy_before) / 1e6:.6f} MJ/kg"
            ),
        ])

    lines.append("")

    lines.extend(
        _format_orbit_section(
            "FINAL ORBIT",
            report.final_orbit,
            body,
        )
    )

    lines.extend([
        "",
        "VALIDATION",
        "-" * 65,
    ])

    for check in report.validations:

        status = (
            "PASS"
            if check.passed
            else "FAIL"
        )

        lines.append(
            f"[{status}] "
            f"{check.name}: "
            f"{check.message}"
        )

    lines.extend([
        "",
        "=" * 65,
    ])

    return "\n".join(
        lines
    )


def print_mission_report(
    mission: MissionResult,
    body: CelestialBody,
):
    """
    Build and print a complete mission report.
    """

    report = build_mission_report(
        mission,
        body,
    )

    print(
        format_mission_report(
            report,
            body,
        )
    )