"""Generic impulsive orbital mission engine."""

from dataclasses import dataclass
from typing import Literal

import numpy as np

from src.orbital.state import OrbitalState

from src.physics.bodies import CelestialBody
from src.physics.two_body import state_derivative

from src.integrators.rk4 import RK4

from src.simulation.simulator import Simulator
from src.simulation.result import SimulationResult

from src.orbital.elements import (
    radial_velocity,
    orbital_eccentricity,
    orbital_period_from_state,
)

from src.mission.maneuver import (
    ManeuverResult,
    apply_delta_v,
    prograde_burn as apply_prograde_burn,
    retrograde_burn as apply_retrograde_burn,
    tangential_burn as apply_tangential_burn,
)

from src.mission.orbit_changes import (
    plan_circularization,
    plan_set_apoapsis,
    plan_set_periapsis,
)


PhaseType = Literal[
    "coast",
    "burn",
]


def _copy_state(
    state: OrbitalState,
) -> OrbitalState:
    """
    Return an independent copy of an OrbitalState.
    """

    return OrbitalState(
        position=np.array(
            state.position,
            dtype=float,
            copy=True,
        ),
        velocity=np.array(
            state.velocity,
            dtype=float,
            copy=True,
        ),
    )


@dataclass(frozen=True)
class MissionPhase:
    """
    One phase of a mission.
    """

    kind: PhaseType
    label: str

    start_time: float
    end_time: float

    state_before: OrbitalState
    state_after: OrbitalState

    simulation_result: SimulationResult | None = None
    maneuver_result: ManeuverResult | None = None


@dataclass(frozen=True)
class MissionResult:
    """
    Immutable result of a complete mission.
    """

    initial_state: OrbitalState
    final_state: OrbitalState

    phases: tuple[
        MissionPhase,
        ...
    ]

    elapsed_time: float

    @property
    def coast_phases(
        self,
    ) -> tuple[
        MissionPhase,
        ...
    ]:

        return tuple(
            phase
            for phase in self.phases
            if phase.kind == "coast"
        )

    @property
    def burn_phases(
        self,
    ) -> tuple[
        MissionPhase,
        ...
    ]:

        return tuple(
            phase
            for phase in self.phases
            if phase.kind == "burn"
        )

    @property
    def total_delta_v(
        self,
    ) -> float:

        return float(
            sum(
                phase
                .maneuver_result
                .delta_v_magnitude

                for phase
                in self.burn_phases

                if phase.maneuver_result
                is not None
            )
        )

    def trajectory_positions(
        self,
    ) -> np.ndarray:
        """
        Return all propagated trajectory positions.
        """

        arrays = []

        first = True

        for phase in self.coast_phases:

            positions = (
                phase
                .simulation_result
                .positions
            )

            if first:

                arrays.append(
                    positions
                )

                first = False

            else:

                arrays.append(
                    positions[1:]
                )

        if not arrays:

            return np.empty(
                (0, 3),
                dtype=float,
            )

        return np.vstack(
            arrays
        )

    def trajectory_times(
        self,
    ) -> np.ndarray:
        """
        Return mission-relative trajectory times.
        """

        arrays = []

        first = True

        for phase in self.coast_phases:

            times = (
                phase.start_time
                + phase
                .simulation_result
                .times
            )

            if first:

                arrays.append(
                    times
                )

                first = False

            else:

                arrays.append(
                    times[1:]
                )

        if not arrays:

            return np.empty(
                0,
                dtype=float,
            )

        return np.concatenate(
            arrays
        )


class Mission:
    """
    Generic impulsive orbital mission.

    The mission may start from any OrbitalState and
    contain arbitrary coast and impulsive-burn phases.
    """

    def __init__(
        self,
        initial_state: OrbitalState,
        body: CelestialBody,
        dt: float = 10.0,
    ):

        if dt <= 0.0:
            raise ValueError(
                "Time step dt must be positive."
            )

        self.body = body
        self.dt = float(
            dt
        )

        self._initial_state = (
            _copy_state(
                initial_state
            )
        )

        self._current_state = (
            _copy_state(
                initial_state
            )
        )

        self._elapsed_time = 0.0

        self._phases: list[
            MissionPhase
        ] = []

        derivative = lambda state: state_derivative(
            state,
            self.body,
        )

        self._simulator = Simulator(
            integrator=RK4(),
            derivative=derivative,
            dt=self.dt,
        )

    @property
    def current_state(
        self,
    ) -> OrbitalState:

        return _copy_state(
            self._current_state
        )

    @property
    def elapsed_time(
        self,
    ) -> float:

        return self._elapsed_time

    @property
    def burn_phases(
        self,
    ) -> tuple[
        MissionPhase,
        ...
    ]:

        return tuple(
            phase
            for phase in self._phases
            if phase.kind == "burn"
        )

    # ==================================================
    # PROPAGATION
    # ==================================================

    def coast(
        self,
        duration: float,
        label: str | None = None,
    ) -> SimulationResult:
        """
        Numerically propagate for a fixed duration.
        """

        if duration <= 0.0:
            raise ValueError(
                "Coast duration must be positive."
            )

        state_before = (
            _copy_state(
                self._current_state
            )
        )

        start_time = (
            self._elapsed_time
        )

        simulation = (
            self._simulator.run(
                initial_state=self._current_state,
                duration=duration,
            )
        )

        self._current_state = (
            _copy_state(
                simulation.final_state
            )
        )

        self._elapsed_time += (
            duration
        )

        if label is None:
            label = (
                f"Coast "
                f"{len(self._phases) + 1}"
            )

        phase = MissionPhase(
            kind="coast",
            label=label,
            start_time=start_time,
            end_time=self._elapsed_time,
            state_before=state_before,
            state_after=(
                _copy_state(
                    self._current_state
                )
            ),
            simulation_result=simulation,
            maneuver_result=None,
        )

        self._phases.append(
            phase
        )

        return simulation

    # ==================================================
    # APSIS EVENT DETECTION
    # ==================================================

    def _find_next_apsis_time(
        self,
        kind: Literal[
            "apoapsis",
            "periapsis",
        ],
    ) -> float:
        """
        Find the time until the next requested apsis.
        """

        if kind not in (
            "apoapsis",
            "periapsis",
        ):
            raise ValueError(
                "kind must be 'apoapsis' or 'periapsis'."
            )

        eccentricity = (
            orbital_eccentricity(
                self._current_state,
                self.body,
            )
        )

        if eccentricity < 1e-8:
            raise ValueError(
                "A circular orbit has no unique "
                "apoapsis or periapsis."
            )

        period = (
            orbital_period_from_state(
                self._current_state,
                self.body,
            )
        )

        search_duration = (
            1.05
            * period
        )

        probe = (
            self._simulator.run(
                initial_state=self._current_state,
                duration=search_duration,
            )
        )

        positions = (
            probe.positions
        )

        velocities = (
            probe.velocities
        )

        times = (
            probe.times
        )

        radii = np.linalg.norm(
            positions,
            axis=1,
        )

        radial_velocities = (
            np.sum(
                positions
                * velocities,
                axis=1,
            )
            / radii
        )

        tolerance = 1e-8

        previous_sign = None
        previous_index = None

        for index, value in enumerate(
            radial_velocities
        ):

            if value > tolerance:
                sign = 1

            elif value < -tolerance:
                sign = -1

            else:
                sign = 0

            if sign == 0:
                continue

            if previous_sign is None:

                previous_sign = sign
                previous_index = index

                continue

            apoapsis_crossing = (
                previous_sign == 1
                and sign == -1
            )

            periapsis_crossing = (
                previous_sign == -1
                and sign == 1
            )

            found = (
                (
                    kind == "apoapsis"
                    and apoapsis_crossing
                )
                or
                (
                    kind == "periapsis"
                    and periapsis_crossing
                )
            )

            if found:

                i0 = previous_index
                i1 = index

                t0 = times[i0]
                t1 = times[i1]

                vr0 = (
                    radial_velocities[i0]
                )

                vr1 = (
                    radial_velocities[i1]
                )

                event_time = (
                    t0
                    - vr0
                    * (
                        t1 - t0
                    )
                    / (
                        vr1 - vr0
                    )
                )

                if event_time <= 0.0:
                    raise RuntimeError(
                        "Detected apsis time is not positive."
                    )

                return float(
                    event_time
                )

            previous_sign = sign
            previous_index = index

        raise RuntimeError(
            f"Unable to detect the next {kind} "
            "within one orbital period."
        )

    def coast_until_apsis(
        self,
        kind: Literal[
            "apoapsis",
            "periapsis",
        ],
        label: str | None = None,
    ) -> SimulationResult:

        duration = (
            self._find_next_apsis_time(
                kind
            )
        )

        if label is None:

            if kind == "apoapsis":
                label = "Coast to apoapsis"

            else:
                label = "Coast to periapsis"

        return self.coast(
            duration=duration,
            label=label,
        )

    def coast_until_apoapsis(
        self,
        label: str | None = None,
    ) -> SimulationResult:

        return self.coast_until_apsis(
            kind="apoapsis",
            label=label,
        )

    def coast_until_periapsis(
        self,
        label: str | None = None,
    ) -> SimulationResult:

        return self.coast_until_apsis(
            kind="periapsis",
            label=label,
        )

    # ==================================================
    # MANEUVERS
    # ==================================================

    def _record_burn(
        self,
        maneuver: ManeuverResult,
        label: str | None,
    ) -> ManeuverResult:

        if label is None:

            label = (
                f"Burn "
                f"{len(self.burn_phases) + 1}"
            )

        phase = MissionPhase(
            kind="burn",
            label=label,
            start_time=self._elapsed_time,
            end_time=self._elapsed_time,
            state_before=(
                _copy_state(
                    maneuver.state_before
                )
            ),
            state_after=(
                _copy_state(
                    maneuver.state_after
                )
            ),
            simulation_result=None,
            maneuver_result=maneuver,
        )

        self._phases.append(
            phase
        )

        self._current_state = (
            _copy_state(
                maneuver.state_after
            )
        )

        return maneuver

    def burn(
        self,
        delta_v_vector: np.ndarray,
        label: str | None = None,
    ) -> ManeuverResult:

        maneuver = apply_delta_v(
            self._current_state,
            delta_v_vector,
        )

        return self._record_burn(
            maneuver,
            label,
        )

    def prograde_burn(
        self,
        delta_v: float,
        label: str | None = None,
    ) -> ManeuverResult:

        maneuver = (
            apply_prograde_burn(
                self._current_state,
                delta_v,
            )
        )

        return self._record_burn(
            maneuver,
            label,
        )

    def retrograde_burn(
        self,
        delta_v: float,
        label: str | None = None,
    ) -> ManeuverResult:

        maneuver = (
            apply_retrograde_burn(
                self._current_state,
                delta_v,
            )
        )

        return self._record_burn(
            maneuver,
            label,
        )

    def tangential_burn(
        self,
        delta_v: float,
        label: str | None = None,
    ) -> ManeuverResult:

        maneuver = (
            apply_tangential_burn(
                self._current_state,
                delta_v,
            )
        )

        return self._record_burn(
            maneuver,
            label,
        )

    # ==================================================
    # HIGH-LEVEL ORBIT CHANGES
    # ==================================================

    def circularize(
        self,
        label: str | None = None,
    ) -> ManeuverResult:

        plan = plan_circularization(
            state=self._current_state,
            body=self.body,
        )

        if label is None:
            label = "Circularization burn"

        return self.burn(
            delta_v_vector=(
                plan.delta_v_vector
            ),
            label=label,
        )

    def set_apoapsis_to(
        self,
        target_radius: float,
        label: str | None = None,
    ) -> ManeuverResult:

        plan = plan_set_apoapsis(
            state=self._current_state,
            target_radius=target_radius,
            body=self.body,
        )

        if label is None:
            label = "Set apoapsis"

        return self.burn(
            delta_v_vector=(
                plan.delta_v_vector
            ),
            label=label,
        )

    def set_periapsis_to(
        self,
        target_radius: float,
        label: str | None = None,
    ) -> ManeuverResult:

        plan = plan_set_periapsis(
            state=self._current_state,
            target_radius=target_radius,
            body=self.body,
        )

        if label is None:
            label = "Set periapsis"

        return self.burn(
            delta_v_vector=(
                plan.delta_v_vector
            ),
            label=label,
        )

    # ==================================================
    # RESULT
    # ==================================================

    def result(
        self,
    ) -> MissionResult:

        return MissionResult(
            initial_state=(
                _copy_state(
                    self._initial_state
                )
            ),
            final_state=(
                _copy_state(
                    self._current_state
                )
            ),
            phases=tuple(
                self._phases
            ),
            elapsed_time=(
                self._elapsed_time
            ),
        )