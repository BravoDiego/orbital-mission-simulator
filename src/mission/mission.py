from dataclasses import dataclass
from typing import Literal
from math import pi, sqrt

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody
from src.physics.two_body import state_derivative

from src.integrators.rk4 import RK4

from src.simulation.simulator import Simulator
from src.simulation.result import SimulationResult

from src.mission.maneuver import (
    ManeuverResult,
    apply_delta_v,
    prograde_burn as apply_prograde_burn,
    retrograde_burn as apply_retrograde_burn,
    tangential_burn as apply_tangential_burn,
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


def radial_velocity(
    state: OrbitalState,
) -> float:
    """
    Return the radial component of the spacecraft velocity.

    Positive:
        spacecraft moving away from the central body.

    Negative:
        spacecraft moving toward the central body.
    """

    position = np.asarray(
        state.position,
        dtype=float,
    )

    velocity = np.asarray(
        state.velocity,
        dtype=float,
    )

    radius = np.linalg.norm(
        position
    )

    if radius == 0.0:
        raise ValueError(
            "Radial velocity is undefined at the body's center."
        )

    return float(
        np.dot(
            position,
            velocity,
        )
        / radius
    )


def orbital_eccentricity(
    state: OrbitalState,
    body: CelestialBody,
) -> float:
    """
    Compute orbital eccentricity from a Cartesian state.
    """

    r_vector = np.asarray(
        state.position,
        dtype=float,
    )

    v_vector = np.asarray(
        state.velocity,
        dtype=float,
    )

    r = np.linalg.norm(
        r_vector
    )

    if r == 0.0:
        raise ValueError(
            "Orbital eccentricity is undefined at the body's center."
        )

    h_vector = np.cross(
        r_vector,
        v_vector,
    )

    e_vector = (
        np.cross(
            v_vector,
            h_vector,
        )
        / body.mu
        - r_vector / r
    )

    return float(
        np.linalg.norm(
            e_vector
        )
    )


def orbital_period_from_state(
    state: OrbitalState,
    body: CelestialBody,
) -> float:
    """
    Compute the orbital period from the spacecraft state.

    Only bound elliptical orbits have a finite period.
    """

    r = np.linalg.norm(
        state.position
    )

    v = np.linalg.norm(
        state.velocity
    )

    specific_energy = (
        0.5 * v**2
        - body.mu / r
    )

    if specific_energy >= 0.0:
        raise ValueError(
            "The current orbit is not bound. "
            "A finite orbital period does not exist."
        )

    semi_major_axis = (
        -body.mu
        / (2.0 * specific_energy)
    )

    return (
        2.0
        * pi
        * sqrt(
            semi_major_axis**3
            / body.mu
        )
    )

@dataclass(frozen=True)
class MissionPhase:
    """
    One phase of a mission.

    A phase can be either:
    - a coast propagated numerically,
    - an instantaneous impulsive burn.
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
    Complete result of a mission.
    """

    initial_state: OrbitalState
    final_state: OrbitalState

    phases: tuple[MissionPhase, ...]

    elapsed_time: float

    @property
    def coast_phases(
        self,
    ) -> tuple[MissionPhase, ...]:
        """
        Return all propagation phases.
        """

        return tuple(
            phase
            for phase in self.phases
            if phase.kind == "coast"
        )

    @property
    def burn_phases(
        self,
    ) -> tuple[MissionPhase, ...]:
        """
        Return all impulsive maneuver phases.
        """

        return tuple(
            phase
            for phase in self.phases
            if phase.kind == "burn"
        )

    @property
    def total_delta_v(
        self,
    ) -> float:
        """
        Return the total mission delta-v budget.

        Delta-v magnitudes are added, regardless of direction.
        """

        total = 0.0

        for phase in self.burn_phases:

            if phase.maneuver_result is not None:

                total += (
                    phase
                    .maneuver_result
                    .delta_v_magnitude
                )

        return total

    def trajectory_positions(
        self,
    ) -> np.ndarray:
        """
        Return all numerically propagated positions.

        Burn phases are instantaneous and therefore do not
        generate additional trajectory points.
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
                # Avoid duplicating the common state between
                # two consecutive propagation phases.
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
        Return mission-relative times corresponding to
        trajectory_positions().
        """

        arrays = []

        first = True

        for phase in self.coast_phases:

            times = (
                phase.start_time
                + phase.simulation_result.times
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

    A mission starts from any OrbitalState and can contain
    arbitrary coast phases and impulsive maneuvers.
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
        self.dt = dt

        self._initial_state = _copy_state(
            initial_state
        )

        self._current_state = _copy_state(
            initial_state
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
        """
        Current spacecraft state.
        """

        return _copy_state(
            self._current_state
        )

    @property
    def elapsed_time(
        self,
    ) -> float:
        """
        Current mission elapsed time.
        """

        return self._elapsed_time

    def coast(
        self,
        duration: float,
        label: str | None = None,
    ) -> SimulationResult:
        """
        Numerically propagate the spacecraft for a duration.
        """

        if duration <= 0.0:
            raise ValueError(
                "Coast duration must be positive."
            )

        state_before = _copy_state(
            self._current_state
        )

        start_time = (
            self._elapsed_time
        )

        simulation = self._simulator.run(
            initial_state=self._current_state,
            duration=duration,
        )

        self._current_state = _copy_state(
            simulation.final_state
        )

        self._elapsed_time += duration

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
            state_after=_copy_state(
                self._current_state
            ),
            simulation_result=simulation,
            maneuver_result=None,
        )

        self._phases.append(
            phase
        )

        return simulation

    def _find_next_apsis_time(
        self,
        kind: Literal[
            "apoapsis",
            "periapsis",
        ],
    ) -> float:
        """
        Find the time until the next requested apsis.

        Apoapsis:
            radial velocity changes from positive to negative.

        Periapsis:
            radial velocity changes from negative to positive.

        The trajectory is first probed numerically over slightly
        more than one orbital period. The zero crossing is then
        linearly interpolated between the two surrounding samples.
        """

        if kind not in (
            "apoapsis",
            "periapsis",
        ):
            raise ValueError(
                "kind must be 'apoapsis' or 'periapsis'."
            )

        eccentricity = orbital_eccentricity(
            self._current_state,
            self.body,
        )

        # On a perfectly circular orbit every point can be
        # regarded as both periapsis and apoapsis.
        if eccentricity < 1e-8:
            raise ValueError(
                "A circular orbit has no unique "
                "apoapsis or periapsis."
            )

        period = orbital_period_from_state(
            self._current_state,
            self.body,
        )

        # Slightly more than one period guarantees that the
        # requested apsis can be found even if we start just
        # after that same apsis.
        search_duration = (
            1.05 * period
        )

        probe = self._simulator.run(
            initial_state=self._current_state,
            duration=search_duration,
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
                positions * velocities,
                axis=1,
            )
            / radii
        )

        # Ignore extremely small numerical values around zero.
        tolerance = 1e-8

        previous_sign = None
        previous_index = None

        for index in range(
            len(radial_velocities)
        ):

            value = (
                radial_velocities[index]
            )

            if value > tolerance:
                sign = 1

            elif value < -tolerance:
                sign = -1

            else:
                sign = 0

            # Ignore exact/numerical zero values.
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
                kind == "apoapsis"
                and apoapsis_crossing
            ) or (
                kind == "periapsis"
                and periapsis_crossing
            )

            if found:

                i0 = previous_index
                i1 = index

                t0 = times[i0]
                t1 = times[i1]

                vr0 = radial_velocities[i0]
                vr1 = radial_velocities[i1]

                # Linear interpolation of:
                #
                # radial_velocity(t_event) = 0
                #
                event_time = (
                    t0
                    - vr0
                    * (t1 - t0)
                    / (vr1 - vr0)
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
        """
        Propagate until the next requested apsis.
        """

        duration = self._find_next_apsis_time(
            kind
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
        """
        Propagate until the next apoapsis.
        """

        return self.coast_until_apsis(
            kind="apoapsis",
            label=label,
        )


    def coast_until_periapsis(
        self,
        label: str | None = None,
    ) -> SimulationResult:
        """
        Propagate until the next periapsis.
        """

        return self.coast_until_apsis(
            kind="periapsis",
            label=label,
        )

    def _record_burn(
        self,
        maneuver: ManeuverResult,
        label: str | None,
    ) -> ManeuverResult:
        """
        Store a burn in the mission history.
        """

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
            state_before=_copy_state(
                maneuver.state_before
            ),
            state_after=_copy_state(
                maneuver.state_after
            ),
            simulation_result=None,
            maneuver_result=maneuver,
        )

        self._phases.append(
            phase
        )

        self._current_state = _copy_state(
            maneuver.state_after
        )

        return maneuver

    @property
    def burn_phases(
        self,
    ) -> tuple[MissionPhase, ...]:

        return tuple(
            phase
            for phase in self._phases
            if phase.kind == "burn"
        )

    def burn(
        self,
        delta_v_vector: np.ndarray,
        label: str | None = None,
    ) -> ManeuverResult:
        """
        Apply an arbitrary vector delta-v.
        """

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
        """
        Apply a positive prograde burn.
        """

        maneuver = apply_prograde_burn(
            self._current_state,
            delta_v,
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
        """
        Apply a positive retrograde burn.
        """

        maneuver = apply_retrograde_burn(
            self._current_state,
            delta_v,
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
        """
        Apply a signed tangential burn.

        Positive -> prograde
        Negative -> retrograde
        """

        maneuver = apply_tangential_burn(
            self._current_state,
            delta_v,
        )

        return self._record_burn(
            maneuver,
            label,
        )

    def result(
        self,
    ) -> MissionResult:
        """
        Build an immutable snapshot of the mission result.
        """

        return MissionResult(
            initial_state=_copy_state(
                self._initial_state
            ),
            final_state=_copy_state(
                self._current_state
            ),
            phases=tuple(
                self._phases
            ),
            elapsed_time=self._elapsed_time,
        )