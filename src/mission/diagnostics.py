"""Scientific diagnostics for complete orbital missions."""

from dataclasses import dataclass

import numpy as np

from src.orbital.state import OrbitalState
from src.physics.bodies import CelestialBody

from src.orbital.conservation import (
    specific_orbital_energy,
)

from src.orbital.elements import (
    semi_major_axis,
    orbital_eccentricity,
)

from src.mission.mission import (
    MissionResult,
)


@dataclass(frozen=True)
class MissionDiagnostics:
    """
    Time-resolved orbital diagnostics.
    """

    times: np.ndarray

    positions: np.ndarray
    velocities: np.ndarray

    radii: np.ndarray
    speeds: np.ndarray

    specific_energy: np.ndarray
    semi_major_axis: np.ndarray
    eccentricity: np.ndarray

    event_types: tuple[
        str,
        ...
    ]

    phase_labels: tuple[
        str,
        ...
    ]

    @property
    def burn_indices(
        self,
    ) -> np.ndarray:

        return np.array([
            index

            for index, event_type
            in enumerate(
                self.event_types
            )

            if event_type == "burn"
        ], dtype=int)


def _states_from_arrays(
    positions: np.ndarray,
    velocities: np.ndarray,
) -> list[OrbitalState]:
    """
    Build OrbitalState objects from Cartesian histories.
    """

    return [
        OrbitalState(
            position=np.array(
                position,
                dtype=float,
                copy=True,
            ),
            velocity=np.array(
                velocity,
                dtype=float,
                copy=True,
            ),
        )

        for position, velocity
        in zip(
            positions,
            velocities,
        )
    ]


def _compute_orbital_quantities(
    positions: np.ndarray,
    velocities: np.ndarray,
    body: CelestialBody,
):
    """
    Compute diagnostics using the canonical orbital modules.
    """

    states = _states_from_arrays(
        positions,
        velocities,
    )

    radii = np.array([
        np.linalg.norm(
            state.position
        )

        for state
        in states
    ], dtype=float)

    speeds = np.array([
        np.linalg.norm(
            state.velocity
        )

        for state
        in states
    ], dtype=float)

    energies = np.array([
        specific_orbital_energy(
            state,
            body,
        )

        for state
        in states
    ], dtype=float)

    semi_major_axes = np.array([
        semi_major_axis(
            state,
            body,
        )

        for state
        in states
    ], dtype=float)

    eccentricities = np.array([
        orbital_eccentricity(
            state,
            body,
        )

        for state
        in states
    ], dtype=float)

    return (
        radii,
        speeds,
        energies,
        semi_major_axes,
        eccentricities,
    )


def compute_mission_diagnostics(
    mission: MissionResult,
    body: CelestialBody,
) -> MissionDiagnostics:
    """
    Reconstruct a complete time-resolved mission history.

    Burn states are inserted at the same mission time as
    the pre-burn state so impulsive discontinuities remain
    visible in diagnostic plots.
    """

    times = [
        0.0
    ]

    positions = [
        mission
        .initial_state
        .position
        .copy()
    ]

    velocities = [
        mission
        .initial_state
        .velocity
        .copy()
    ]

    event_types = [
        "start"
    ]

    phase_labels = [
        "Mission start"
    ]

    for phase in mission.phases:

        if phase.kind == "coast":

            simulation = (
                phase.simulation_result
            )

            for index in range(
                1,
                len(
                    simulation.times
                ),
            ):

                times.append(
                    phase.start_time
                    + simulation.times[index]
                )

                positions.append(
                    simulation
                    .positions[index]
                    .copy()
                )

                velocities.append(
                    simulation
                    .velocities[index]
                    .copy()
                )

                event_types.append(
                    "coast"
                )

                phase_labels.append(
                    phase.label
                )

        elif phase.kind == "burn":

            maneuver = (
                phase.maneuver_result
            )

            times.append(
                phase.start_time
            )

            positions.append(
                maneuver
                .state_after
                .position
                .copy()
            )

            velocities.append(
                maneuver
                .state_after
                .velocity
                .copy()
            )

            event_types.append(
                "burn"
            )

            phase_labels.append(
                phase.label
            )

        else:

            raise ValueError(
                f"Unknown mission phase type: "
                f"{phase.kind}"
            )

    times_array = np.asarray(
        times,
        dtype=float,
    )

    positions_array = np.asarray(
        positions,
        dtype=float,
    )

    velocities_array = np.asarray(
        velocities,
        dtype=float,
    )

    (
        radii,
        speeds,
        energies,
        semi_major_axes,
        eccentricities,
    ) = _compute_orbital_quantities(
        positions_array,
        velocities_array,
        body,
    )

    return MissionDiagnostics(
        times=times_array,
        positions=positions_array,
        velocities=velocities_array,
        radii=radii,
        speeds=speeds,
        specific_energy=energies,
        semi_major_axis=(
            semi_major_axes
        ),
        eccentricity=(
            eccentricities
        ),
        event_types=tuple(
            event_types
        ),
        phase_labels=tuple(
            phase_labels
        ),
    )