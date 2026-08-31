from dataclasses import dataclass

import numpy as np

from src.physics.bodies import CelestialBody
from src.mission.mission import MissionResult


@dataclass(frozen=True)
class MissionDiagnostics:
    """
    Time-resolved orbital diagnostics for a complete mission.
    """

    times: np.ndarray

    positions: np.ndarray
    velocities: np.ndarray

    radii: np.ndarray
    speeds: np.ndarray

    specific_energy: np.ndarray
    semi_major_axis: np.ndarray
    eccentricity: np.ndarray

    event_types: tuple[str, ...]
    phase_labels: tuple[str, ...]

    @property
    def burn_indices(self) -> np.ndarray:
        """
        Indices corresponding to post-burn states.
        """

        return np.array([
            index
            for index, event_type
            in enumerate(self.event_types)
            if event_type == "burn"
        ], dtype=int)


def _compute_orbital_quantities(
    positions: np.ndarray,
    velocities: np.ndarray,
    body: CelestialBody,
):
    """
    Compute orbital quantities from Cartesian states.
    """

    radii = np.linalg.norm(
        positions,
        axis=1,
    )

    speeds = np.linalg.norm(
        velocities,
        axis=1,
    )

    if np.any(radii == 0.0):
        raise ValueError(
            "Orbital diagnostics are undefined at the body's center."
        )

    # --------------------------------------------------
    # Specific orbital energy
    # --------------------------------------------------

    specific_energy = (
        0.5 * speeds**2
        - body.mu / radii
    )

    # --------------------------------------------------
    # Semi-major axis
    #
    # epsilon = -mu / (2a)
    # --------------------------------------------------

    semi_major_axis = np.full(
        len(specific_energy),
        np.nan,
        dtype=float,
    )

    non_parabolic = (
        np.abs(specific_energy)
        > 1e-12
    )

    semi_major_axis[
        non_parabolic
    ] = (
        -body.mu
        / (
            2.0
            * specific_energy[
                non_parabolic
            ]
        )
    )

    # --------------------------------------------------
    # Eccentricity vector
    #
    # e = (v × h)/mu - r_hat
    # --------------------------------------------------

    angular_momentum = np.cross(
        positions,
        velocities,
    )

    eccentricity_vectors = (
        np.cross(
            velocities,
            angular_momentum,
        )
        / body.mu
        - positions
        / radii[:, np.newaxis]
    )

    eccentricity = np.linalg.norm(
        eccentricity_vectors,
        axis=1,
    )

    return (
        radii,
        speeds,
        specific_energy,
        semi_major_axis,
        eccentricity,
    )


def compute_mission_diagnostics(
    mission: MissionResult,
    body: CelestialBody,
) -> MissionDiagnostics:
    """
    Reconstruct a complete time-resolved mission history.

    Coast phases contribute all their numerical RK4 samples.

    Burn phases add a new state at the same mission time as
    the state immediately before the burn. This preserves the
    instantaneous discontinuity in velocity and orbital elements.
    """

    times = [
        0.0
    ]

    positions = [
        mission.initial_state.position.copy()
    ]

    velocities = [
        mission.initial_state.velocity.copy()
    ]

    event_types = [
        "start"
    ]

    phase_labels = [
        "Mission start"
    ]

    # --------------------------------------------------
    # Reconstruct phase timeline
    # --------------------------------------------------

    for phase in mission.phases:

        if phase.kind == "coast":

            simulation = (
                phase.simulation_result
            )

            # Index 0 is already represented by the
            # previous mission state.
            for index in range(
                1,
                len(simulation.times),
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

            # Same time as before the burn:
            # only velocity changes.
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
                f"Unknown mission phase type: {phase.kind}"
            )

    # --------------------------------------------------
    # Convert to arrays
    # --------------------------------------------------

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
        specific_energy,
        semi_major_axis,
        eccentricity,
    ) = _compute_orbital_quantities(
        positions=positions_array,
        velocities=velocities_array,
        body=body,
    )

    return MissionDiagnostics(
        times=times_array,
        positions=positions_array,
        velocities=velocities_array,
        radii=radii,
        speeds=speeds,
        specific_energy=specific_energy,
        semi_major_axis=semi_major_axis,
        eccentricity=eccentricity,
        event_types=tuple(
            event_types
        ),
        phase_labels=tuple(
            phase_labels
        ),
    )