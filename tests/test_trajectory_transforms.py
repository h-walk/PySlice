import numpy as np
import pytest

from pyslice import Trajectory


def _trajectory():
    return Trajectory(
        atom_types=np.array([5, 7]),
        positions=np.array(
            [
                [[-1.0, -2.0, 0.5], [6.2, 4.5, 3.2]],
                [[2.5, 1.0, -0.2], [4.9, 7.0, 2.0]],
            ],
            dtype=np.float32,
        ),
        velocities=np.ones((2, 2, 3), dtype=np.float32),
        box_matrix=np.array(
            [
                [5.0, 1.0, 0.0],
                [0.0, 4.0, 0.5],
                [0.0, 0.0, 3.0],
            ],
            dtype=np.float32,
        ),
        timestep=0.005,
    )


def test_to_ase_supports_numeric_types_and_selected_frame():
    trajectory = Trajectory(
        atom_types=np.array([14]),
        positions=np.array([[[0.0, 0.0, 0.0]], [[1.0, 1.0, 1.0]]]),
        velocities=np.array([[[0.0, 0.0, 0.0]], [[2.0, 2.0, 2.0]]]),
        box_matrix=np.diag([3.0, 3.0, 3.0]),
        timestep=0.01,
    )

    atoms = trajectory.to_ase(frame=1)

    assert atoms.get_chemical_symbols() == ["Si"]
    np.testing.assert_allclose(atoms.positions, [[1.0, 1.0, 1.0]])
    np.testing.assert_allclose(atoms.get_velocities(), [[2.0, 2.0, 2.0]])


def test_center_of_mass_drift_is_mass_weighted_and_removable():
    trajectory = Trajectory(
        atom_types=np.array(["H", "He"]),
        positions=np.array(
            [
                [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
                [[1.0, -2.0, 0.5], [3.0, -2.0, 0.5]],
                [[-0.5, 1.0, 1.5], [1.5, 1.0, 1.5]],
            ]
        ),
        velocities=np.array(
            [
                [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
                [[-2.0, 0.5, 1.0], [-2.0, 0.5, 1.0]],
                [[0.25, -1.0, 2.0], [0.25, -1.0, 2.0]],
            ]
        ),
        box_matrix=np.diag([10.0, 10.0, 10.0]),
        timestep=0.01,
    )

    np.testing.assert_allclose(
        trajectory.get_center_of_mass_drift(),
        [[0.0, 0.0, 0.0], [1.0, -2.0, 0.5], [-0.5, 1.0, 1.5]],
    )

    corrected = trajectory.remove_center_of_mass_drift()

    np.testing.assert_allclose(
        corrected.get_center_of_mass(),
        np.broadcast_to(corrected.get_center_of_mass()[0], (3, 3)),
        atol=1e-14,
    )
    masses = corrected.to_ase().get_masses()
    corrected_com_velocity = (
        np.einsum("fai,a->fi", corrected.velocities, masses) / masses.sum()
    )
    np.testing.assert_allclose(corrected_com_velocity, 0.0, atol=1e-14)
    np.testing.assert_allclose(
        corrected.positions[:, 1] - corrected.positions[:, 0],
        trajectory.positions[:, 1] - trajectory.positions[:, 0],
    )
    assert not np.shares_memory(corrected.positions, trajectory.positions)
