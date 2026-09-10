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


def test_fold_positions_to_orthogonal_box_wraps_selected_axes():
    folded = _trajectory().fold_positions_to_orthogonal_box(axes=(0,))

    np.testing.assert_allclose(folded.box_matrix, np.diag([5.0, 4.0, 3.0]))
    np.testing.assert_allclose(folded.positions[:, :, 0], [[4.0, 1.2], [2.5, 4.9]], atol=1e-6)
    np.testing.assert_allclose(folded.positions[:, :, 1:], _trajectory().positions[:, :, 1:])
    assert folded.timestep == 0.005


def test_fold_positions_to_orthogonal_box_validates_inputs():
    trajectory = _trajectory()

    with pytest.raises(ValueError, match="axes"):
        trajectory.fold_positions_to_orthogonal_box(axes=(3,))

    with pytest.raises(ValueError, match="lengths"):
        trajectory.fold_positions_to_orthogonal_box(lengths=(5.0, 4.0))

    with pytest.raises(ValueError, match="Cannot build"):
        trajectory.fold_positions_to_orthogonal_box(lengths=(5.0, 0.0, 3.0))


def test_slice_positions_shifts_coordinates_into_new_box():
    trajectory = Trajectory(
        atom_types=np.array([5, 7]),
        positions=np.array([[[5.5, 1.0, 1.0], [8.5, 1.0, 1.0]]]),
        velocities=np.zeros((1, 2, 3)),
        box_matrix=np.diag([10.0, 4.0, 3.0]),
        timestep=0.005,
    )

    cropped = trajectory.slice_positions(x_range=(5.0, 9.0))

    np.testing.assert_allclose(cropped.positions[0, :, 0], [0.5, 3.5])
    assert cropped.box_matrix[0, 0] == pytest.approx(4.0)


def test_tilt_positions_rotates_cell_with_atoms():
    trajectory = _trajectory()
    tilted = trajectory.tilt_positions(alpha=0.2, beta=-0.1)

    assert not np.allclose(tilted.box_matrix, trajectory.box_matrix)
    np.testing.assert_allclose(
        np.linalg.norm(tilted.box_matrix, axis=1),
        np.linalg.norm(trajectory.box_matrix, axis=1),
    )


def test_tile_positions_rejects_wrong_trajectory_count():
    trajectory = _trajectory()
    with pytest.raises(ValueError, match="one entry per tile"):
        trajectory.tile_positions((2, 1, 1), trajectories=[trajectory])


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


def test_trajectory_rejects_nonfinite_data_and_invalid_sampling_requests():
    trajectory = _trajectory()
    invalid_positions = trajectory.positions.copy()
    invalid_positions[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        Trajectory(
            atom_types=trajectory.atom_types,
            positions=invalid_positions,
            velocities=trajectory.velocities,
            box_matrix=trajectory.box_matrix,
            timestep=trajectory.timestep,
        )

    with pytest.raises(ValueError, match="between 1"):
        trajectory.random_frames(trajectory.n_frames + 1)
    with pytest.raises(ValueError, match="sigma"):
        trajectory.generate_random_displacements(2, sigma=-0.1)
