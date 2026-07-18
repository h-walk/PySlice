import numpy as np

from pyslice.multislice.trajectory import Trajectory


def make_trajectory():
    positions = np.arange(5 * 2 * 3, dtype=float).reshape(5, 2, 3)
    velocities = positions + 100.0
    return Trajectory(
        atom_types=np.array(["Si", "Si"]),
        positions=positions,
        velocities=velocities,
        box_matrix=np.diag([10.0, 10.0, 10.0]),
        timestep=0.25,
    )


def test_slice_timesteps_default_starts_at_first_frame():
    trajectory = make_trajectory()

    sliced = trajectory.slice_timesteps()

    assert sliced.n_frames == trajectory.n_frames
    np.testing.assert_array_equal(sliced.positions[0], trajectory.positions[0])
    assert sliced.timestep == trajectory.timestep


def test_select_timesteps_preserves_frame_axis_for_scalar_and_tuple():
    trajectory = make_trajectory()

    scalar = trajectory.select_timesteps(2)
    assert scalar.positions.shape == (1, trajectory.n_atoms, 3)
    np.testing.assert_array_equal(scalar.positions[0], trajectory.positions[2])
    assert scalar.timestep == 0.0

    tuple_selected = trajectory.select_timesteps((1, 3))
    assert tuple_selected.positions.shape == (2, trajectory.n_atoms, 3)
    np.testing.assert_array_equal(tuple_selected.positions[0], trajectory.positions[1])
    np.testing.assert_array_equal(tuple_selected.positions[1], trajectory.positions[3])


def test_random_frame_helpers_are_reproducible():
    trajectory = make_trajectory()

    frames_a = trajectory.random_frames(3, seed=11)
    frames_b = trajectory.random_frames(3, seed=11)
    legacy = trajectory.get_random_timesteps(3, seed=11)

    np.testing.assert_array_equal(frames_a.positions, frames_b.positions)
    np.testing.assert_array_equal(frames_a.positions, legacy.positions)
    assert frames_a.positions.shape == (3, trajectory.n_atoms, 3)


def test_generate_random_displacements_uses_local_rng():
    trajectory = make_trajectory()

    np.random.seed(123)
    first_global = np.random.random()
    displaced_a = trajectory.generate_random_displacements(4, sigma=0.1, seed=5)
    second_global = np.random.random()

    np.random.seed(123)
    assert first_global == np.random.random()
    assert second_global == np.random.random()

    displaced_b = trajectory.generate_random_displacements(4, sigma=0.1, seed=5)
    assert displaced_a.positions.shape == (4, trajectory.n_atoms, 3)
    np.testing.assert_array_equal(displaced_a.positions, displaced_b.positions)
    np.testing.assert_array_equal(
        displaced_a.velocities,
        np.broadcast_to(trajectory.velocities[0], displaced_a.velocities.shape),
    )
