import numpy as np
import pytest

from pyslice.backend import to_cpu
from pyslice.multislice.calculators import MultisliceCalculator, checkCache
from pyslice.multislice.trajectory import Trajectory


class _TinyTrajectory:
    n_frames = 2
    n_atoms = 2
    atom_types = np.array([14, 14])
    box_matrix = np.eye(3)
    positions = np.zeros((2, 2, 3))


def test_cache_layer_indices_participate_in_cache_key():
    calc = MultisliceCalculator(force_cpu=True)

    legacy = calc._generate_cache_key(
        _TinyTrajectory,
        aperture=10,
        voltage_eV=60e3,
        slice_thickness=1.0,
        sampling=0.2,
        probe_positions=[(0.0, 0.0)],
        spatial_decoherence=None,
        temporal_decoherence=None,
    )
    subset = calc._generate_cache_key(
        _TinyTrajectory,
        aperture=10,
        voltage_eV=60e3,
        slice_thickness=1.0,
        sampling=0.2,
        probe_positions=[(0.0, 0.0)],
        spatial_decoherence=None,
        temporal_decoherence=None,
        stored_layer_indices=(1, 3),
    )
    explicit_default_layers = calc._generate_cache_key(
        _TinyTrajectory,
        aperture=10,
        voltage_eV=60e3,
        slice_thickness=1.0,
        sampling=0.2,
        probe_positions=[(0.0, 0.0)],
        spatial_decoherence=None,
        temporal_decoherence=None,
        stored_layer_indices=None,
    )

    assert subset != legacy
    assert explicit_default_layers == legacy


def test_resolve_active_layers_validates_indices():
    calc = MultisliceCalculator(force_cpu=True)
    calc.cache_levels = ["slices"]
    calc.nz = 5

    calc.cache_layer_indices = [3, 1, 3]
    assert calc._resolve_active_layers() == [1, 3]

    calc.cache_layer_indices = [5]
    with pytest.raises(ValueError, match="out-of-range"):
        calc._resolve_active_layers()


def test_cache_key_uses_resolved_active_layers():
    traj = _make_tiny_trajectory()

    first = MultisliceCalculator(force_cpu=True)
    first.setup(
        traj,
        aperture=5,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        cache_levels=["slices"],
        cache_layer_indices=[3, 1, 1],
    )

    second = MultisliceCalculator(force_cpu=True)
    second.setup(
        traj,
        aperture=5,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        cache_levels=["slices"],
        cache_layer_indices=[1, 3],
    )

    default = MultisliceCalculator(force_cpu=True)
    default.setup(
        traj,
        aperture=5,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        cache_levels=["slices"],
    )

    assert first._active_layers == [1, 3]
    assert first.cache_key == second.cache_key
    assert default.cache_key != first.cache_key


def test_selective_layer_cache_run_has_expected_layers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        cache_levels=["slices"],
        cache_layer_indices=[1, 3],
    )

    wave = calc.run()

    assert wave.layer.tolist() == [1, 3]
    assert to_cpu(wave.array).shape[-1] == 2


def test_cache_rejects_mismatched_layer_count(tmp_path):
    cache_file = tmp_path / "frame_0.npy"
    np.save(cache_file, np.zeros((1, 2, 2, 3, 1), dtype=np.complex128))

    cache_exists, _ = checkCache(cache_file, ["slices"], expected_n_layers=2)
    assert not cache_exists

    cache_exists, frame_data = checkCache(cache_file, ["slices"], expected_n_layers=3)
    assert cache_exists
    assert to_cpu(frame_data).shape[-2] == 3


def _make_tiny_trajectory():
    positions = np.array([[[1.5, 1.5, 1.5]]], dtype=float)
    return Trajectory(
        atom_types=np.array([14]),
        positions=positions,
        velocities=np.zeros_like(positions),
        box_matrix=np.diag([4.0, 4.0, 3.0]),
        timestep=0.1,
    )
