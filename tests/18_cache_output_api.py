import numpy as np
import pytest

from pyslice.backend import to_cpu
from pyslice.multislice.calculators import MultisliceCalculator, checkCache
from pyslice.multislice.trajectory import Trajectory


def test_output_slice_indices_are_normalized_and_validated():
    calc = MultisliceCalculator(force_cpu=True)
    calc.output = "slices"
    calc.nz = 5

    calc.output_slice_indices = [3, 1, 3]
    assert calc._resolve_output_layers() == [1, 3]

    calc.output_slice_indices = [5]
    with pytest.raises(ValueError, match="out-of-range"):
        calc._resolve_output_layers()


def test_selected_output_slice_indices_participate_in_cache_key():
    traj = _make_tiny_trajectory()

    first = MultisliceCalculator(force_cpu=True)
    first.setup(
        traj,
        aperture=5,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        output="slices",
        output_slice_indices=[3, 1, 1],
    )

    second = MultisliceCalculator(force_cpu=True)
    second.setup(
        traj,
        aperture=5,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        output="slices",
        output_slice_indices=[1, 3],
    )

    all_slices = MultisliceCalculator(force_cpu=True)
    all_slices.setup(
        traj,
        aperture=5,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        output="slices",
    )

    assert first._output_layers == [1, 3]
    assert first.cache_key == second.cache_key
    assert all_slices.cache_key != first.cache_key


def test_cached_files_with_mismatched_layer_counts_are_ignored(tmp_path):
    cache_file = tmp_path / "frame_0.npy"
    np.save(cache_file, np.zeros((1, 2, 2, 3, 1), dtype=np.complex128))

    cache_exists, _ = checkCache(cache_file, True, expected_n_layers=2)
    assert not cache_exists

    cache_exists, frame_data = checkCache(cache_file, True, expected_n_layers=3)
    assert cache_exists
    assert to_cpu(frame_data).shape[-2] == 3


def test_exit_output_returns_final_layer_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        output="exit",
    )

    wave = calc.run()

    assert wave.layer.tolist() == [3]
    assert to_cpu(wave.array).shape[-1] == 1


def test_slice_output_returns_selected_layers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        output="slices",
        output_slice_indices=[1, 3],
    )

    wave = calc.run()

    assert wave.layer.tolist() == [1, 3]
    assert to_cpu(wave.array).shape[-1] == 2


def test_cache_potentials_routes_potential_cache(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        cache=False,
        cache_potentials=True,
    )

    calc.run()

    potential_files = sorted(calc.output_dir.glob("potential_0_*.npy"))
    assert potential_files
    assert not (calc.output_dir / "frame_0.npy").exists()


def test_legacy_cache_keywords_report_new_api():
    calc = MultisliceCalculator(force_cpu=True)
    traj = _make_tiny_trajectory()
    legacy_wave_cache = "cache" + "_levels"
    legacy_layer_selection = "cache" + "_layer_indices"
    legacy_materialization = "store" + "_full"

    with pytest.raises(TypeError) as excinfo:
        calc.setup(traj, **{legacy_wave_cache: []})
    message = str(excinfo.value)
    assert legacy_wave_cache in message
    assert "output='exit'" in message
    assert "output='slices'" in message
    assert "cache=True/False" in message
    assert "cache_potentials=True/False" in message
    assert "keep_wavefunctions=False" in message
    assert "Old arguments were not applied" in message

    with pytest.raises(TypeError, match=legacy_layer_selection):
        calc.setup(traj, **{legacy_layer_selection: [1]})

    with pytest.raises(TypeError, match=legacy_materialization):
        calc.setup(traj, **{legacy_materialization: False})


def _make_tiny_trajectory():
    positions = np.array([[[1.5, 1.5, 1.5]]], dtype=float)
    return Trajectory(
        atom_types=np.array([14]),
        positions=positions,
        velocities=np.zeros_like(positions),
        box_matrix=np.diag([4.0, 4.0, 3.0]),
        timestep=0.1,
    )
