import numpy as np
import pytest

from pyslice.backend import to_numpy
from pyslice.multislice.calculators import MultisliceCalculator, checkCache
from pyslice.multislice.trajectory import Trajectory


def test_return_layers_are_normalized_and_validated():
    calc = MultisliceCalculator(force_cpu=True)
    calc.nz = 5

    calc.return_layers = [3, 1, 3]
    assert calc._resolve_return_layers() == [1, 3]

    calc.return_layers = -1
    assert calc._resolve_return_layers() == [4]

    calc.return_layers = "all"
    assert calc._resolve_return_layers() == [0, 1, 2, 3, 4]

    calc.return_layers = None
    assert calc._resolve_return_layers() == []

    calc.return_layers = []
    assert calc._resolve_return_layers() == []

    calc.return_layers = [5]
    with pytest.raises(ValueError, match="out-of-range"):
        calc._resolve_return_layers()


def test_selected_return_layers_participate_in_cache_key():
    traj = _make_tiny_trajectory()

    first = MultisliceCalculator(force_cpu=True)
    first.setup(
        traj,
        aperture=5,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        return_layers=[3, 1, 1],
    )

    second = MultisliceCalculator(force_cpu=True)
    second.setup(
        traj,
        aperture=5,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        return_layers=[1, 3],
    )

    all_layers = MultisliceCalculator(force_cpu=True)
    all_layers.setup(
        traj,
        aperture=5,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        return_layers="all",
    )

    no_return = MultisliceCalculator(force_cpu=True)
    no_return.setup(
        traj,
        aperture=5,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        return_layers=None,
    )

    exit_wave = MultisliceCalculator(force_cpu=True)
    exit_wave.setup(
        traj,
        aperture=5,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
    )

    assert first._return_layers == [1, 3]
    assert first.cache_key == second.cache_key
    assert all_layers.cache_key != first.cache_key
    assert no_return.cache_key == exit_wave.cache_key


def test_setup_defocus_applies_probe_defocus_and_partitions_cache():
    traj = _make_tiny_trajectory()

    from_setup = MultisliceCalculator(force_cpu=True)
    from_setup.setup(
        traj,
        aperture=20,
        voltage_eV=100e3,
        sampling=0.5,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        defocus=250.0,
    )

    manual = MultisliceCalculator(force_cpu=True)
    manual.setup(
        traj,
        aperture=20,
        voltage_eV=100e3,
        sampling=0.5,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
    )
    manual.base_probe.defocus(250.0)

    focused = MultisliceCalculator(force_cpu=True)
    focused.setup(
        traj,
        aperture=20,
        voltage_eV=100e3,
        sampling=0.5,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
    )

    np.testing.assert_allclose(
        to_numpy(from_setup.base_probe._array),
        to_numpy(manual.base_probe._array),
    )
    assert from_setup.cache_key != focused.cache_key


def test_setup_defocus_rejects_prism_path():
    calc = MultisliceCalculator(force_cpu=True)

    with pytest.raises(NotImplementedError, match="PRISM defocus"):
        calc.setup(
            _make_tiny_trajectory(),
            aperture=20,
            voltage_eV=100e3,
            sampling=0.5,
            slice_thickness=1.0,
            probe_positions=[(2.0, 2.0)],
            prism=2,
            defocus=250.0,
        )


def test_cached_files_with_mismatched_layer_counts_are_ignored(tmp_path):
    cache_file = tmp_path / "frame_0.npy"
    np.save(cache_file, np.zeros((1, 2, 2, 3, 1), dtype=np.complex128))
    backend = MultisliceCalculator(force_cpu=True)._backend

    cache_exists, _ = checkCache(cache_file, True, backend, expected_n_layers=2)
    assert not cache_exists

    cache_exists, frame_data = checkCache(cache_file, True, backend, expected_n_layers=3)
    assert cache_exists
    assert to_numpy(frame_data).shape[-2] == 3


def test_default_return_layers_returns_final_layer_metadata(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
    )

    wave = calc.run()

    assert wave.layer.tolist() == [3]
    assert to_numpy(wave.array).shape[-1] == 1


def test_return_layers_returns_selected_layers(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        return_layers=[1, 3],
    )

    wave = calc.run()

    assert wave.layer.tolist() == [1, 3]
    assert to_numpy(wave.array).shape[-1] == 2


def test_none_return_layers_suppresses_wavefunction_return(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    calc = MultisliceCalculator(force_cpu=True)
    calc.setup(
        _make_tiny_trajectory(),
        aperture=5,
        voltage_eV=60e3,
        sampling=1.0,
        slice_thickness=1.0,
        probe_positions=[(2.0, 2.0)],
        return_layers=None,
        cache_wavefunctions=False,
    )

    wave = calc.run()

    assert wave.layer.tolist() == []
    assert to_numpy(wave.array).shape[-1] == 0
    assert not (calc.output_dir / "frame_0.npy").exists()


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
        return_layers=None,
        cache_wavefunctions=False,
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
    intermediate_output = "output"
    intermediate_cache = "cache"

    with pytest.raises(TypeError) as excinfo:
        calc.setup(traj, **{legacy_wave_cache: []})
    message = str(excinfo.value)
    assert legacy_wave_cache in message
    assert "return_layers=-1" in message
    assert "return_layers='all'" in message
    assert "cache_wavefunctions=True/False" in message
    assert "cache_potentials=True/False" in message
    assert "return_layers=None" in message
    assert "return_layers=[]" in message
    assert "Old arguments were not applied" in message

    with pytest.raises(TypeError, match=legacy_layer_selection):
        calc.setup(traj, **{legacy_layer_selection: [1]})

    with pytest.raises(TypeError, match=legacy_materialization):
        calc.setup(traj, **{legacy_materialization: False})

    with pytest.raises(TypeError, match=intermediate_output):
        calc.setup(traj, **{intermediate_output: "slices"})

    with pytest.raises(TypeError, match=intermediate_cache):
        calc.setup(traj, **{intermediate_cache: False})


def _make_tiny_trajectory():
    positions = np.array([[[1.5, 1.5, 1.5]]], dtype=float)
    return Trajectory(
        atom_types=np.array([14]),
        positions=positions,
        velocities=np.zeros_like(positions),
        box_matrix=np.diag([4.0, 4.0, 3.0]),
        timestep=0.1,
    )
