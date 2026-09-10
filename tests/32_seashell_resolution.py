"""Tests for the sea-eco resolution layer (pyslice.data.seashell).

The contract: when sea-eco is importable, every PySlice result *is* a sea-eco
container with no conversion call — results are first-class ``Signal``s and a
``Trajectory`` resolves implicitly to an ``atomic-structure`` collection. When
sea-eco is absent, PySlice still imports and simulates, and SEA operations fail
with one actionable message.
"""
import numpy as np
import pytest
from ase.build import bulk

from pyslice.data import seashell
from pyslice.io.loader import Loader

sea_only = pytest.mark.skipif(not seashell.sea_available, reason="sea-eco not installed")


@pytest.fixture()
def trajectory():
    """Single-frame diamond trajectory."""
    return Loader(atoms=bulk("C", "diamond", a=3.567, cubic=True)).load()


def test_layer_is_the_only_sea_eco_import_site():
    """All sea-eco coupling stays in the resolution layer."""
    from pathlib import Path

    source_root = Path(__file__).resolve().parent.parent / "src" / "pyslice"
    offenders = []
    for path in source_root.rglob("*.py"):
        if path.name in ("seashell.py", "NOTsignal.py", "ALSONOTsignal.py"):
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if stripped.startswith(("from pySEA", "import pySEA")):
                offenders.append(f"{path.relative_to(source_root)}: {stripped}")
    assert not offenders, "sea-eco must only be imported by data/seashell.py:\n" + "\n".join(offenders)


@sea_only
def test_results_are_first_class_signals(tmp_path, monkeypatch):
    """WFData/HAADFData/TACAWData are usable Signals straight out of the run."""
    monkeypatch.chdir(tmp_path)
    from pySEA.sea_eco.architecture.base_structure import Signal

    from pyslice.multislice.calculators import MultisliceCalculator
    from pyslice.postprocessing.tacaw_data import TACAWData

    trajectory = Loader(atoms=bulk("C", "diamond", a=3.567, cubic=True)).load()
    from pyslice import Trajectory
    positions = np.repeat(trajectory.positions, 4, axis=0)
    positions[:, 1, 0] += 0.05 * np.sin(2 * np.pi * np.arange(4) / 4)
    trajectory = Trajectory(trajectory.atom_types, positions,
                            np.zeros_like(positions), trajectory.box_matrix, 0.005)
    calculator = MultisliceCalculator(force_cpu=True)
    calculator.setup(trajectory, aperture=20.0, voltage_eV=100e3, sampling=0.05,
                     slice_thickness=1.78, probe_xs=[1.0, 2.0], probe_ys=[1.0, 2.0],
                     ADF=(60, 200), cache_wavefunctions=False)
    wf_data, haadf = calculator.run()
    haadf.calculateADF(60, 200)
    tacaw = TACAWData(wf_data)

    for expected_name, result in [("Wavefunction", wf_data), ("HAADF", haadf), ("TACAW", tacaw)]:
        assert isinstance(result, Signal)
        assert result.name == expected_name
        # state Signal.__init__ would have established
        assert result.Provenance is not None
        assert type(result.Analysis).__name__ == "AnalysisCollection"
        assert result.dimension_signature == result._local_dimensions.get_names()
        # already a container: resolve is the identity, never a conversion
        assert seashell.resolve(result) is result


@sea_only
def test_trajectory_resolves_implicitly_and_caches(trajectory):
    structure = trajectory.sea
    assert type(structure).__name__ == "SignalCollection"
    assert structure.schema_profile == "atomic-structure"
    assert trajectory.sea is structure  # cached
    # a transform yields a new object, which resolves on its own
    tiled = trajectory.tile_positions((2, 1, 1))
    assert tiled.sea is not structure
    assert tiled.sea["atoms"]["position"].data.shape[0] == 2 * trajectory.n_atoms


@sea_only
def test_resolution_needs_no_setup_import():
    """resolve() self-registers the built-in resolvers."""
    seashell._RESOLVERS.clear()
    seashell._BUILTINS_LOADED = False
    trajectory = Loader(atoms=bulk("C", "diamond", a=3.567)).load()
    assert type(seashell.resolve(trajectory)).__name__ == "SignalCollection"


@sea_only
def test_arrays_resolve_to_minimal_signals():
    signal = seashell.resolve(np.zeros((4, 5)), name="probe")
    assert signal.name == "probe"
    assert signal.data.shape == (4, 5)


@sea_only
def test_unknown_type_names_the_extension_point():
    with pytest.raises(TypeError, match="register_resolver"):
        seashell.resolve(object())


def test_register_resolver_extension_point():
    marker = object()
    seashell.register_resolver("_TestOnlyType", lambda obj, **kw: marker)
    assert seashell._RESOLVERS["_TestOnlyType"](None) is marker
    del seashell._RESOLVERS["_TestOnlyType"]


def test_require_sea_message_is_actionable(monkeypatch):
    monkeypatch.setattr(seashell, "sea_available", False)
    with pytest.raises(ImportError, match=r"pyslice\[sea\]"):
        seashell.require_sea("Doing the thing")
