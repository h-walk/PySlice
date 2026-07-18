"""Loader regressions: OVITO conventions, element mapping, indexing."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyslice.backend import NumpyBackend, TORCH_AVAILABLE, to_numpy
if TORCH_AVAILABLE:
    from pyslice.backend import TorchBackend
from pyslice.multislice import calculators as calculators_module
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.multislice.multislice import PrismProbe, Probe, Propagate, wavelength
from pyslice.multislice.potentials import Potential
from pyslice.multislice.trajectory import Trajectory
from pyslice.io.loader import Loader
from pyslice.postprocessing.haadf_data import HAADFData
from pyslice.postprocessing.tacaw_data import TACAWData
from pyslice.postprocessing.wf_data import WFData



from _regression_helpers import *  # shared builders (see __all__)


def test_ovito_type_ids_are_never_used_as_atomic_numbers(tmp_path):
    # LAMMPS type IDs must not be treated as Z. Element identity comes from an
    # explicit mapping or the file's embedded element names; otherwise abort.
    ids = np.array([1, 1, 2])

    # embedded element names present -> mapped to element symbols
    resolved = _loader(tmp_path)._resolve_ovito_types(ids, {1: "Si", 2: "O"})
    assert list(resolved) == ["Si", "Si", "O"]

    # no names and no mapping -> abort asking for an exact mapping
    with pytest.raises(ValueError, match="atom_mapping"):
        _loader(tmp_path)._resolve_ovito_types(ids, {})
    # a numeric/junk type name is NOT accepted as an element
    with pytest.raises(ValueError, match="atom_mapping"):
        _loader(tmp_path)._resolve_ovito_types(np.array([1]), {1: "1"})

    # explicit mapping wins and yields atomic numbers
    resolved = _loader(tmp_path, atom_mapping={1: "Si", 2: "O"})._resolve_ovito_types(ids, {})
    assert list(resolved) == [14, 14, 8]

    # explicit mapping must be exact: a missing type aborts
    with pytest.raises(ValueError, match="missing entries"):
        _loader(tmp_path, atom_mapping={1: "Si"})._resolve_ovito_types(ids, {2: "O"})

def test_ovito_cell_matrix_transposed_to_row_convention():
    from pyslice.io.loader import _ovito_cell_to_row_convention
    a = np.array([10.0, 0.0, 0.0]); b = np.array([2.0, 10.0, 0.0])
    c = np.array([1.0, 1.5, 10.0]); origin = np.array([-5.0, 0.0, 0.0])
    ovito_matrix = np.column_stack([a, b, c, origin])  # vectors in columns, origin last
    box, got_origin = _ovito_cell_to_row_convention(ovito_matrix)
    # rows are the lattice vectors (ASE / PySlice convention), not columns
    np.testing.assert_allclose(box[0], a)
    np.testing.assert_allclose(box[1], b)
    np.testing.assert_allclose(box[2], c)
    np.testing.assert_allclose(got_origin, origin)
    # a plain orthorhombic zero-origin cell is unchanged by the transpose
    orth = np.column_stack([np.diag([8.0, 6.0, 4.0]), np.zeros(3)])
    box2, o2 = _ovito_cell_to_row_convention(orth)
    np.testing.assert_allclose(box2, np.diag([8.0, 6.0, 4.0]))
    np.testing.assert_allclose(o2, np.zeros(3))

def test_loader_parse_index_forms():
    from pyslice.io.loader import _parse_index
    assert _parse_index(":") == slice(None, None, None)
    assert _parse_index(":3") == slice(None, 3, None)
    assert _parse_index("-3:") == slice(-3, None, None)
    assert _parse_index("::2") == slice(None, None, 2)
    assert _parse_index("3:5") == slice(3, 5, None)
    assert _parse_index("3-5") == slice(3, 6, None)   # inclusive dash range
    assert _parse_index("1") == 1
    assert _parse_index(-1) == -1
    import pytest as _pytest
    with _pytest.raises(ValueError):
        _parse_index("nonsense")

def test_multiframe_cif_loads_all_frames_and_index_selects(tmp_path):
    ase_io = pytest.importorskip("ase.io")
    from ase import Atoms
    frames = [Atoms("H", positions=[[float(i), 0.0, 0.0]], cell=[10, 10, 10], pbc=True)
              for i in range(6)]
    cif = tmp_path / "traj.cif"
    ase_io.write(str(cif), frames)
    from pyslice.io.loader import Loader

    def load(index):
        return Loader(str(cif), timestep=0.5, index=index).load()

    # default reads every image (previously only the last one survived)
    full = load(":")
    assert full.n_frames == 6
    np.testing.assert_allclose(full.positions[:, 0, 0], np.arange(6))
    # selectors
    assert load(":3").n_frames == 3
    np.testing.assert_allclose(load("-3:").positions[:, 0, 0], [3, 4, 5])
    np.testing.assert_allclose(load("3-5").positions[:, 0, 0], [3, 4, 5])  # inclusive
    strided = load("::2")
    np.testing.assert_allclose(strided.positions[:, 0, 0], [0, 2, 4])
    assert strided.timestep == 1.0                    # step rescales timestep
    assert load(1).n_frames == 1

def test_loader_cache_tracks_source_parser_and_mapping_inputs(tmp_path, monkeypatch):
    source = tmp_path / "trajectory.fake"
    source.write_text("1")
    calls = []

    def fake_parse(self):
        calls.append((source.read_text(), self.atomic_numbers))
        x = float(source.read_text())
        positions = np.array([[[x, 0.0, 0.0]]], dtype=np.float32)
        atom_type = 14 if self.atomic_numbers is None else self.atomic_numbers[1]
        return Trajectory(
            atom_types=np.array([atom_type]), positions=positions,
            velocities=np.zeros_like(positions), box_matrix=np.eye(3) * 10,
            timestep=self.timestep)

    monkeypatch.setattr(Loader, "_load_via_ovito", fake_parse)
    assert Loader(str(source)).load().positions[0, 0, 0] == 1
    assert Loader(str(source)).load().positions[0, 0, 0] == 1
    assert len(calls) == 1                              # unchanged cache hit

    source.write_text("2")
    assert Loader(str(source)).load().positions[0, 0, 0] == 2
    assert len(calls) == 2                              # source invalidated

    mapped = Loader(str(source), atom_mapping={1: "C"}).load()
    assert mapped.atom_types.tolist() == [6]
    assert len(calls) == 3                              # mapping invalidated
    assert Loader(str(source), atom_mapping={1: 6}).load().atom_types.tolist() == [6]
    assert len(calls) == 3                              # canonical mapping cache hit

    # Legacy existence-only caches have no provenance and must be reparsed.
    Loader(str(source))._get_cache_files()["metadata"].unlink()
    Loader(str(source)).load()
    assert len(calls) == 4
