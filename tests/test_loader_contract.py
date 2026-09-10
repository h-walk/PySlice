from pathlib import Path

import numpy as np
import pytest
from ase.build import bulk

from pyslice import Loader, Trajectory


def _trajectory():
    positions = np.array([[[1.0, 1.0, 1.0]]])
    return Trajectory(
        atom_types=np.array([14]),
        positions=positions,
        velocities=np.zeros_like(positions),
        box_matrix=np.diag([3.0, 3.0, 3.0]),
        timestep=0.01,
    )


def test_loader_cache_manifest_invalidates_changed_source(tmp_path):
    source = tmp_path / "trajectory.dump"
    source.write_text("first")
    loader = Loader(source, timestep=0.01, atom_mapping={1: "Si"})
    loader._save_to_cache(_trajectory())

    assert loader._load_from_cache() is not None
    source.write_text("changed source contents")
    assert loader._load_from_cache() is None


def test_deprecated_loader_mapping_is_applied(tmp_path):
    source = tmp_path / "trajectory.dump"
    source.write_text("placeholder")

    with pytest.warns(DeprecationWarning, match="atom_mapping"):
        loader = Loader(source, atomic_numbers={1: 14})

    assert loader.atomic_numbers == {1: 14}


def test_ase_variable_cell_trajectory_is_rejected():
    first = bulk("Si", "diamond", a=5.431, cubic=True)
    second = first.copy()
    second.set_cell(first.cell * 1.01, scale_atoms=True)

    with pytest.raises(ValueError, match="Variable-cell"):
        Loader(atoms=[first, second], timestep=0.01).load()


@pytest.mark.parametrize("variable_cell", [False, True])
def test_ovito_validates_fixed_triclinic_cells_in_row_convention(
    tmp_path, monkeypatch, variable_cell
):
    import sys
    from types import SimpleNamespace

    source = tmp_path / "tilted.dump"
    source.write_text("mock OVITO input")
    cell = np.array([[3., 0., 0.], [1., 4., 0.], [0.5, 1., 5.]])
    origin = np.array([1., 2., 3.])
    frames = []
    for i in range(2):
        frame_cell = cell * (1.01 if variable_cell and i else 1)
        frames.append(SimpleNamespace(
            cell=SimpleNamespace(matrix=np.column_stack((frame_cell.T, origin))),
            particles=SimpleNamespace(
                positions=np.array([[1.5, 2.5, 3.5]]),
                particle_types=np.array([1]),
            ),
        ))
    pipeline = SimpleNamespace(source=SimpleNamespace(num_frames=2),
                               modifiers=[], compute=lambda i: frames[i])
    monkeypatch.setitem(sys.modules, "ovito.io", SimpleNamespace(
        import_file=lambda *_args, **_kwargs: pipeline))
    monkeypatch.setitem(sys.modules, "ovito.modifiers", SimpleNamespace(
        UnwrapTrajectoriesModifier=lambda: None))
    loader = Loader(source, atom_mapping={1: "Si"})
    if variable_cell:
        with pytest.raises(RuntimeError, match="Variable-cell"):
            loader._load_via_ovito_direct()
    else:
        result = loader._load_via_ovito_direct()
        np.testing.assert_allclose(result.box_matrix, cell)
        np.testing.assert_allclose(result.positions, np.full((2, 1, 3), 0.5))
