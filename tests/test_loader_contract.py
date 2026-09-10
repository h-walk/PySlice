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
