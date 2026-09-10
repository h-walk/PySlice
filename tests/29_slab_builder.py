"""Tests for pyslice.io.build (ASE-backed periodic slab construction).

Covers zone-axis slabs, in-plane orthogonalization of hexagonal cells,
thickness→layers conversion, lateral repeats, vacuum, the diagonal-box
guarantee, and the reciprocal-lattice helpers used by the planner.
"""
import numpy as np
import pytest
from ase.build import bulk, graphene

from pyslice.io.build import (
    atom_symbols,
    build_slab,
    first_bragg_g,
    orthogonal_supercell_matrix,
    trajectory_to_ase,
)


@pytest.fixture()
def diamond():
    """Primitive diamond-carbon cell."""
    return bulk("C", "diamond", a=3.567)


def test_diamond_110_slab_is_orthogonal_and_dense(diamond):
    slab, record = build_slab(diamond, (1, 1, 0), thickness_A=40.0, min_lateral_A=15.0)

    box = slab.box_matrix
    assert np.allclose(box, np.diag(np.diag(box)))
    # thickness rounded UP to whole layers
    assert record["thickness_A"] >= 40.0
    assert record["layers"] == int(np.ceil(40.0 / record["layer_height_A"]))
    # exact crystal density: diamond has 8 atoms per a^3
    expected = 8 * np.prod(record["box_A"]) / 3.567**3
    assert slab.n_atoms == pytest.approx(expected)
    assert min(record["box_A"][0], record["box_A"][1]) >= 15.0


def test_layers_override_thickness(diamond):
    _, record = build_slab(diamond, (1, 1, 0), thickness_A=40.0, layers=3)
    assert record["layers"] == 3


def test_graphene_orthogonalization_and_vacuum():
    sheet = graphene(a=2.46)
    sheet.cell[2, 2] = 6.7
    slab, record = build_slab(sheet, (0, 0, 1), layers=1, min_lateral_A=20.0, vacuum_A=10.0)

    box = slab.box_matrix
    assert np.allclose(box, np.diag(np.diag(box)))
    # the orthorhombic graphene cell is a x a*sqrt(3): area per orbit preserved
    assert not np.array_equal(record["orthogonal_supercell"], np.eye(3).tolist())
    assert record["box_A"][2] == pytest.approx(6.7 + 10.0)
    # vacuum splits evenly: the sheet sits mid-box
    z = slab.positions[0][:, 2]
    assert z.min() == pytest.approx(5.0, abs=1e-5)
    # atom density per area is preserved by the supercell (box_A is rounded
    # to 4 decimals in the record, so compare loosely)
    area = record["box_A"][0] * record["box_A"][1]
    assert slab.n_atoms == pytest.approx(2 * area / (2.46 * 2.46 * np.sqrt(3) / 2), rel=1e-4)


def test_orthogonal_supercell_matrix_identity_for_cubic():
    assert np.array_equal(orthogonal_supercell_matrix(np.diag([3.0, 3.0, 3.0])), np.eye(3, dtype=int))


def test_orthogonal_supercell_matrix_rejects_irrational_cell():
    # equal-length oblique vectors are ALWAYS orthogonalizable via a±b, so
    # use an anisotropic oblique cell with no small integer orthogonal combo
    cell = np.array([[1.0, 0.0, 0.0], [0.4, 1.3, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="rotate_to"):
        orthogonal_supercell_matrix(cell, max_index=3)


def test_first_bragg_g_in_plane_ignores_vacuum_axis():
    sheet = graphene(a=2.46)
    sheet.cell[2, 2] = 6.7
    cell = np.array(sheet.get_cell())
    g_in_plane = first_bragg_g(cell)
    # hexagonal d_100 = a*sqrt(3)/2
    assert 1.0 / g_in_plane == pytest.approx(2.46 * np.sqrt(3) / 2, rel=1e-6)
    assert first_bragg_g(cell, in_plane=False) == pytest.approx(1.0 / 6.7, rel=1e-6)


def test_atom_symbols_normalizes_ints_and_strings(diamond):
    from pyslice.io.loader import Loader

    trajectory = Loader(atoms=diamond).load()
    assert atom_symbols(trajectory) == ["C", "C"]
    trajectory.atom_types = np.array([6, 6])
    assert atom_symbols(trajectory) == ["C", "C"]
    ase_atoms = trajectory_to_ase(trajectory)
    assert ase_atoms.get_chemical_symbols() == ["C", "C"]
    trajectory.atom_types = np.array([0, 6])
    with pytest.raises(ValueError, match="atom_mapping"):
        atom_symbols(trajectory)
