"""Tests for pyslice.io.databases (Materials Project + COD clients).

Network calls are mocked; set ``PYSLICE_DB_LIVE_TESTS=1`` to also run the
keyless live COD round-trip on a machine with open egress.
"""
import os

import pytest

import pyslice.io.databases as databases
from pyslice.io.databases import (
    DatabaseError,
    _hill_formula,
    _parse_formula,
    _structure_dict_to_cif,
    fetch_cif,
    search_structures,
)

MP_STRUCTURE = {
    "lattice": {
        "a": 3.867, "b": 3.867, "c": 3.867,
        "alpha": 60.0, "beta": 60.0, "gamma": 60.0,
        "matrix": [[3.3489, 0.0, 1.9335], [1.1163, 3.1574, 1.9335], [0.0, 0.0, 3.867]],
    },
    "sites": [
        {"species": [{"element": "Si", "occu": 1}], "abc": [0.875, 0.875, 0.875]},
        {"species": [{"element": "Si", "occu": 1}], "abc": [0.125, 0.125, 0.125]},
    ],
}


def test_parse_formula_counts():
    assert _parse_formula("SiO2") == {"Si": 1, "O": 2}
    assert _parse_formula("Ba Ti O3") == {"Ba": 1, "Ti": 1, "O": 3}
    with pytest.raises(ValueError, match="Cannot parse"):
        _parse_formula("Si(O)2")


def test_hill_formula_ordering():
    assert _hill_formula("SiO2") == "O2 Si"
    assert _hill_formula("C2H6O") == "C2 H6 O"
    assert _hill_formula("BN") == "B N"
    assert _hill_formula("H2O") == "H2 O"


def test_search_requires_a_filter():
    with pytest.raises(ValueError, match="formula"):
        search_structures("cod")


def test_mp_requires_api_key(monkeypatch):
    for var in ("PYSLICE_MP_API_KEY", "MP_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(DatabaseError, match="PYSLICE_MP_API_KEY"):
        search_structures("mp", formula="Si")


def test_mp_search_normalizes_and_sorts(monkeypatch):
    monkeypatch.setenv("MP_API_KEY", "test-key")
    captured = {}

    def fake_get_json(url, headers=None, timeout=30.0):
        captured["url"] = url
        captured["headers"] = headers
        return {"data": [
            {"material_id": "mp-2", "formula_pretty": "Si", "symmetry": {"symbol": "P1"},
             "nsites": 4, "energy_above_hull": 0.4},
            {"material_id": "mp-149", "formula_pretty": "Si", "symmetry": {"symbol": "Fd-3m", "crystal_system": "Cubic"},
             "nsites": 2, "energy_above_hull": 0.0},
        ]}

    monkeypatch.setattr(databases, "_http_get_json", fake_get_json)
    entries = search_structures("mp", formula="Si", limit=5)
    assert captured["headers"]["X-API-KEY"] == "test-key"
    assert "formula=Si" in captured["url"]
    assert [e["id"] for e in entries] == ["mp-149", "mp-2"]  # stable first
    assert entries[0]["spacegroup"] == "Fd-3m"


def test_cod_search_uses_hill_formula_and_strict_elements(monkeypatch):
    captured = {}

    def fake_get_json(url, headers=None, timeout=30.0):
        captured["url"] = url
        return [{"file": 12345, "formula": "- O2 Si -", "sg": "P 1", "a": "4.9", "b": "4.9", "c": "5.4"}]

    monkeypatch.setattr(databases, "_http_get_json", fake_get_json)
    entries = search_structures("cod", formula="SiO2", elements=["Si", "O"], limit=3)
    assert "formula=O2+Si" in captured["url"]
    assert "el1=Si" in captured["url"] and "el2=O" in captured["url"]
    assert "strictmax=2" in captured["url"]
    assert entries[0]["id"] == "12345"
    assert entries[0]["formula"] == "O2 Si"


def test_fetch_cif_cod_writes_file(monkeypatch, tmp_path):
    monkeypatch.setattr(databases, "_http_get", lambda url, headers=None, timeout=30.0: b"data_test\n_cell_length_a 1.0\n")
    path = fetch_cif("cod", "12345", output_dir=tmp_path)
    assert path.name == "cod_12345.cif"
    assert path.read_text().startswith("data_test")


def test_fetch_cif_cod_rejects_non_numeric_id(tmp_path):
    with pytest.raises(DatabaseError, match="numeric"):
        fetch_cif("cod", "mp-149", output_dir=tmp_path)


def test_mp_structure_dict_renders_loadable_p1_cif(tmp_path):
    cif_text = _structure_dict_to_cif(MP_STRUCTURE, "mp_149")
    assert "'P 1'" in cif_text
    path = tmp_path / "mp_149.cif"
    path.write_text(cif_text)

    from pyslice.io.loader import Loader

    trajectory = Loader(filename=str(path)).load()
    assert trajectory.n_atoms == 2
    assert sorted(set(trajectory.atom_types)) == ["Si"]


def test_fetch_cif_mp_via_mocked_api(monkeypatch, tmp_path):
    monkeypatch.setenv("MP_API_KEY", "test-key")
    monkeypatch.setattr(
        databases, "_http_get_json",
        lambda url, headers=None, timeout=30.0: {"data": [{"material_id": "mp-149", "structure": MP_STRUCTURE}]},
    )
    path = fetch_cif("mp", "mp-149", output_dir=tmp_path)
    assert path.name == "mp_mp_149.cif"
    assert "_cell_length_a" in path.read_text()


@pytest.mark.skipif(not os.environ.get("PYSLICE_DB_LIVE_TESTS"), reason="set PYSLICE_DB_LIVE_TESTS=1 for live COD access")
def test_live_cod_round_trip(tmp_path):
    entries = search_structures("cod", elements=["Ti", "O"], limit=3)
    assert entries, "COD returned no Ti-O entries"
    from pyslice.io.databases import load_structure_from_database

    trajectory = load_structure_from_database("cod", entries[0]["id"], output_dir=tmp_path)
    assert trajectory.n_atoms > 0
