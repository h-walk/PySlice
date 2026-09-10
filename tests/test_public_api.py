"""Contract tests for the curated top-level package API."""

import pyslice


def test_declared_public_api_is_available():
    assert isinstance(pyslice.__all__, tuple)
    assert len(pyslice.__all__) == len(set(pyslice.__all__))
    for name in pyslice.__all__:
        assert hasattr(pyslice, name), name


def test_primary_workflow_is_available_from_top_level():
    expected = {
        "Loader",
        "Trajectory",
        "MultisliceCalculator",
        "WFData",
        "TACAWData",
        "HAADFData",
        "ORBMDCalculator",
        "FAIRChemMDCalculator",
    }
    assert expected <= set(pyslice.__all__)


def test_common_wildcard_import_leaks_are_absent():
    for name in ("np", "Path", "hashlib", "Optional"):
        assert not hasattr(pyslice, name)
