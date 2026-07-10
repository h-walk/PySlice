import numpy as np
import pytest

from pyslice.backend import NumpyBackend
from pyslice.multislice.multislice import Probe
from pyslice.postprocessing.haadf_data import HAADFData
from pyslice.postprocessing.tacaw_data import TACAWData
from pyslice.postprocessing.wf_data import WFData


pysea_base = pytest.importorskip("pySEA.sea_eco.architecture.base_structure")


def _make_wf_data(tmp_path):
    backend = NumpyBackend()
    xs = np.linspace(0.0, 2.0, 2, endpoint=False)
    ys = np.linspace(0.0, 2.0, 2, endpoint=False)
    probe = Probe(
        xs,
        ys,
        mrad=0,
        eV=100e3,
        backend=backend,
        probe_positions=[(1.0, 1.0)],
        defer_shifts=True,
    )

    return WFData(
        probe_positions=[(1.0, 1.0)],
        probe_xs=[1.0],
        probe_ys=[1.0],
        time=np.array([0.0, 0.005]),
        kxs=np.array([-0.5, 0.0]),
        kys=np.array([-0.5, 0.0]),
        xs=xs,
        ys=ys,
        layer=np.array([0]),
        array=np.ones((1, 2, 2, 2, 1), dtype=np.complex128),
        probe=probe,
        backend=backend,
        cache_dir=tmp_path,
    )


def _record_operations(signal):
    return [record.operation for record in signal.Analysis.processing_records]


def test_wfdata_uses_real_sea_eco_signal(tmp_path):
    wf = _make_wf_data(tmp_path)

    assert isinstance(wf, pysea_base.Signal)
    assert wf.signal_type == "Diffraction"
    assert wf.dimensions.get_names() == ["probe", "time", "kx", "ky", "layer"]
    assert wf.metadata.General.title == "Multislice Wavefunction"
    assert hasattr(wf, "Analysis")
    assert "WFData.__init__" in _record_operations(wf)


def test_wfdata_sea_roundtrip_preserves_signal_and_provenance(tmp_path):
    wf = _make_wf_data(tmp_path)
    path = tmp_path / "wavefunction.sea"

    wf.to_sea(path)
    loaded = WFData.load(path)

    assert isinstance(loaded, pysea_base.Signal)
    assert loaded.Provenance.seaid == wf.Provenance.seaid
    assert loaded.dimensions.get_names() == ["probe", "time", "kx", "ky", "layer"]
    np.testing.assert_allclose(loaded.data, wf.data)
    assert "WFData.__init__" in _record_operations(loaded)


def test_wfdata_records_in_place_processing(tmp_path):
    wf = _make_wf_data(tmp_path)

    wf.applyMask(radius=0.75)

    assert "WFData.applyMask" in _record_operations(wf)
    assert hasattr(wf.metadata, "Processing")


def test_tacawdata_records_wfdata_parent(tmp_path):
    wf = _make_wf_data(tmp_path)

    tacaw = TACAWData(wf, force_rerun=True)

    assert isinstance(tacaw, pysea_base.Signal)
    assert "TACAWData.from_wf_data" in _record_operations(tacaw)
    assert wf.Provenance.seaid in tacaw.Provenance.Analysis_parent
    assert tacaw.Provenance.seaid in wf.Provenance.Analysis_child


def test_tacawdata_sea_roundtrip_preserves_signal_and_parent_link(tmp_path):
    wf = _make_wf_data(tmp_path)
    tacaw = TACAWData(wf, force_rerun=True)
    path = tmp_path / "tacaw.sea"

    tacaw.to_sea(path)
    loaded = TACAWData.load(path)

    assert isinstance(loaded, pysea_base.Signal)
    assert loaded.Provenance.seaid == tacaw.Provenance.seaid
    assert wf.Provenance.seaid in loaded.Provenance.Analysis_parent
    np.testing.assert_allclose(loaded.data, tacaw.data)
    assert "TACAWData.from_wf_data" in _record_operations(loaded)


def test_tacaw_analysis_methods_return_tracked_signals_by_default(tmp_path):
    wf = _make_wf_data(tmp_path)
    tacaw = TACAWData(wf, force_rerun=True)

    derived = [
        tacaw.spectrum(),
        tacaw.spectrum_image(0.0),
        tacaw.diffraction(),
        tacaw.spectral_diffraction(0.0),
        tacaw.masked_spectrum(),
        tacaw.dispersion(np.array([tacaw.kxs[0]]), np.array([tacaw.kys[0]])),
    ]

    for signal in derived:
        assert isinstance(signal, pysea_base.Signal)
        assert tacaw.Provenance.seaid in signal.Provenance.Analysis_parent
        assert signal.Provenance.seaid in tacaw.Provenance.Analysis_child
        assert signal.Analysis.processing_records

    operations = {
        record.operation
        for signal in derived
        for record in signal.Analysis.processing_records
    }
    assert {
        "TACAWData.spectrum",
        "TACAWData.spectrum_image",
        "TACAWData.diffraction",
        "TACAWData.spectral_diffraction",
        "TACAWData.masked_spectrum",
        "TACAWData.dispersion",
    }.issubset(operations)


def test_tacaw_analysis_methods_can_still_return_arrays(tmp_path):
    wf = _make_wf_data(tmp_path)
    tacaw = TACAWData(wf, force_rerun=True)

    raw_arrays = [
        tacaw.spectrum(as_signal=False),
        tacaw.spectrum_image(0.0, as_signal=False),
        tacaw.diffraction(as_signal=False),
        tacaw.spectral_diffraction(0.0, as_signal=False),
        tacaw.masked_spectrum(as_signal=False),
        tacaw.dispersion(np.array([tacaw.kxs[0]]), np.array([tacaw.kys[0]]), as_signal=False),
    ]

    assert all(isinstance(array, np.ndarray) for array in raw_arrays)


def test_haadfdata_records_parent_and_adf_processing(tmp_path):
    wf = _make_wf_data(tmp_path)

    haadf = HAADFData(wf)
    haadf.calculateADF(inner_mrad=0, outer_mrad=1000)

    operations = _record_operations(haadf)
    assert "HAADFData.from_wf_data" in operations
    assert "HAADFData.calculateADF" in operations
    assert wf.Provenance.seaid in haadf.Provenance.Analysis_parent


def test_haadfdata_sea_roundtrip_preserves_signal_and_processing(tmp_path):
    wf = _make_wf_data(tmp_path)
    haadf = HAADFData(wf)
    haadf.calculateADF(inner_mrad=0, outer_mrad=1000)
    path = tmp_path / "haadf.sea"

    haadf.to_sea(path)
    loaded = HAADFData.load(path)

    assert isinstance(loaded, pysea_base.Signal)
    assert loaded.Provenance.seaid == haadf.Provenance.seaid
    assert wf.Provenance.seaid in loaded.Provenance.Analysis_parent
    np.testing.assert_allclose(loaded.data, haadf.data)
    assert {
        "HAADFData.from_wf_data",
        "HAADFData.calculateADF",
    }.issubset(_record_operations(loaded))
