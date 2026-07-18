"""
Test sea-eco integration features in PySlice.

Part 1: Tests Signal, Dimensions, Dimension, GeneralMetadata, SignalSet,
        and .sea file serialization round-trips.

Part 2: Tests that PySlice data classes (WFData, TACAWData, HAADFData)
        properly inherit from Signal with actual multislice simulations.
"""
import sys
import os
import numpy as np

try:
    import pyslice
except ModuleNotFoundError:
    sys.path.insert(0, '../src')

from pyslice.data import (
    Signal, Dimensions, Dimension, GeneralMetadata,
    SignalSet, AcquisitionSet, load
)

# Ensure output directory exists
os.makedirs("outputs", exist_ok=True)

print("=" * 60)
print("Testing sea-eco Integration")
print("=" * 60)


def test_dimension_creation():
    """Test Dimension class with various calibration methods."""
    print("\n--- Testing Dimension Creation ---")

    # Create dimension with explicit values
    dim_values = Dimension(
        name='energy',
        space='energy',
        units='eV',
        values=np.linspace(0, 100, 101)
    )
    assert dim_values.name == 'energy'
    assert dim_values.units == 'eV'
    assert len(dim_values.values) == 101
    assert np.isclose(dim_values.values[0], 0)
    assert np.isclose(dim_values.values[-1], 100)
    print(f"  Created dimension with values: {dim_values}")

    # Create dimension with scale/offset
    dim_scaled = Dimension(
        name='position',
        space='position',
        units='nm',
        size=50,
        scale=0.1,
        offset=-2.5
    )
    assert dim_scaled.size == 50
    assert np.isclose(dim_scaled.values[0], -2.5)
    assert np.isclose(dim_scaled.values[1], -2.4)
    print(f"  Created dimension with scale/offset: {dim_scaled}")

    # Test dimension indexing
    idx = dim_values.find_nearest_index(50.0)
    assert idx == 50, f"Expected index 50, got {idx}"
    print(f"  find_nearest_index(50.0) = {idx}")

    print("  PASSED: Dimension creation")
    return dim_values, dim_scaled


def test_dimensions_collection():
    """Test Dimensions class for managing multiple dimensions."""
    print("\n--- Testing Dimensions Collection ---")

    dims = Dimensions([
        Dimension(name='x', space='position', units='nm', values=np.arange(10)),
        Dimension(name='y', space='position', units='nm', values=np.arange(20)),
        Dimension(name='energy', space='energy', units='eV', values=np.arange(100)),
    ], nav_dimensions=[0, 1], sig_dimensions=[2])

    assert dims.ndim == 3
    assert dims.get_names() == ['x', 'y', 'energy']
    assert dims.nav_dimensions == [0, 1]
    assert dims.sig_dimensions == [2]
    print(f"  Created Dimensions: {dims}")
    print(f"  Navigation dims: {dims.nav_dimensions}")
    print(f"  Signal dims: {dims.sig_dimensions}")

    # Test dimension access by name
    energy_dim = dims['energy']
    assert energy_dim.name == 'energy'
    print(f"  Accessed by name: dims['energy'] = {energy_dim}")

    # Test dimension access by index
    x_dim = dims[0]
    assert x_dim.name == 'x'
    print(f"  Accessed by index: dims[0] = {x_dim}")

    print("  PASSED: Dimensions collection")
    return dims


def test_general_metadata():
    """Test GeneralMetadata class."""
    print("\n--- Testing GeneralMetadata ---")

    meta = GeneralMetadata({
        'General': {
            'title': 'Test Signal',
            'authors': ['Alice', 'Bob'],
            'date': '2024-01-15'
        },
        'Simulation': {
            'voltage_eV': 100000.0,
            'aperture_mrad': 30.0,
            'sampling_A': 0.1
        },
        'Custom': {
            'notes': 'Test metadata',
            'version': 1.0
        }
    })

    # Test attribute access
    assert meta.General.title == 'Test Signal'
    assert meta.Simulation.voltage_eV == 100000.0
    assert meta.Custom.notes == 'Test metadata'
    print(f"  General.title: {meta.General.title}")
    print(f"  Simulation.voltage_eV: {meta.Simulation.voltage_eV}")
    print(f"  Custom.notes: {meta.Custom.notes}")

    # Test nested dict-like access
    assert hasattr(meta, 'General')
    assert hasattr(meta.General, 'authors')
    print(f"  General.authors: {meta.General.authors}")

    print("  PASSED: GeneralMetadata")
    return meta


def test_signal_creation():
    """Test Signal class creation with various configurations."""
    print("\n--- Testing Signal Creation ---")

    # Create a 3D signal (x, y, energy)
    data = np.random.rand(10, 20, 100).astype(np.float32)

    signal = Signal(
        data=data,
        name='TestSpectrum',
        dimensions=Dimensions([
            Dimension(name='x', space='position', units='nm', values=np.arange(10) * 0.5),
            Dimension(name='y', space='position', units='nm', values=np.arange(20) * 0.5),
            Dimension(name='energy', space='energy', units='eV', values=np.arange(100) * 0.5),
        ], nav_dimensions=[0, 1], sig_dimensions=[2]),
        signal_type='EELS',
        metadata=GeneralMetadata({
            'General': {'title': 'Test EELS Spectrum'},
            'Acquisition': {'exposure_ms': 100}
        })
    )

    assert signal.name == 'TestSpectrum'
    assert signal.signal_type == 'EELS'
    assert signal.data.shape == (10, 20, 100)
    assert signal.dimensions.ndim == 3
    print(f"  Created signal: {signal.name}")
    print(f"  Shape: {signal.data.shape}")
    print(f"  Signal type: {signal.signal_type}")
    print(f"  Dimensions: {signal.dimensions.get_names()}")

    print("  PASSED: Signal creation")
    return signal


def test_signal_serialization():
    """Test Signal .sea file serialization round-trip."""
    print("\n--- Testing Signal Serialization ---")

    # Create a signal with all features
    original_data = np.random.rand(8, 12, 50).astype(np.float32)

    original = Signal(
        data=original_data,
        name='SerializationTest',
        dimensions=Dimensions([
            Dimension(name='probe', space='index', values=np.arange(8)),
            Dimension(name='time', space='time', units='ps', values=np.arange(12) * 0.005),
            Dimension(name='energy', space='energy', units='eV', values=np.linspace(0, 10, 50)),
        ], nav_dimensions=[0, 1], sig_dimensions=[2]),
        signal_type='2D-EELS',
        metadata=GeneralMetadata({
            'General': {'title': 'Serialization Test'},
            'Simulation': {'voltage_eV': 100000, 'sampling': 0.1}
        })
    )

    # Save to .sea file
    output_path = 'outputs/test_sea_eco.sea'
    original.to_sea(output_path)
    assert os.path.exists(output_path)
    print(f"  Saved to: {output_path}")

    # Load back using from_sea method
    loaded = Signal()
    loaded.from_sea(output_path)

    # Verify data
    assert np.allclose(loaded.data, original_data)
    print("  Data matches: PASSED")

    # Verify attributes
    assert loaded.name == original.name
    assert loaded.signal_type == original.signal_type
    print(f"  Name matches: {loaded.name}")
    print(f"  Signal type matches: {loaded.signal_type}")

    # Verify dimensions
    assert loaded.dimensions.ndim == original.dimensions.ndim
    for i, (orig_dim, load_dim) in enumerate(zip(original.dimensions.dimensions, loaded.dimensions.dimensions)):
        assert orig_dim.name == load_dim.name
        assert np.allclose(orig_dim.values, load_dim.values)
    print(f"  Dimensions match: {loaded.dimensions.get_names()}")

    # Verify metadata
    assert loaded.metadata.General.title == 'Serialization Test'
    assert loaded.metadata.Simulation.voltage_eV == 100000
    print("  Metadata matches: PASSED")

    print("  PASSED: Signal serialization round-trip")
    return output_path


def test_load_function(sea_file_path):
    """Test the standalone load() function."""
    print("\n--- Testing load() Function ---")

    loaded = load(sea_file_path)

    assert isinstance(loaded, Signal)
    assert loaded.data is not None
    assert loaded.name == 'SerializationTest'
    print(f"  Loaded signal: {loaded.name}")
    print(f"  Type: {type(loaded).__name__}")
    print(f"  Shape: {loaded.data.shape}")

    print("  PASSED: load() function")


def test_signal_set():
    """Test SignalSet for managing collections of signals."""
    print("\n--- Testing SignalSet ---")

    # Create multiple signals
    signals = []
    for i in range(3):
        sig = Signal(
            data=np.random.rand(10, 10).astype(np.float32) * (i + 1),
            name=f'Signal_{i}',
            dimensions=Dimensions([
                Dimension(name='x', values=np.arange(10).astype(np.float32)),
                Dimension(name='y', values=np.arange(10).astype(np.float32)),
            ]),
            signal_type='Image'
        )
        signals.append(sig)

    # Create SignalSet
    signal_set = SignalSet(signals=signals)

    assert len(signal_set.signals) == 3
    print(f"  Created SignalSet with {len(signal_set.signals)} signals")

    # Access individual signals
    for i, sig in enumerate(signal_set.signals):
        print(f"    [{i}] {sig.name}: shape={sig.data.shape}")

    # Access by index
    assert signal_set[0].name == 'Signal_0'
    print("  Access by index signal_set[0]: PASSED")

    # Access by name
    assert signal_set['Signal_1'].name == 'Signal_1'
    print("  Access by name signal_set['Signal_1']: PASSED")

    # Save SignalSet
    output_path = 'outputs/test_signal_set.sea'
    signal_set.to_sea(output_path)
    assert os.path.exists(output_path)
    print(f"  Saved SignalSet to: {output_path}")

    # Load SignalSet
    loaded_set = SignalSet()
    loaded_set.from_sea(output_path)
    assert len(loaded_set.signals) == 3
    print(f"  Loaded SignalSet with {len(loaded_set.signals)} signals")

    print("  PASSED: SignalSet")


def test_complex_signal():
    """Test Signal with complex-valued data."""
    print("\n--- Testing Complex Signal ---")

    # Create complex wavefunction-like data
    real_part = np.random.rand(20, 30)
    imag_part = np.random.rand(20, 30)
    complex_data = (real_part + 1j * imag_part).astype(np.complex64)

    signal = Signal(
        data=complex_data,
        name='Wavefunction',
        dimensions=Dimensions([
            Dimension(name='kx', space='reciprocal', units='1/A',
                     values=np.linspace(-5, 5, 20)),
            Dimension(name='ky', space='reciprocal', units='1/A',
                     values=np.linspace(-5, 5, 30)),
        ]),
        signal_type='Diffraction'
    )

    assert signal.data.dtype == np.complex64
    assert signal.data.shape == (20, 30)
    print(f"  Created complex signal: {signal.name}")
    print(f"  dtype: {signal.data.dtype}")

    # Save and load complex signal
    output_path = 'outputs/test_complex_signal.sea'
    signal.to_sea(output_path)

    loaded = Signal()
    loaded.from_sea(output_path)

    assert np.allclose(loaded.data, complex_data)
    print("  Complex data round-trip: PASSED")

    print("  PASSED: Complex signal")


def test_wfdata_is_signal():
    """Test that WFData inherits from Signal with actual simulation."""
    print("\n--- Testing WFData is Signal (with simulation) ---")

    from pyslice.io.loader import Loader
    from pyslice.multislice.calculators import MultisliceCalculator
    from pyslice.postprocessing.wf_data import WFData

    # Test parameters
    dump = "inputs/hBN_truncated.lammpstrj"
    dt = 0.005
    types = {1: "B", 2: "N"}

    # Load trajectory and run simulation
    trajectory = Loader(dump, timestep=dt, atom_mapping=types).load()
    calculator = MultisliceCalculator()
    calculator.setup(trajectory, aperture=0, voltage_eV=100e3, sampling=0.1, slice_thickness=0.5)
    wf_data = calculator.run()

    # Test inheritance
    assert isinstance(wf_data, Signal), "WFData should be an instance of Signal"
    assert isinstance(wf_data, WFData), "WFData should still be an instance of WFData"
    print("  PASSED: WFData is instance of Signal")

    # Test Signal attributes exist
    assert hasattr(wf_data, 'dimensions'), "WFData should have dimensions attribute"
    assert hasattr(wf_data, 'metadata'), "WFData should have metadata attribute"
    assert wf_data.signal_type == 'Diffraction'
    print("  PASSED: WFData has Signal attributes")

    # Test dimensions
    assert wf_data.dimensions is not None
    dim_names = wf_data.dimensions.get_names()
    assert 'probe' in dim_names and 'time' in dim_names and 'kx' in dim_names
    print(f"  Dimensions: {dim_names}")

    # Test backward compatibility
    assert hasattr(wf_data, 'probe_positions')
    assert hasattr(wf_data, 'kxs')
    assert hasattr(wf_data, 'array')
    print("  PASSED: WFData backward compatible")

    # Test data property
    data = wf_data.data
    assert isinstance(data, np.ndarray)
    print(f"  Data shape: {data.shape}")
    print("  PASSED: WFData data property works")

    return wf_data


def test_tacawdata_is_signal(wf_data):
    """Test that TACAWData inherits from Signal."""
    print("\n--- Testing TACAWData is Signal ---")

    from pyslice.postprocessing.tacaw_data import TACAWData

    tacaw = TACAWData(wf_data)

    # Test inheritance
    assert isinstance(tacaw, Signal)
    assert isinstance(tacaw, TACAWData)
    print("  PASSED: TACAWData is instance of Signal")

    # Test Signal attributes
    assert tacaw.signal_type == '2D-EELS'
    dim_names = tacaw.dimensions.get_names()
    assert 'frequency' in dim_names and 'kx' in dim_names
    print(f"  Dimensions: {dim_names}")

    # Test backward compatibility
    assert hasattr(tacaw, 'frequencies')
    assert hasattr(tacaw, 'intensity')
    print("  PASSED: TACAWData backward compatible")

    # Test data property
    data = tacaw.data
    assert isinstance(data, np.ndarray)
    print(f"  Data shape: {data.shape}")
    print("  PASSED: TACAWData data property works")

    return tacaw


def test_haadf_is_signal():
    """Test that HAADFData inherits from Signal."""
    print("\n--- Testing HAADFData is Signal (with simulation) ---")

    from pyslice.io.loader import Loader
    from pyslice.multislice.multislice import probe_grid
    from pyslice.multislice.calculators import MultisliceCalculator
    from pyslice.postprocessing.haadf_data import HAADFData

    # Test parameters
    dump = "inputs/hBN_truncated.lammpstrj"
    dt = 0.005
    types = {1: "B", 2: "N"}
    a, b = 2.4907733333333337, 2.1570729817355123

    # Load trajectory and run simulation with probe grid
    trajectory = Loader(dump, timestep=dt, atom_mapping=types).load()
    trajectory = trajectory.slice_timesteps(ith=10)

    xy = probe_grid([0, 2*a], [0, 2*b], 2, 2)

    calculator = MultisliceCalculator()
    calculator.setup(trajectory, aperture=30, voltage_eV=100e3, sampling=0.2,
                     slice_thickness=1.0, probe_positions=xy)
    wf_data = calculator.run()

    haadf = HAADFData(wf_data)

    # Test inheritance
    assert isinstance(haadf, Signal)
    assert isinstance(haadf, HAADFData)
    print("  PASSED: HAADFData is instance of Signal")

    # Test Signal attributes
    assert haadf.signal_type == 'Image'
    print("  PASSED: HAADFData Signal attributes")

    # Calculate ADF
    haadf.calculateADF(inner_mrad=45, outer_mrad=150)
    assert haadf.adf is not None
    assert haadf.data is not None
    print(f"  ADF shape: {haadf.data.shape}")

    # Check dimensions
    dim_names = haadf.dimensions.get_names()
    assert 'x' in dim_names and 'y' in dim_names
    print(f"  Dimensions: {dim_names}")

    # Test metadata
    assert haadf.metadata is not None
    print("  PASSED: HAADFData complete")

    return haadf


def test_tacawdata_sea_roundtrip(tacaw):
    """Test TACAWData .sea serialization round-trip using PySliceSerial mixin."""
    print("\n--- Testing TACAWData .sea Round-trip (PySliceSerial) ---")

    from pathlib import Path
    from pyslice.postprocessing.tacaw_data import TACAWData
    from pyslice.backend import to_numpy

    # Store original values
    original_array = to_numpy(tacaw._array)
    original_freqs = to_numpy(tacaw.frequencies)
    original_positions = list(tacaw.probe_positions)
    original_cache_dir = tacaw.cache_dir

    # Save using to_sea
    output_path = 'outputs/test_tacaw_serial.sea'
    tacaw.to_sea(output_path)
    assert os.path.exists(output_path)
    print(f"  Saved TACAWData to: {output_path}")

    # Load using classmethod
    loaded = TACAWData.load(output_path)
    assert isinstance(loaded, TACAWData)
    print("  Loaded TACAWData using TACAWData.load()")

    # Verify data integrity
    assert np.allclose(loaded._array, original_array), "Array data mismatch"
    print("  PASSED: Array data matches")

    assert np.allclose(loaded.frequencies, original_freqs), "Frequencies mismatch"
    print("  PASSED: Frequencies match")

    assert loaded.probe_positions == original_positions, "Probe positions mismatch"
    assert isinstance(loaded.probe_positions, list), "Probe positions should be list"
    assert isinstance(loaded.probe_positions[0], tuple), "Probe positions elements should be tuples"
    print("  PASSED: Probe positions match (list of tuples)")

    assert isinstance(loaded.cache_dir, Path), "cache_dir should be Path"
    assert str(loaded.cache_dir) == str(original_cache_dir), "cache_dir mismatch"
    print("  PASSED: cache_dir is Path and matches")

    # Verify kxs, kys are restored
    assert loaded._kxs is not None, "_kxs should be restored"
    assert loaded._kys is not None, "_kys should be restored"
    print("  PASSED: Coordinate arrays restored")

    # Verify excluded attrs are None
    assert loaded.probe is None, "probe should be None after load"
    assert loaded._wf_array is None, "_wf_array should be None after load"
    print("  PASSED: Excluded attrs are None")

    print("  PASSED: TACAWData .sea round-trip complete")
    return loaded


def test_wfdata_sea_roundtrip(wf_data):
    """Test WFData .sea serialization round-trip using PySliceSerial mixin."""
    print("\n--- Testing WFData .sea Round-trip (PySliceSerial) ---")

    from pathlib import Path
    from pyslice.postprocessing.wf_data import WFData
    from pyslice.backend import to_numpy

    # Store original values
    original_array = to_numpy(wf_data._array)
    original_positions = list(wf_data.probe_positions)
    original_kxs = to_numpy(wf_data.kxs)

    # Save using to_sea
    output_path = 'outputs/test_wf_serial.sea'
    wf_data.to_sea(output_path)
    assert os.path.exists(output_path)
    print(f"  Saved WFData to: {output_path}")

    # Load using classmethod
    loaded = WFData.load(output_path)
    assert isinstance(loaded, WFData)
    print("  Loaded WFData using WFData.load()")

    # Verify data integrity
    assert np.allclose(loaded._array, original_array), "Array data mismatch"
    print("  PASSED: Array data matches")

    assert loaded.probe_positions == original_positions, "Probe positions mismatch"
    assert isinstance(loaded.probe_positions, list), "Probe positions should be list"
    print("  PASSED: Probe positions match")

    assert np.allclose(loaded.kxs, original_kxs), "kxs mismatch"
    print("  PASSED: Coordinate arrays match")

    assert isinstance(loaded.cache_dir, Path), "cache_dir should be Path"
    print("  PASSED: cache_dir is Path")

    print("  PASSED: WFData .sea round-trip complete")
    return loaded


def test_haadfdata_sea_roundtrip(haadf):
    """Test HAADFData .sea serialization round-trip using PySliceSerial mixin."""
    print("\n--- Testing HAADFData .sea Round-trip (PySliceSerial) ---")

    from pathlib import Path
    from pyslice.postprocessing.haadf_data import HAADFData
    from pyslice.backend import to_numpy

    # Store original values
    original_array = to_numpy(haadf._array)
    original_positions = list(haadf.probe_positions)

    # Save using to_sea
    output_path = 'outputs/test_haadf_serial.sea'
    haadf.to_sea(output_path)
    assert os.path.exists(output_path)
    print(f"  Saved HAADFData to: {output_path}")

    # Load using classmethod
    loaded = HAADFData.load(output_path)
    assert isinstance(loaded, HAADFData)
    print("  Loaded HAADFData using HAADFData.load()")

    # Verify data integrity
    assert np.allclose(loaded._array, original_array), "Array data mismatch"
    print("  PASSED: Array data matches")

    # HAADFData stores probe_positions as array, compare as arrays
    loaded_positions = np.array(loaded.probe_positions) if isinstance(loaded.probe_positions, list) else loaded.probe_positions
    original_positions_arr = np.array(original_positions)
    assert np.allclose(loaded_positions, original_positions_arr), "Probe positions mismatch"
    print("  PASSED: Probe positions match")

    assert isinstance(loaded.cache_dir, Path), "cache_dir should be Path"
    print("  PASSED: cache_dir is Path")

    print("  PASSED: HAADFData .sea round-trip complete")
    return loaded


if __name__ == '__main__':
    # Run all tests
    print("=" * 60)
    print("PART 1: sea-eco Module Tests")
    print("=" * 60)
    test_dimension_creation()
    test_dimensions_collection()
    test_general_metadata()
    signal = test_signal_creation()
    sea_path = test_signal_serialization()
    test_load_function(sea_path)
    test_signal_set()
    test_complex_signal()

    print("\n" + "=" * 60)
    print("PART 2: PySlice Signal Integration Tests (with simulations)")
    print("=" * 60)
    wf_data = test_wfdata_is_signal()
    tacaw = test_tacawdata_is_signal(wf_data)
    haadf = test_haadf_is_signal()

    print("\n" + "=" * 60)
    print("PART 3: PySliceSerial Mixin Tests (.sea round-trips)")
    print("=" * 60)
    test_tacawdata_sea_roundtrip(tacaw)
    test_wfdata_sea_roundtrip(wf_data)
    test_haadfdata_sea_roundtrip(haadf)

    print("\n" + "=" * 60)
    print("All sea-eco integration tests PASSED!")
    print("=" * 60)
