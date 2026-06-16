"""
PySliceSerial mixin for HDF5/SEA serialization of PySlice data classes.

This mixin provides generalized serialization/deserialization for classes
that inherit from Signal but have special attributes (tensors, Paths, etc.)
that need conversion for HDF5 storage.
"""
import numpy as np
from pathlib import Path
from h5py import File, Group, Dataset

try:
    import sys
    sys.path.insert(1,"../../")
    from pySEA.sea_eco.architecture.base_structure_numpy import Signal, Dimensions, Dimension, Metadata, safe_decode
except Exception:
    class Signal:
        def to_sea(self,*args,**kwargs):
            raise ImportError("pySEA is required for .to_sea() serialization")
    Dimensions,Dimension,Metadata = None,None,None

def _to_numpy(x):
    """Convert tensor or array-like to numpy array."""
    if x is None:
        return None
    if hasattr(x, 'cpu'):
        return x.cpu().numpy()
    return np.asarray(x)


class PySliceSerial:
    """
    Mixin class providing generalized HDF5/SEA serialization for PySlice data classes.

    Subclasses should define a `_sea_config` dict with the following optional keys:

    - tensor_attrs: List of attribute names that are torch tensors (converted to numpy)
    - path_attrs: List of attribute names that are Path objects (converted to string)
    - tuple_list_attrs: List of attribute names that are lists of tuples (converted to arrays)
    - exclude_attrs: List of attribute names to exclude from serialization
    - force_datasets: List of attribute names to store as HDF5 datasets (not attrs)
    - default_attrs: Dict of default values to set during deserialization

    Example:
        class MyData(PySliceSerial, Signal):
            _sea_config = {
                'tensor_attrs': ['_array', '_kxs', '_kys'],
                'path_attrs': ['cache_dir'],
                'tuple_list_attrs': ['probe_positions'],
                'exclude_attrs': ['probe', '_wf_array'],
                'force_datasets': ['_array', 'probe_positions'],
            }
    """

    _sea_config = {}

    def to_hdf5_group(self, parent_group, force_datasets=None, name=None):
        """Serialize to HDF5 group with automatic type conversions."""
        config = getattr(self, '_sea_config', {})

        tensor_attrs = config.get('tensor_attrs', [])
        path_attrs = config.get('path_attrs', [])
        tuple_list_attrs = config.get('tuple_list_attrs', [])
        exclude_attrs = config.get('exclude_attrs', [])
        config_force_datasets = config.get('force_datasets', [])

        if force_datasets is None:
            force_datasets = ['data']
        force_datasets = list(force_datasets) + config_force_datasets

        # Store originals for restoration
        originals = {}

        # Convert tensor attributes to numpy
        for attr in tensor_attrs:
            if hasattr(self, attr):
                originals[attr] = getattr(self, attr)
                setattr(self, attr, _to_numpy(getattr(self, attr)))
                # Ensure tensor attrs are stored as datasets
                if attr not in force_datasets:
                    force_datasets.append(attr)

        # Convert Path attributes to string
        for attr in path_attrs:
            if hasattr(self, attr):
                originals[attr] = getattr(self, attr)
                val = getattr(self, attr)
                setattr(self, attr, str(val) if val is not None else None)

        # Convert list of tuples to numpy array
        for attr in tuple_list_attrs:
            if hasattr(self, attr):
                originals[attr] = getattr(self, attr)
                val = getattr(self, attr)
                if val is not None:
                    setattr(self, attr, np.array(val))
                if attr not in force_datasets:
                    force_datasets.append(attr)

        # Temporarily remove non-serializable attributes
        for attr in exclude_attrs:
            if hasattr(self, attr):
                originals[attr] = getattr(self, attr)
                try:
                    delattr(self, attr)
                except AttributeError:
                    setattr(self, attr, None)

        try:
            # Call parent's to_hdf5_group (Signal's method)
            result = super().to_hdf5_group(parent_group, force_datasets=force_datasets, name=name)
        finally:
            # Restore original attributes
            for attr, val in originals.items():
                setattr(self, attr, val)

        return result

    def to_sea(self, file_path, force_datasets=None):
        """Save to .sea file with automatic type conversions."""
        config = getattr(self, '_sea_config', {})
        config_force_datasets = config.get('force_datasets', [])

        if force_datasets is None:
            force_datasets = ['data']
        force_datasets = list(force_datasets) + config_force_datasets

        super().to_sea(file_path, force_datasets=force_datasets)

    def from_hdf5_group(self, group):
        """Deserialize from HDF5 group with automatic type conversions."""
        #from .signal import safe_decode, Dimensions, GeneralMetadata

        config = getattr(self, '_sea_config', {})

        tensor_attrs = config.get('tensor_attrs', [])
        path_attrs = config.get('path_attrs', [])
        tuple_list_attrs = config.get('tuple_list_attrs', [])
        exclude_attrs = config.get('exclude_attrs', [])
        default_attrs = config.get('default_attrs', {})

        # Build mapping from storage keys (without _) to internal attr names
        key_map = {}
        for attr in tensor_attrs + path_attrs + tuple_list_attrs:
            storage_key = attr[1:] if attr.startswith('_') else attr
            key_map[storage_key] = attr

        # Initialize default attributes (Signal expects these)
        signal_defaults = {
            '_original_metadata': None,
            '_parent_SignalSet': None,
            'detector': None,
            'is_lazy': False,
            '_array': None,
        }
        for attr, val in signal_defaults.items():
            setattr(self, attr, val)

        # Initialize excluded attrs to None
        for attr in exclude_attrs:
            setattr(self, attr, None)

        # Initialize user-specified defaults
        for attr, val in default_attrs.items():
            setattr(self, attr, val)

        # Read datasets (large arrays)
        for key, item in group.items():
            if isinstance(item, Dataset):
                value = item[()]
                attr_name = key_map.get(key, key)
                # Check for private version
                if not hasattr(self, attr_name) and hasattr(self, f'_{key}'):
                    attr_name = f'_{key}'
                setattr(self, attr_name, value)

        # Read attributes (scalars/small values)
        for key, val in group.attrs.items():
            if key == 'sea_type':
                continue
            decoded_val = safe_decode(val)
            attr_name = key_map.get(key, key)

            # Check for private version
            if not hasattr(self, attr_name):
                if hasattr(self, f'_{key}'):
                    attr_name = f'_{key}'

            setattr(self, attr_name, decoded_val)

        # Handle nested groups (Dimensions, GeneralMetadata)
        for key, item in group.items():
            if isinstance(item, Group):
                sea_type = safe_decode(item.attrs.get('sea_type', b''))
                if sea_type == 'Dimensions':
                    dims = Dimensions()
                    dims.from_hdf5_group(item)
                    self._local_dimensions = dims
                elif sea_type == 'GeneralMetadata':
                    meta = GeneralMetadata()
                    meta.from_hdf5_group(item)
                    if key == 'metadata' or key == 'Metadata':
                        self.metadata = meta

        # Post-process: convert types back
        for attr in path_attrs:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if val is not None and isinstance(val, str):
                    setattr(self, attr, Path(val))

        for attr in tuple_list_attrs:
            if hasattr(self, attr):
                val = getattr(self, attr)
                if val is not None and isinstance(val, np.ndarray):
                    setattr(self, attr, [tuple(p) for p in val])

    @classmethod
    def load(cls, file_path):
        """
        Load an object from a .sea file.

        Args:
            file_path: Path to the .sea file

        Returns:
            Instance of the class populated with data from the file
        """
        file_path = Path(file_path)
        if file_path.suffix != '.sea':
            file_path = file_path.with_suffix('.sea')

        # Create empty instance without calling __init__
        obj = cls.__new__(cls)

        with File(file_path, 'r') as f:
            if len(f) != 1:
                raise ValueError("The HDF5 file contains multiple groups.")
            main_group = f[list(f.keys())[0]]
            obj.from_hdf5_group(main_group)

        return obj
