"""
PySliceSerial mixin for HDF5/SEA serialization of PySlice data classes.

This mixin provides generalized serialization/deserialization for classes
that inherit from Signal but have special attributes (tensors, Paths, etc.)
that need conversion for HDF5 storage.
"""
import numpy as np
from copy import deepcopy
from functools import wraps
from pathlib import Path
from h5py import File
from ..backend import to_numpy

try:
    from pySEA.sea_eco.architecture.base_structure import (
        Signal,
        Dimensions,
        Dimension,
        Metadata,
        SignalQuantities,
    )
except ModuleNotFoundError as exc:
    _optional_sea_modules = {
        "pySEA",
        "pySEA.sea_eco",
        "pySEA.sea_eco.architecture",
        "pySEA.sea_eco.architecture.base_structure",
    }
    if exc.name not in _optional_sea_modules:
        raise

    class Signal:
        def to_sea(self,*args,**kwargs):
            raise ImportError("pySEA is required for .to_sea() serialization")
    Dimensions,Dimension,Metadata,SignalQuantities = None,None,None,None

SEA_ECO_AVAILABLE = Dimensions is not None
GeneralMetadata = Metadata

def _to_numpy(x):
    """Convert tensor or array-like to numpy array."""
    if x is None:
        return None
    return to_numpy(x)


def _seaid_string(value):
    """Return a SEAID string for a signal-like object or ID value."""
    provenance = getattr(value, "Provenance", value)
    seaid = getattr(provenance, "seaid", None)
    if isinstance(seaid, str):
        return seaid
    return "" if provenance is None else str(provenance)


def _recording_enabled(target):
    """Return whether pySEA and the target object currently allow recording."""
    if target is None or not hasattr(target, "Analysis") or not hasattr(target, "Provenance"):
        return False
    if not SEA_ECO_AVAILABLE:
        return False
    from pySEA.sea_eco.pipeline._record import is_recording
    if not is_recording():
        return False
    checker = getattr(target, "is_recording", None)
    if callable(checker):
        return bool(checker())
    return bool(getattr(target, "_Signal__recording", True))


def _append_processing_summary(target, record):
    """Mirror sea-eco's lightweight processing summary in object metadata."""
    if Metadata is None or not hasattr(target, "metadata") or target.metadata is None:
        return
    if not hasattr(target.metadata, "Processing"):
        target.metadata.Processing = Metadata()
    elif isinstance(target.metadata.Processing, dict):
        target.metadata.Processing = Metadata(target.metadata.Processing)
    step_key = f"step_{len(target.Analysis.processing_records)}"
    summary = f"{record.timestamp} {record.operation} ({record.callable_label})"
    setattr(target.metadata.Processing, step_key, summary)


def record_pyslice_operation(
    target,
    operation,
    *,
    inputs=None,
    parameters=None,
    callable_obj=None,
    in_place=False,
):
    """Append a PySlice processing record without forcing pipeline execution.

    This records scientific PySlice actions on sea-eco Signal-compatible
    objects. In-place actions are kept as ledger entries on the object itself;
    derived objects also receive parent/child SEAID links.
    """
    if not _recording_enabled(target):
        return None

    from pySEA.sea_eco.pipeline._record import (
        NodeRecord,
        _make_json_safe,
        serialize_call_args,
        serialize_call_kwargs,
    )

    inputs = list(inputs or [])
    if in_place and target not in inputs:
        inputs.insert(0, target)

    unique_inputs = []
    seen = set()
    for item in inputs:
        if item is None or id(item) in seen:
            continue
        unique_inputs.append(item)
        seen.add(id(item))

    parameters = dict(parameters or {})
    parameters.setdefault("in_place", bool(in_place))
    kwargs_serializable = _make_json_safe(parameters)

    callable_obj = callable_obj or record_pyslice_operation
    module = getattr(callable_obj, "__module__", "")
    qualname = getattr(callable_obj, "__qualname__", getattr(callable_obj, "__name__", operation))
    input_seaids = [
        _seaid_string(item)
        for item in unique_inputs
        if hasattr(item, "Provenance")
    ]
    output_seaid = _seaid_string(target)

    record = NodeRecord(
        operation=operation,
        callable_ref_module=module,
        callable_ref_qualname=qualname,
        input_seaids=input_seaids,
        output_seaid=output_seaid,
        args_repr="",
        kwargs_serializable=kwargs_serializable,
        args_serializable=serialize_call_args(tuple(unique_inputs), data_mode="object"),
        call_kwargs_serializable=serialize_call_kwargs(kwargs_serializable, data_mode="object"),
    )

    target.Analysis._parent = target
    target.Analysis.add_processing_record(record, name=record.name)
    _append_processing_summary(target, record)

    for source in unique_inputs:
        if source is target or not hasattr(source, "Provenance"):
            continue
        target.Provenance.add_parent(_seaid_string(source))
        source.Provenance.add_child(output_seaid)
        source_analysis = getattr(source, "Analysis", None)
        if source_analysis is None:
            continue
        source_analysis._parent = source
        source_analysis.add_analysis_output(target, owns_output=False)
        source_record = deepcopy(record)
        source_analysis.add_processing_record(source_record, name=source_record.name)

    return record


def track_pyslice_action(func):
    """Decorator for PySlice methods that mutate a Signal-compatible object."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        depth = int(getattr(self, "_pyslice_tracking_depth", 0))
        self._pyslice_tracking_depth = depth + 1
        try:
            result = func(self, *args, **kwargs)
        finally:
            if depth:
                self._pyslice_tracking_depth = depth
            else:
                delattr(self, "_pyslice_tracking_depth")

        # Convenience methods may call another tracked PySlice method. Record
        # only the outer operation so one user action produces one receipt.
        if depth:
            return result

        parameters = {
            "args": args,
            "kwargs": kwargs,
        }
        target = result if hasattr(result, "Analysis") and result is not self else self
        record_pyslice_operation(
            target,
            f"{type(self).__name__}.{func.__name__}",
            inputs=[self],
            parameters=parameters,
            callable_obj=func,
            in_place=target is self,
        )
        return result
    return wrapper


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
        config = getattr(self, '_sea_config', {})

        if Dimensions is not None:
            Signal.__init__(self, data=None)

        tensor_attrs = config.get('tensor_attrs', [])
        path_attrs = config.get('path_attrs', [])
        tuple_list_attrs = config.get('tuple_list_attrs', [])
        exclude_attrs = config.get('exclude_attrs', [])
        default_attrs = config.get('default_attrs', {})

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

        # Seed PySlice private fields so sea-eco's generic loader maps public
        # storage keys (``kxs``, ``time``, ``array``) back onto private attrs.
        for attr in tensor_attrs + path_attrs + tuple_list_attrs:
            if attr.startswith('_') and not hasattr(self, attr):
                setattr(self, attr, None)

        # Initialize excluded attrs to None
        for attr in exclude_attrs:
            setattr(self, attr, None)

        # Initialize user-specified defaults
        for attr, val in default_attrs.items():
            setattr(self, attr, val)

        # Let sea-eco rehydrate nested SEA objects such as Provenance,
        # AnalysisCollection, NodeRecord, Dimensions, and Metadata.
        if Dimensions is None:
            raise ImportError("pySEA is required for .sea deserialization")
        super().from_hdf5_group(group)

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
