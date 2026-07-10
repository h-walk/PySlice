"""SEA integration surface for PySlice data objects."""

from .pyslice_serial import (
    PySliceSerial,
    Signal,
    Dimensions,
    Dimension,
    Metadata,
    GeneralMetadata,
    SEA_ECO_AVAILABLE,
    record_pyslice_operation,
    track_pyslice_action,
)

if SEA_ECO_AVAILABLE:
    from pySEA.sea_eco.architecture.base_structure import (
        SEAFile,
        SignalCollection,
        SignalQuantities,
        SignalSet,
    )
    from pySEA.sea_eco.io import load
else:
    SEAFile = None
    SignalCollection = None
    SignalQuantities = None
    SignalSet = None

    def load(*args, **kwargs):
        raise ImportError("pySEA is required for .sea loading")

AcquisitionSet = SignalSet

__all__ = [
    "PySliceSerial",
    "Signal",
    "Dimensions",
    "Dimension",
    "Metadata",
    "GeneralMetadata",
    "SEA_ECO_AVAILABLE",
    "record_pyslice_operation",
    "track_pyslice_action",
    "SignalQuantities",
    "SignalCollection",
    "SignalSet",
    "AcquisitionSet",
    "SEAFile",
    "load",
]
