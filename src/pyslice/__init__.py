"""Intentional top-level imports for PySlice.

The stable workflow core, optional integrations, advanced building blocks,
experimental SED objects, and historical compatibility helpers are introduced
in ``docs/user-guide/index.md`` and the linked workflow guides. Inclusion in
:data:`__all__` means a name is an intentional convenience import; it is not by
itself a blanket stability tier. Public docstrings are the current API source.
"""

try:
    from importlib.metadata import version as _package_version
    __version__ = _package_version("pyslice")
except Exception:
    __version__ = "dev"

from .backend import (
    Backend,
    NumpyBackend,
    TORCH_AVAILABLE,
    TorchBackend,
    make_backend,
    to_cpu,
    to_numpy,
)
from .io.databases import DatabaseError, search_structures, fetch_cif, load_structure_from_database
from .io.loader import Loader
from .md.molecular_dynamics import (
    FAIRChemMDCalculator,
    MDCalculator,
    MDConvergenceChecker,
    ORBMDCalculator,
    analyze_md_trajectory,
)
from .multislice.calculators import MultisliceCalculator, SEDCalculator
from .multislice.multislice import (
    Probe,
    PrismProbe,
    Propagate,
    aberrationFunction,
    calculateObject,
    create_batched_probes,
    wavelength,
)
from .multislice.potentials import Potential, grid_from_trajectory
from .multislice.sed import SED
from .multislice.trajectory import Trajectory
from .postprocessing.haadf_data import HAADFData
from .postprocessing.tacaw_data import SEDData, TACAWData, bose_correction_factor
from .postprocessing.testtools import differ
from .postprocessing.wf_data import WFData

__all__ = (
    "__version__",
    # Primary workflow
    "Loader",
    "Trajectory",
    "MultisliceCalculator",
    "WFData",
    "TACAWData",
    "HAADFData",
    # Structure database integrations
    "DatabaseError",
    "search_structures",
    "fetch_cif",
    "load_structure_from_database",
    # Molecular dynamics
    "MDCalculator",
    "ORBMDCalculator",
    "FAIRChemMDCalculator",
    "MDConvergenceChecker",
    "analyze_md_trajectory",
    # Advanced multislice building blocks
    "Probe",
    "PrismProbe",
    "Potential",
    "Propagate",
    "create_batched_probes",
    "grid_from_trajectory",
    "wavelength",
    "aberrationFunction",
    "calculateObject",
    # Spectral-energy-density analysis
    "SED",
    "SEDData",
    "SEDCalculator",
    "bose_correction_factor",
    # Backend interoperation
    "Backend",
    "NumpyBackend",
    "TorchBackend",
    "TORCH_AVAILABLE",
    "make_backend",
    "to_cpu",
    "to_numpy",
    # Historical validation helper used by the scientific regression scripts
    "differ",
)
