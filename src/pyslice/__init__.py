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

def _warn_on_pbc_jumps(obj: "Trajectory") -> None:
    """Warn when consecutive frames contain displacements typical of wrapped PBC jumps."""
    try:
        import numpy as np

        positions = np.asarray(obj.positions, dtype=float)
        box = np.asarray(obj.box_matrix, dtype=float)
    except Exception:
        return
    if positions.ndim != 3 or positions.shape[0] < 2:
        return
    step = np.diff(positions, axis=0)
    magnitudes = np.linalg.norm(step, axis=-1)
    max_step = float(magnitudes.max())
    threshold = 5.0
    if box.ndim == 2 and box.shape == (3, 3) and np.all(np.isfinite(box)):
        lengths = np.linalg.norm(box, axis=1)
        lengths = lengths[lengths > 0]
        if lengths.size:
            threshold = max(1.0, 0.5 * float(lengths.min()))
    if max_step > threshold:
        import warnings
        warnings.warn(
            "PySlice detected inter-frame atomic displacements "
            f"up to {max_step:.2f} Å (threshold {threshold:.2f} Å). "
            "This can indicate atoms jumping across a periodic boundary because "
            "MD coordinates were wrapped rather than unwrapped/haircut. "
            "PySlice does not automatically remove these jumps; if they are "
            "unphysical for your analysis, unwrap the trajectory or apply a "
            "haircutting/minimum-image correction before TACAW or other "
            "frame-difference calculations.",
            stacklevel=2,
        )

_pyslice_trajectory_init = Trajectory.__init__


def _pyslice_trajectory_init_with_pbc_warning(self, *args, **kwargs):
    _pyslice_trajectory_init(self, *args, **kwargs)
    _warn_on_pbc_jumps(self)


Trajectory.__init__ = _pyslice_trajectory_init_with_pbc_warning
