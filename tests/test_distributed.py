"""Distributed map/reduce regressions (CPU-simulated multi-rank)."""
from types import SimpleNamespace

import numpy as np
import pytest

from pyslice.backend import NumpyBackend, TORCH_AVAILABLE, to_numpy
if TORCH_AVAILABLE:
    from pyslice.backend import TorchBackend
from pyslice.multislice import calculators as calculators_module
from pyslice.multislice.calculators import MultisliceCalculator
from pyslice.multislice.multislice import PrismProbe, Probe, Propagate, wavelength
from pyslice.multislice.potentials import Potential
from pyslice.multislice.trajectory import Trajectory
from pyslice.io.loader import Loader
from pyslice.postprocessing.haadf_data import HAADFData
from pyslice.postprocessing.tacaw_data import TACAWData
from pyslice.postprocessing.wf_data import WFData



from _regression_helpers import *  # shared builders (see __all__)


def test_distributed_ensemble_equals_serial_on_cpu(tmp_path):
    # The distributed map-reduce (partial per rank -> file-based reduce) must
    # reproduce the serial ensemble average. Simulated multi-rank on CPU.
    from pyslice.postprocessing.tacaw_data import (
        TACAWData, TACAWAccumulator, reduce_tacaw_partials)
    from pyslice.multislice.distributed import assign_units, run_tacaw_ensemble, dist_env

    rng = np.random.RandomState(0)
    n = 64
    sigs = [rng.randn(2, n) + 1j * rng.randn(2, n) for _ in range(16)]   # 16 traj, 2 probes
    producers = [(lambda s=s, i=i: _multiprobe_wf(tmp_path / f"t{i}", s))
                 for i, s in enumerate(sigs)]

    # assignment partitions the 16 units across 4 ranks
    seen = set()
    for r in range(4):
        u = assign_units(16, r, 4)
        assert len(u) == 4
        seen |= set(u)
    assert seen == set(range(16))
    assert dist_env() == (0, 1, 0)                       # single-process fallback

    # 4 ranks each write a partial; the reduce equals the serial ensemble
    out = tmp_path / "reduce"
    for r in range(4):
        run_tacaw_ensemble(producers, out, rank=r, world=4)
    dist_spec = to_numpy(reduce_tacaw_partials(out)._array)
    serial = to_numpy(TACAWData([p() for p in producers])._array)
    np.testing.assert_allclose(dist_spec, serial, atol=1e-9)

    # and the file-based reduce equals a single streaming accumulator
    acc = TACAWAccumulator()
    for p in producers:
        acc.add(p())
    np.testing.assert_allclose(dist_spec, to_numpy(acc.finalize()._array), atol=1e-9)

    # fault tolerance: a missing rank's partial just drops its trajectories
    import glob
    import os
    os.remove(sorted(glob.glob(str(out / "partial_*.npz")))[3])
    partial = to_numpy(reduce_tacaw_partials(out)._array)
    serial12 = to_numpy(TACAWData([producers[i]() for i in range(16) if i % 4 != 3])._array)
    np.testing.assert_allclose(partial, serial12, atol=1e-9)

def test_distributed_probe_batched_reduce(tmp_path):
    # Probe batching across ranks: each rank owns a probe subset; the reduce
    # stitches the full-scan spectrum together.
    from pyslice.postprocessing.tacaw_data import TACAWData, reduce_tacaw_partials
    from pyslice.multislice.distributed import run_tacaw_ensemble
    rng = np.random.RandomState(1)
    n = 64
    full_sig = rng.randn(4, n) + 1j * rng.randn(4, n)          # 4 probes
    # one unit per probe; unit i produces a 1-probe WFData for probe i, row i
    producers = [(lambda i=i: _multiprobe_wf(tmp_path / f"p{i}", full_sig[[i]]))
                 for i in range(4)]
    out = tmp_path / "reduce"
    for r in range(2):     # 2 ranks, 2 probes each
        run_tacaw_ensemble(producers, out, rank=r, world=2,
                           n_probes=4, rows_of=lambda i: [i])
    spec = to_numpy(reduce_tacaw_partials(out)._array)
    full = to_numpy(TACAWData(_multiprobe_wf(tmp_path / "full", full_sig))._array)
    assert spec.shape[0] == 4
    np.testing.assert_allclose(spec, full, atol=1e-9)
