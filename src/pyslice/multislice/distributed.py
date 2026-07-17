"""Distributed / multi-GPU orchestration for ensemble TACAW (Phase 3).

The reduce is deliberately map-reduce and file-based so it works across nodes on
a shared filesystem, is fault tolerant, and is fully testable on a single CPU:

    map    : each rank accumulates its assigned (trajectory, probe-batch) units
             into a host-resident TACAWAccumulator and writes one partial .npz.
    reduce : reduce_tacaw_partials() sums the partials -> the averaged TACAWData.

Multiple GPUs are just multiple ranks (one process per GPU, pinned to
``cuda:{local_rank}``). Nothing here needs torch.distributed/NCCL; that is only a
single-node fast path for small outputs (see the design note) and is orthogonal.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Union


def dist_env():
    """Return ``(rank, world_size, local_rank)`` from the launcher environment.

    Understands ``torchrun`` (RANK/WORLD_SIZE/LOCAL_RANK) and SLURM
    (SLURM_PROCID/SLURM_NTASKS/SLURM_LOCALID); falls back to a single process
    ``(0, 1, 0)`` when neither is set.
    """
    def geti(*names, default=0):
        for n in names:
            v = os.environ.get(n)
            if v not in (None, ""):
                return int(v)
        return default
    rank = geti('RANK', 'SLURM_PROCID', default=0)
    world = geti('WORLD_SIZE', 'SLURM_NTASKS', default=1)
    local = geti('LOCAL_RANK', 'SLURM_LOCALID', default=0)
    return rank, world, local


def assign_units(n_units: int, rank: int, world: int) -> List[int]:
    """Strided assignment of unit indices to ``rank`` (unit i -> rank i % world).

    Balanced for equal-cost units and stable across runs.
    """
    if world <= 0:
        raise ValueError("world size must be positive")
    if not (0 <= rank < world):
        raise ValueError(f"rank {rank} out of range [0, {world})")
    return list(range(rank, n_units, world))


def run_tacaw_ensemble(
    producers: Sequence[Union[Callable[[], "object"], "object"]],
    out_dir: Union[str, Path],
    *,
    rank: Optional[int] = None,
    world: Optional[int] = None,
    segment_length: Optional[int] = None,
    overlap: float = 0.0,
    window=None,
    n_probes: Optional[int] = None,
    rows_of: Optional[Callable[[int], object]] = None,
    reduce: bool = False,
    backend=None,
):
    """Accumulate this rank's assigned units and write its partial for the reduce.

    Args:
        producers: one entry per work unit -- a WFData, or (preferred) a
            zero-arg callable returning a WFData, so exit waves are produced and
            freed one unit at a time. A unit is a ``(trajectory, probe-batch)``.
        out_dir: shared directory for ``partial_<rank>.npz``.
        rank, world: this process's identity; default from :func:`dist_env`.
        segment_length, overlap, window: Welch parameters (see TACAWData).
        n_probes: total probe count when probe-batching (units cover subsets);
            ``None`` for full-grid trajectories.
        rows_of: ``i -> probe rows`` this unit fills (probe batching); ``None`` =
            all rows.
        reduce: if True, also reduce all partials (call from rank 0 only, after a
            barrier that guarantees every rank has written its partial) and
            return the averaged TACAWData. Otherwise return the partial path.

    Returns:
        The partial-file path, or the averaged ``TACAWData`` if ``reduce=True``.
    """
    from pyslice.postprocessing.tacaw_data import TACAWAccumulator, reduce_tacaw_partials
    if rank is None or world is None:
        r, w, _ = dist_env()
        rank = r if rank is None else rank
        world = w if world is None else world
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    acc = TACAWAccumulator(segment_length=segment_length, overlap=overlap,
                           window=window, n_probes=n_probes)
    my_units = assign_units(len(producers), rank, world)
    for i in my_units:
        item = producers[i]
        wf = item() if callable(item) else item
        acc.add(wf, rows=None if rows_of is None else rows_of(i))
        del wf
    partial_path = out_dir / f"partial_{rank:04d}.npz"
    acc.save_partial(partial_path)

    if reduce:
        return reduce_tacaw_partials(out_dir, backend=backend)
    return str(partial_path)
