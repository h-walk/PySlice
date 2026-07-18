"""
Single benchmark test - runs ONE system size or probe count test.
Called by wrapper script to avoid Dask caching between tests.
"""
import sys,os
import warnings
warnings.filterwarnings('ignore')  # Suppress warnings

# Redirect stderr to suppress progress bars
import io
stderr_backup = sys.stderr
sys.stderr = io.StringIO()

from pyslice.multislice.potentials import Potential
from pyslice.multislice.multislice import Probe, Propagate, create_batched_probes
from pyslice.multislice.trajectory import Trajectory
import numpy as np
import time
from ase.build import bulk
import gc
import json

try:
    import torch
    import dask
    dask.config.set(scheduler='synchronous')
    dask.config.set({'array.cache': None})
    import abtem

    has_cuda = torch.cuda.is_available()
    has_mps = torch.backends.mps.is_available()

    if has_cuda:
        device = 'cuda'
        abtem_device = 'gpu'
        force_cpu = False
    elif has_mps:
        device = 'cpu'
        abtem_device = 'cpu'
        force_cpu = True
    else:
        device = 'cpu'
        abtem_device = 'cpu'
        force_cpu = False

except ImportError:
    print("PyTorch or AbTem not available")
    sys.exit(1)

def create_trajectory(supercell_size):
    atoms = bulk('Cu', 'fcc', a=3.615, cubic=True)
    atoms *= (supercell_size, supercell_size, supercell_size)
    box_x = atoms.cell[0,0]
    box_y = atoms.cell[1,1]
    box_z = atoms.cell[2,2]
    trajectory = Trajectory(
        positions=atoms.positions.reshape(1, -1, 3),
        velocities=np.zeros((1, len(atoms), 3)),
        box_matrix=np.diag([box_x, box_y, box_z]),
        atom_types=np.array([29] * len(atoms)),
        timestep=1.0
    )
    return trajectory, atoms

def probe_positions(box_x, box_y, n_probes):
    if n_probes == 1:
        return [(box_x/2, box_y/2)]
    n_side = int(np.ceil(np.sqrt(n_probes)))
    positions = []
    xs = np.linspace(box_x*0.2, box_x*0.8, n_side)
    ys = np.linspace(box_y*0.2, box_y*0.8, n_side)
    for x in xs:
        for y in ys:
            positions.append((x, y))
            if len(positions) >= n_probes:
                break
        if len(positions) >= n_probes:
            break
    return positions[:n_probes]

if __name__ == "__main__":
    # Parse command line: test_type size/count
    test_type = sys.argv[1]  # 'system' or 'probe'
    test_param = int(sys.argv[2])  # size or probe count

    if test_type == 'system':
        # System size scaling test
        size = test_param
        trajectory, atoms = create_trajectory(size)
        box_x, box_y = atoms.cell[0,0], atoms.cell[1,1]
        positions = trajectory.positions[0]
        atom_types = trajectory.atom_types
        atom_type_names = [29] * len(atom_types)

        dx = dy = 0.1
        dz = 0.5
        xs = np.arange(0, box_x, dx)
        ys = np.arange(0, box_y, dy)
        zs = np.arange(0, trajectory.box_matrix[2,2], dz)
        probe_pos = [(box_x/2, box_y/2)]

        # PySlice test
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

        start = time.perf_counter()
        potential = Potential(xs, ys, zs, positions, atom_type_names, kind="kirkland", device=device)
        probe = Probe(xs, ys, mrad=30.0, eV=200e3, device=device)
        batched_probe = create_batched_probes(probe, probe_pos, device=device)
        exit_wave = Propagate(batched_probe, potential, device=device, progress=False, onthefly=False)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        pyslice_time = end - start
        del potential, probe, batched_probe, exit_wave

        # AbTem test
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

        abtem.config.set({"device": abtem_device})

        atoms_copy = atoms.copy()
        start = time.perf_counter()
        potential = abtem.Potential(atoms_copy, sampling=0.1, slice_thickness=0.5)
        probe = abtem.Probe(energy=200e3, semiangle_cutoff=30.0, defocus=0.0)
        probe.grid.match(potential)
        wave = probe.build()
        exit_wave = wave.multislice(potential)
        result = exit_wave.compute()
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        abtem_time = end - start

        # Write results to file
        with open(f'outputs/result_system_{size}.txt', 'w') as f:
            f.write(f"{size},{len(atoms)},{pyslice_time},{abtem_time}\n")

    elif test_type == 'probe':
        # Multi-probe test
        n_probes = test_param
        trajectory, atoms = create_trajectory(10)
        box_x, box_y = atoms.cell[0,0], atoms.cell[1,1]
        positions = trajectory.positions[0]
        atom_types = trajectory.atom_types
        atom_type_names = [29] * len(atom_types)

        dx = dy = 0.1
        dz = 0.5
        xs = np.arange(0, box_x, dx)
        ys = np.arange(0, box_y, dy)
        zs = np.arange(0, trajectory.box_matrix[2,2], dz)
        probe_pos = probe_positions(box_x, box_y, n_probes)

        # PySlice test
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

        start = time.perf_counter()
        potential = Potential(xs, ys, zs, positions, atom_type_names, kind="kirkland", device=device)
        probe = Probe(xs, ys, mrad=30.0, eV=200e3, device=device)
        batched_probe = create_batched_probes(probe, probe_pos, device=device)
        exit_waves = Propagate(batched_probe, potential, device=device, progress=False, onthefly=False)
        if device == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        pyslice_time = end - start
        del potential, probe, batched_probe, exit_waves

        # AbTem test
        gc.collect()
        if device == 'cuda':
            torch.cuda.empty_cache()

        abtem.config.set({"device": abtem_device})

        if n_probes == 1:
            start = time.perf_counter()
            atoms_copy = atoms.copy()
            potential = abtem.Potential(atoms_copy, sampling=0.1, slice_thickness=0.5)
            probe = abtem.Probe(energy=200e3, semiangle_cutoff=30.0, defocus=0.0)
            probe.grid.match(potential)
            wave = probe.build()
            exit_wave = wave.multislice(potential)
            result = exit_wave.compute()
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
        else:
            positions_list = probe_positions(box_x, box_y, n_probes)
            scan_pos = [(x/box_x, y/box_y) for x,y in positions_list]
            n_side = int(np.ceil(np.sqrt(n_probes)))
            start_xy = [min(p[0] for p in scan_pos), min(p[1] for p in scan_pos)]
            end_xy = [max(p[0] for p in scan_pos), max(p[1] for p in scan_pos)]
            if start_xy[0] == end_xy[0]: end_xy[0] += 0.01
            if start_xy[1] == end_xy[1]: end_xy[1] += 0.01
            scan = abtem.GridScan(start=start_xy, end=end_xy, gpts=[n_side, n_side])
            detector = abtem.FlexibleAnnularDetector()

            start = time.perf_counter()
            atoms_copy = atoms.copy()
            potential = abtem.Potential(atoms_copy, sampling=0.1, slice_thickness=0.5)
            probe = abtem.Probe(energy=200e3, semiangle_cutoff=30.0, defocus=0.0)
            probe.grid.match(potential)
            result = probe.scan(potential, scan=scan, detectors=detector, max_batch=n_probes).compute()
            if device == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()

        abtem_time = end - start

        # Write results to file
        with open(f'outputs/result_probe_{n_probes}.txt', 'w') as f:
            f.write(f"{n_probes},{pyslice_time},{abtem_time}\n")
