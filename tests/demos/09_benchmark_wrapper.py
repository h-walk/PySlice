"""
Benchmark wrapper - runs each test in a fresh Python process to avoid Dask caching.
Now performs multiple simulations for statistical analysis.
"""
import subprocess
import json
import numpy as np
import matplotlib.pyplot as plt

# Test parameters
supercell_sizes = [8, 10, 12, 15, 18, 20, 25, 30]
probe_counts = [1, 4, 9, 16, 25, 49, 64, 100]
n_runs = 3  # Number of simulations per configuration

# Store all runs for statistical analysis
system_results = {
    'sizes': [],
    'n_atoms': [],
    'pyslice_runs': [],  # Will store list of lists
    'abtem_runs': [],    # Will store list of lists
    'pyslice_mean': [],
    'pyslice_std': [],
    'abtem_mean': [],
    'abtem_std': [],
    'speedup_mean': [],
    'speedup_std': []
}

probe_results = {
    'counts': [],
    'pyslice_runs': [],  # Will store list of lists
    'abtem_runs': [],    # Will store list of lists
    'pyslice_mean': [],
    'pyslice_std': [],
    'abtem_mean': [],
    'abtem_std': [],
    'speedup_mean': [],
    'speedup_std': []
}

print("="*50)
print("SYSTEM SIZE SCALING (single probe)")
print(f"Running {n_runs} simulations per configuration")
print("="*50)

for size in supercell_sizes:
    print(f"\n--- Testing {size}³ supercell ---")

    pyslice_times = []
    abtem_times = []
    speedups = []
    n_atoms_val = None

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", end=" ", flush=True)

        # Run in fresh Python process
        result = subprocess.run(
            ['python3', '09_benchmark_single.py', 'system', str(size)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            continue

        # Read results from file
        with open(f'result_system_{size}.txt', 'r') as f:
            line = f.read().strip()
            size_val, n_atoms, pyslice_time, abtem_time = line.split(',')
            n_atoms_val = int(n_atoms)
            pyslice_time = float(pyslice_time)
            abtem_time = float(abtem_time)

        pyslice_times.append(pyslice_time)
        abtem_times.append(abtem_time)
        speedups.append(abtem_time / pyslice_time)

        print(f"PySlice: {pyslice_time:.3f}s, AbTem: {abtem_time:.3f}s")

    if pyslice_times:  # Only process if we have successful runs
        # Calculate statistics
        pyslice_mean = np.mean(pyslice_times)
        pyslice_std = np.std(pyslice_times)
        abtem_mean = np.mean(abtem_times)
        abtem_std = np.std(abtem_times)
        speedup_mean = np.mean(speedups)
        speedup_std = np.std(speedups)

        print(f"\n  Summary ({n_atoms_val} atoms):")
        print(f"    PySlice: {pyslice_mean:.3f} ± {pyslice_std:.3f}s")
        print(f"    AbTem: {abtem_mean:.3f} ± {abtem_std:.3f}s")
        print(f"    Speedup: {speedup_mean:.2f} ± {speedup_std:.2f}×")

        # Store results
        system_results['sizes'].append(int(size_val))
        system_results['n_atoms'].append(n_atoms_val)
        system_results['pyslice_runs'].append(pyslice_times)
        system_results['abtem_runs'].append(abtem_times)
        system_results['pyslice_mean'].append(pyslice_mean)
        system_results['pyslice_std'].append(pyslice_std)
        system_results['abtem_mean'].append(abtem_mean)
        system_results['abtem_std'].append(abtem_std)
        system_results['speedup_mean'].append(speedup_mean)
        system_results['speedup_std'].append(speedup_std)

print("\n" + "="*50)
print("MULTI-PROBE SCALING (10³ supercell)")
print(f"Running {n_runs} simulations per configuration")
print("="*50)

for n_probes in probe_counts:
    print(f"\n--- Testing {n_probes} probes ---")

    pyslice_times = []
    abtem_times = []
    speedups = []

    for run in range(n_runs):
        print(f"  Run {run+1}/{n_runs}...", end=" ", flush=True)

        # Run in fresh Python process
        result = subprocess.run(
            ['python3', '09_benchmark_single.py', 'probe', str(n_probes)],
            capture_output=True,
            text=True
        )

        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            continue

        # Read results from file
        with open(f'result_probe_{n_probes}.txt', 'r') as f:
            line = f.read().strip()
            n_probes_val, pyslice_time, abtem_time = line.split(',')
            n_probes_val = int(n_probes_val)
            pyslice_time = float(pyslice_time)
            abtem_time = float(abtem_time)

        pyslice_times.append(pyslice_time)
        abtem_times.append(abtem_time)
        speedups.append(abtem_time / pyslice_time)

        print(f"PySlice: {pyslice_time:.3f}s, AbTem: {abtem_time:.3f}s")

    if pyslice_times:  # Only process if we have successful runs
        # Calculate statistics
        pyslice_mean = np.mean(pyslice_times)
        pyslice_std = np.std(pyslice_times)
        abtem_mean = np.mean(abtem_times)
        abtem_std = np.std(abtem_times)
        speedup_mean = np.mean(speedups)
        speedup_std = np.std(speedups)

        print(f"\n  Summary:")
        print(f"    PySlice: {pyslice_mean:.3f} ± {pyslice_std:.3f}s ({pyslice_mean/n_probes:.4f}s/probe)")
        print(f"    AbTem: {abtem_mean:.3f} ± {abtem_std:.3f}s ({abtem_mean/n_probes:.4f}s/probe)")
        print(f"    Speedup: {speedup_mean:.2f} ± {speedup_std:.2f}×")

        # Store results
        probe_results['counts'].append(n_probes_val)
        probe_results['pyslice_runs'].append(pyslice_times)
        probe_results['abtem_runs'].append(abtem_times)
        probe_results['pyslice_mean'].append(pyslice_mean)
        probe_results['pyslice_std'].append(pyslice_std)
        probe_results['abtem_mean'].append(abtem_mean)
        probe_results['abtem_std'].append(abtem_std)
        probe_results['speedup_mean'].append(speedup_mean)
        probe_results['speedup_std'].append(speedup_std)

# Plot results with error bars
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# System size scaling
if system_results['sizes']:
    ax1.errorbar(system_results['sizes'], system_results['pyslice_mean'],
                 yerr=system_results['pyslice_std'], fmt='o-', linewidth=2,
                 markersize=8, capsize=5, capthick=2, label='PySlice')
    ax1.errorbar(system_results['sizes'], system_results['abtem_mean'],
                 yerr=system_results['abtem_std'], fmt='s-', linewidth=2,
                 markersize=8, capsize=5, capthick=2, label='AbTem')
    ax1.set_xlabel('Supercell Size', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title(f'System Size Scaling (1 probe, {n_runs} runs)', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

# Multi-probe scaling
if probe_results['counts']:
    ax2.errorbar(probe_results['counts'], probe_results['pyslice_mean'],
                 yerr=probe_results['pyslice_std'], fmt='o-', linewidth=2,
                 markersize=8, capsize=5, capthick=2, label='PySlice')
    ax2.errorbar(probe_results['counts'], probe_results['abtem_mean'],
                 yerr=probe_results['abtem_std'], fmt='s-', linewidth=2,
                 markersize=8, capsize=5, capthick=2, label='AbTem')
    ax2.set_xlabel('Number of Probes', fontsize=12)
    ax2.set_ylabel('Time (seconds)', fontsize=12)
    ax2.set_title(f'Multi-Probe Scaling (10³ supercell, {n_runs} runs)', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('benchmark_no_cache_with_stats.png', dpi=300)
print("\nPlot saved to benchmark_no_cache_with_stats.png")
plt.show()

# Print summary
print("\n" + "="*50)
print("BENCHMARK RESULTS WITH STATISTICS")
print("="*50)

if system_results['sizes']:
    print(f"\nSYSTEM SIZE SCALING (single probe, {n_runs} runs per configuration):")
    print(f"{'Size':>5} {'Atoms':>6} {'PySlice(s)':>20} {'AbTem(s)':>20} {'Speedup':>15}")
    print("-"*85)
    for i in range(len(system_results['sizes'])):
        pyslice_str = f"{system_results['pyslice_mean'][i]:.3f} ± {system_results['pyslice_std'][i]:.3f}"
        abtem_str = f"{system_results['abtem_mean'][i]:.3f} ± {system_results['abtem_std'][i]:.3f}"
        speedup_str = f"{system_results['speedup_mean'][i]:.2f} ± {system_results['speedup_std'][i]:.2f}×"
        print(f"{system_results['sizes'][i]:>5} {system_results['n_atoms'][i]:>6} "
              f"{pyslice_str:>20} {abtem_str:>20} {speedup_str:>15}")
    avg_speedup = np.mean(system_results['speedup_mean'])
    avg_speedup_std = np.mean(system_results['speedup_std'])
    print(f"Average speedup: {avg_speedup:.2f} ± {avg_speedup_std:.2f}×")

if probe_results['counts']:
    print(f"\nMULTI-PROBE SCALING (10³ supercell, {n_runs} runs per configuration):")
    print(f"{'Probes':>6} {'PySlice(s)':>20} {'AbTem(s)':>20} {'Speedup':>15}")
    print("-"*70)
    for i in range(len(probe_results['counts'])):
        pyslice_str = f"{probe_results['pyslice_mean'][i]:.3f} ± {probe_results['pyslice_std'][i]:.3f}"
        abtem_str = f"{probe_results['abtem_mean'][i]:.3f} ± {probe_results['abtem_std'][i]:.3f}"
        speedup_str = f"{probe_results['speedup_mean'][i]:.2f} ± {probe_results['speedup_std'][i]:.2f}×"
        print(f"{probe_results['counts'][i]:>6} {pyslice_str:>20} {abtem_str:>20} {speedup_str:>15}")
    avg_speedup = np.mean(probe_results['speedup_mean'])
    avg_speedup_std = np.mean(probe_results['speedup_std'])
    print(f"Average speedup: {avg_speedup:.2f} ± {avg_speedup_std:.2f}×")

