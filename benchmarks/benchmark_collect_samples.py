"""Comprehensive benchmark for ShadowProtocol.collect_samples performance.

This benchmark analyzes the performance characteristics of the shadow tomography
sample collection phase, which is typically the computational bottleneck in
ground state estimation workflows.

Tested dimensions:
1. Backend: Qiskit vs Qulacs
2. Parallelization: Serial (n_jobs=1) vs Parallel (n_jobs=2,4,8)
3. System size: H2, H4, H6, H8, H10 chains (4 to 20 qubits)
4. Sample count: 100, 500, 1000 samples

Usage:
    # Run full benchmark (WARNING: may take 30+ minutes)
    python benchmarks/benchmark_collect_samples.py --full

    # Quick benchmark (small systems only)
    python benchmarks/benchmark_collect_samples.py --quick

    # Custom configuration
    python benchmarks/benchmark_collect_samples.py \
        --hydrogen-atoms 2 4 6 \
        --n-samples 100 500 \
        --n-jobs 1 2 4 \
        --repeats 3 \
        --output results.csv

Results are saved to CSV and plotted for analysis.
"""

import argparse
import time
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

from pyscf import gto, scf
from qiskit.quantum_info import Statevector
from shadow_ci.chemistry import MolecularHamiltonian
from shadow_ci.solvers import FCISolver
from shadow_ci.shadows import ShadowProtocol
from shadow_ci.utils import make_hydrogen_chain


def setup_hydrogen_system(n_hydrogen: int, bond_length: float = 0.74):
    """Setup hydrogen chain system and return trial state.

    Args:
        n_hydrogen: Number of hydrogen atoms
        bond_length: Interatomic distance in Angstroms

    Returns:
        Tuple of (trial_statevector, n_qubits, hamiltonian)
    """
    # Build molecule
    mol_string = make_hydrogen_chain(n_hydrogen, bond_length)
    mol = gto.Mole()
    mol.build(atom=mol_string, basis="sto-3g", verbose=0)

    # Get Hamiltonian
    mf = scf.RHF(mol)
    mf.verbose = 0
    hamiltonian = MolecularHamiltonian.from_pyscf(mf)

    # Get FCI ground state as trial state
    fci_solver = FCISolver(hamiltonian)
    trial_state, _ = fci_solver.solve()

    n_qubits = trial_state.num_qubits

    return trial_state, n_qubits, hamiltonian


def benchmark_collect_samples(
    trial_state: Statevector,
    n_samples: int,
    n_estimators: int,
    use_qulacs: bool,
    n_jobs: int
) -> float:
    """Benchmark a single collect_samples call.

    Args:
        trial_state: Quantum state to sample from
        n_samples: Number of shadow samples
        n_estimators: Number of median-of-means estimators
        use_qulacs: Whether to use Qulacs backend
        n_jobs: Number of parallel workers

    Returns:
        Elapsed time in seconds
    """
    protocol = ShadowProtocol(
        trial_state,
        ensemble_type='clifford',
        use_qulacs=use_qulacs,
        n_jobs=n_jobs
    )

    start_time = time.perf_counter()
    protocol.collect_samples(n_samples, n_estimators, prediction='overlap')
    elapsed_time = time.perf_counter() - start_time

    return elapsed_time


def run_benchmark_suite(
    hydrogen_atoms_list,
    n_samples_list,
    n_jobs_list,
    n_estimators=10,
    repeats=3,
    test_qiskit=True,
    test_qulacs=True
):
    """Run comprehensive benchmark suite.

    Args:
        hydrogen_atoms_list: List of hydrogen chain lengths to test
        n_samples_list: List of sample counts to test
        n_jobs_list: List of thread counts to test
        n_estimators: Number of median-of-means estimators (fixed)
        repeats: Number of times to repeat each configuration
        test_qiskit: Whether to test Qiskit backend
        test_qulacs: Whether to test Qulacs backend

    Returns:
        pandas DataFrame with benchmark results
    """
    results = []
    total_configs = (
        len(hydrogen_atoms_list) *
        len(n_samples_list) *
        len(n_jobs_list) *
        (int(test_qiskit) + int(test_qulacs)) *
        repeats
    )

    config_idx = 0

    print("=" * 80)
    print("Shadow Protocol collect_samples() Benchmark Suite")
    print("=" * 80)
    print(f"Total configurations to test: {total_configs}")
    print(f"Repeats per configuration: {repeats}")
    print(f"Backends: ", end="")
    if test_qiskit: print("Qiskit ", end="")
    if test_qulacs: print("Qulacs ", end="")
    print()
    print("=" * 80)

    # Cache systems to avoid rebuilding
    systems_cache = {}

    for n_hydrogen in hydrogen_atoms_list:
        print(f"\n{'=' * 80}")
        print(f"Setting up H{n_hydrogen} chain system...")
        print(f"{'=' * 80}")

        # Setup system once and cache it
        try:
            trial_state, n_qubits, hamiltonian = setup_hydrogen_system(n_hydrogen)
            systems_cache[n_hydrogen] = (trial_state, n_qubits, hamiltonian)
            print(f"  ✓ System ready: {n_qubits} qubits")
            print(f"  ✓ HF Energy: {hamiltonian.hf_energy:.6f} Ha")
        except Exception as e:
            print(f"  ✗ Failed to setup H{n_hydrogen}: {e}")
            continue

        for n_samples in n_samples_list:
            # Check divisibility
            if n_samples % n_estimators != 0:
                print(f"  ⚠ Skipping n_samples={n_samples} (not divisible by {n_estimators})")
                continue

            for n_jobs in n_jobs_list:
                backends = []
                if test_qiskit:
                    backends.append(('Qiskit', False))
                if test_qulacs:
                    backends.append(('Qulacs', True))

                for backend_name, use_qulacs in backends:
                    config_idx += 1
                    progress = (config_idx / total_configs) * 100

                    print(f"\n[{config_idx}/{total_configs}] ({progress:.1f}%) "
                          f"H{n_hydrogen}, N={n_samples}, jobs={n_jobs}, {backend_name}")

                    times = []
                    for rep in range(repeats):
                        try:
                            elapsed = benchmark_collect_samples(
                                trial_state=trial_state,
                                n_samples=n_samples,
                                n_estimators=n_estimators,
                                use_qulacs=use_qulacs,
                                n_jobs=n_jobs
                            )
                            times.append(elapsed)
                            print(f"  Rep {rep+1}/{repeats}: {elapsed:.4f} s "
                                  f"({n_samples/elapsed:.1f} samples/s)")
                        except Exception as e:
                            print(f"  ✗ Rep {rep+1} failed: {e}")
                            times.append(np.nan)

                    # Compute statistics
                    valid_times = [t for t in times if not np.isnan(t)]
                    if valid_times:
                        mean_time = np.mean(valid_times)
                        std_time = np.std(valid_times)
                        min_time = np.min(valid_times)
                        max_time = np.max(valid_times)

                        results.append({
                            'n_hydrogen': n_hydrogen,
                            'n_qubits': n_qubits,
                            'n_samples': n_samples,
                            'n_jobs': n_jobs,
                            'backend': backend_name,
                            'mean_time_s': mean_time,
                            'std_time_s': std_time,
                            'min_time_s': min_time,
                            'max_time_s': max_time,
                            'samples_per_sec': n_samples / mean_time,
                            'ms_per_sample': 1000 * mean_time / n_samples,
                            'repeats': len(valid_times)
                        })

                        print(f"  → Mean: {mean_time:.4f} ± {std_time:.4f} s")
                        print(f"  → Throughput: {n_samples/mean_time:.1f} samples/s")

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame):
    """Print analysis summary of benchmark results.

    Args:
        df: DataFrame with benchmark results
    """
    print("\n" + "=" * 80)
    print("BENCHMARK ANALYSIS SUMMARY")
    print("=" * 80)

    # Backend comparison (averaged across all configurations)
    print("\n1. Backend Performance (Overall Average)")
    print("-" * 80)
    backend_perf = df.groupby('backend').agg({
        'mean_time_s': 'mean',
        'samples_per_sec': 'mean',
        'ms_per_sample': 'mean'
    })
    print(backend_perf)

    if 'Qiskit' in df['backend'].values and 'Qulacs' in df['backend'].values:
        qiskit_mean = df[df['backend'] == 'Qiskit']['mean_time_s'].mean()
        qulacs_mean = df[df['backend'] == 'Qulacs']['mean_time_s'].mean()
        speedup = qiskit_mean / qulacs_mean
        print(f"\n  → Qulacs Speedup: {speedup:.2f}x faster than Qiskit (average)")

    # Parallelization efficiency
    print("\n2. Parallelization Efficiency")
    print("-" * 80)
    for backend in df['backend'].unique():
        print(f"\n{backend}:")
        backend_df = df[df['backend'] == backend]
        parallel_perf = backend_df.groupby('n_jobs').agg({
            'mean_time_s': 'mean',
            'samples_per_sec': 'mean'
        }).sort_index()
        print(parallel_perf)

        # Compute speedup relative to serial
        if 1 in parallel_perf.index:
            serial_time = parallel_perf.loc[1, 'mean_time_s']
            for n_jobs in parallel_perf.index:
                if n_jobs > 1:
                    parallel_time = parallel_perf.loc[n_jobs, 'mean_time_s']
                    speedup = serial_time / parallel_time
                    efficiency = speedup / n_jobs * 100
                    print(f"  → {n_jobs} threads: {speedup:.2f}x speedup ({efficiency:.1f}% efficiency)")

    # System size scaling
    print("\n3. Scaling with System Size")
    print("-" * 80)
    size_scaling = df.groupby(['n_hydrogen', 'backend']).agg({
        'mean_time_s': 'mean',
        'ms_per_sample': 'mean'
    })
    print(size_scaling)

    # Recommendations
    print("\n4. Recommendations")
    print("-" * 80)

    # Find optimal configuration for each system size
    for n_h in sorted(df['n_hydrogen'].unique()):
        subset = df[df['n_hydrogen'] == n_h]
        best_config = subset.loc[subset['mean_time_s'].idxmin()]
        n_qubits = int(best_config['n_qubits'])

        print(f"\nH{n_h} chain ({n_qubits} qubits):")
        print(f"  Best: {best_config['backend']}, {int(best_config['n_jobs'])} threads")
        print(f"  Time: {best_config['mean_time_s']:.4f} s for {int(best_config['n_samples'])} samples")
        print(f"  Throughput: {best_config['samples_per_sec']:.1f} samples/s")

        # Estimate time for large run
        if best_config['samples_per_sec'] > 0:
            large_samples = 10000
            est_time = large_samples / best_config['samples_per_sec']
            print(f"  Estimated time for 10k samples: {est_time:.1f} s ({est_time/60:.1f} min)")


def plot_results(df: pd.DataFrame, output_path: str):
    """Create comprehensive plots of benchmark results.

    Args:
        df: DataFrame with benchmark results
        output_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    import sys
    from pathlib import Path

    # Try to import plotting config from scripts directory
    scripts_path = Path(__file__).parent.parent / 'scripts'
    if scripts_path.exists():
        sys.path.insert(0, str(scripts_path))
        try:
            from plotting_config import setup_plotting_style
            setup_plotting_style()
        except ImportError:
            print("  ⚠ Could not import plotting_config, using default matplotlib style")
    else:
        print("  ⚠ Using default matplotlib style")

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Backend comparison
    ax = axes[0, 0]
    for backend in df['backend'].unique():
        subset = df[(df['backend'] == backend) & (df['n_jobs'] == 1)]
        grouped = subset.groupby('n_qubits')['mean_time_s'].mean()
        ax.semilogy(grouped.index, grouped.values, 'o-', label=backend, linewidth=2, markersize=6)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Time (s) [log scale]')
    ax.set_title('Backend Comparison (Serial)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Parallelization speedup
    ax = axes[0, 1]
    for backend in df['backend'].unique():
        subset = df[df['backend'] == backend]
        grouped = subset.groupby('n_jobs')['samples_per_sec'].mean()
        ax.plot(grouped.index, grouped.values, 'o-', label=backend, linewidth=2, markersize=6)
    ax.set_xlabel('Number of Threads')
    ax.set_ylabel('Throughput (samples/s)')
    ax.set_title('Parallelization Efficiency')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Scaling with samples
    ax = axes[1, 0]
    for backend in df['backend'].unique():
        subset = df[(df['backend'] == backend) & (df['n_jobs'] == 1)]
        # Pick a medium-sized system
        if len(subset) > 0:
            n_h_mid = sorted(subset['n_hydrogen'].unique())[len(subset['n_hydrogen'].unique())//2]
            subset_size = subset[subset['n_hydrogen'] == n_h_mid]
            if len(subset_size) > 0:
                grouped = subset_size.groupby('n_samples')['mean_time_s'].mean()
                ax.plot(grouped.index, grouped.values, 'o-', label=f'{backend}', linewidth=2, markersize=6)
    ax.set_xlabel('Number of Samples')
    ax.set_ylabel('Time (s)')
    ax.set_title(f'Scaling with Sample Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Time per sample vs system size
    ax = axes[1, 1]
    for backend in df['backend'].unique():
        subset = df[(df['backend'] == backend) & (df['n_jobs'] == 1)]
        grouped = subset.groupby('n_qubits')['ms_per_sample'].mean()
        ax.semilogy(grouped.index, grouped.values, 'o-', label=backend, linewidth=2, markersize=6)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Time per Sample (ms) [log scale]')
    ax.set_title('Sample Cost vs System Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Plot saved to: {output_path}")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description='Benchmark ShadowProtocol.collect_samples performance',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument('--hydrogen-atoms', nargs='+', type=int,
                        default=[2, 4],
                        help='List of hydrogen chain lengths to test (default: 2 4)')
    parser.add_argument('--n-samples', nargs='+', type=int,
                        default=[100, 500],
                        help='List of sample counts to test (default: 100 500)')
    parser.add_argument('--n-jobs', nargs='+', type=int,
                        default=[1, 2],
                        help='List of thread counts to test (default: 1 2)')
    parser.add_argument('--n-estimators', type=int, default=10,
                        help='Number of median-of-means estimators (default: 10)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='Number of times to repeat each config (default: 3)')
    parser.add_argument('--no-qiskit', action='store_true',
                        help='Skip Qiskit backend tests')
    parser.add_argument('--no-qulacs', action='store_true',
                        help='Skip Qulacs backend tests')
    parser.add_argument('--output', type=str, default='collect_samples_benchmark.csv',
                        help='Output CSV file path (default: collect_samples_benchmark.csv)')
    parser.add_argument('--plot', type=str, default='collect_samples_benchmark.png',
                        help='Output plot file path (default: collect_samples_benchmark.png)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: H2, H4, 100 samples, 1-2 threads')
    parser.add_argument('--full', action='store_true',
                        help='Full test: H2-H10, various samples/threads (WARNING: slow!)')

    args = parser.parse_args()

    # Apply presets
    if args.quick:
        args.hydrogen_atoms = [2, 4]
        args.n_samples = [100]
        args.n_jobs = [1, 2]
        args.repeats = 2
    elif args.full:
        args.hydrogen_atoms = [2, 4, 6, 8, 10]
        args.n_samples = [100, 500, 1000]
        args.n_jobs = [1, 2, 4, 8]
        args.repeats = 3

    # Run benchmark suite
    df = run_benchmark_suite(
        hydrogen_atoms_list=args.hydrogen_atoms,
        n_samples_list=args.n_samples,
        n_jobs_list=args.n_jobs,
        n_estimators=args.n_estimators,
        repeats=args.repeats,
        test_qiskit=not args.no_qiskit,
        test_qulacs=not args.no_qulacs
    )

    # Save results
    df.to_csv(args.output, index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {args.output}")
    print(f"{'=' * 80}")

    # Analyze and print summary
    analyze_results(df)

    # Create plots
    plot_results(df, args.plot)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
