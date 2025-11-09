"""Benchmark shadow protocol scaling with N (number of shots).

This script benchmarks the shadow protocol performance as a function of the number
of shadow samples (N), comparing:
1. Qiskit backend (slower, pure Python)
2. Qulacs backend (faster, C++ implementation)

The benchmark measures:
- Sample collection time vs N
- Overlap estimation time vs N
- Time per sample (should be constant)
- Memory usage (optional)

Run with:
    python benchmarks/benchmark_shadow_scaling.py

Or to save results:
    python benchmarks/benchmark_shadow_scaling.py --output results.csv
"""

import time
import argparse
import warnings
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyscf import gto, scf

from shadow_ci.hamiltonian import MolecularHamiltonian
from shadow_ci.shadows import ShadowProtocol, HAS_QULACS
from shadow_ci.utils import Bitstring


def create_test_molecule(system: str = "H2"):
    """Create a test molecule for benchmarking.

    Args:
        system: Molecular system ("H2", "BeH2", "H2O")

    Returns:
        PySCF molecule object
    """
    if system == "H2":
        mol = gto.Mole()
        mol.build(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    elif system == "BeH2":
        mol = gto.Mole()
        mol.build(atom="Be 0 0 0; H 0 0 1.3; H 0 0 -1.3", basis="sto-3g", verbose=0)
    elif system == "H2O":
        mol = gto.Mole()
        mol.build(atom="O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0", basis="sto-3g", verbose=0)
    elif system == "LiH":
        mol = gto.Mole()
        mol.build(atom="Li 0 0 0; H 0 0 1.6", basis="sto-3g", verbose=0)
    else:
        raise ValueError(f"Unknown system: {system}")

    return mol


def benchmark_sample_collection(
    protocol: ShadowProtocol,
    n_samples: int,
    n_estimators: int,
    n_repeats: int = 3
) -> Dict[str, float]:
    """Benchmark sample collection for a given N.

    Args:
        protocol: ShadowProtocol instance
        n_samples: Number of shadow samples
        n_estimators: Number of k-estimators (for median-of-means)
        n_repeats: Number of timing repetitions

    Returns:
        Dictionary with timing statistics
    """
    times = []

    for _ in range(n_repeats):
        start = time.perf_counter()
        protocol.collect_samples(n_samples, n_estimators, prediction='overlap')
        end = time.perf_counter()
        times.append(end - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
    }


def benchmark_overlap_estimation(
    protocol: ShadowProtocol,
    target_bitstrings: List[Bitstring],
    n_repeats: int = 5
) -> Dict[str, float]:
    """Benchmark overlap estimation time.

    Args:
        protocol: ShadowProtocol with collected samples
        target_bitstrings: List of target bitstrings to estimate overlaps
        n_repeats: Number of timing repetitions

    Returns:
        Dictionary with timing statistics
    """
    if protocol.k_estimators is None:
        raise ValueError("Protocol must have collected samples first")

    times = []

    for _ in range(n_repeats):
        start = time.perf_counter()
        for bitstring in target_bitstrings:
            _ = protocol.estimate_overlap(bitstring)
        end = time.perf_counter()
        times.append(end - start)

    return {
        'mean': np.mean(times),
        'std': np.std(times),
        'min': np.min(times),
        'max': np.max(times),
        'per_overlap': np.mean(times) / len(target_bitstrings),
    }


def run_scaling_benchmark(
    system: str = "H2",
    n_samples_list: List[int] = [10, 20, 50, 100, 200, 500],
    n_estimators: int = 2,
    n_repeats: int = 3,
    test_qiskit: bool = True,
    test_qulacs: bool = True,
) -> pd.DataFrame:
    """Run a complete scaling benchmark.

    Args:
        system: Molecular system to test
        n_samples_list: List of N values to benchmark
        n_estimators: Number of k-estimators (must divide n_samples)
        n_repeats: Number of timing repetitions
        test_qiskit: Whether to test Qiskit backend
        test_qulacs: Whether to test Qulacs backend

    Returns:
        DataFrame with benchmark results
    """
    print(f"=== Shadow Protocol Scaling Benchmark ===")
    print(f"System: {system}")
    print(f"N values: {n_samples_list}")
    print(f"K estimators: {n_estimators}")
    print(f"Repeats per N: {n_repeats}")
    print(f"Qulacs available: {HAS_QULACS}")
    print()

    # Setup molecule and Hamiltonian
    mol = create_test_molecule(system)
    mf = scf.RHF(mol)
    mf.verbose = 0
    mf.kernel()
    hamiltonian = MolecularHamiltonian.from_pyscf(mf)

    # Get trial state (HF state for simplicity)
    trial_state = hamiltonian.get_hf_state()
    n_qubits = trial_state.num_qubits

    print(f"Molecule: {system}")
    print(f"Number of qubits: {n_qubits}")
    print(f"Number of single excitations: {len(hamiltonian.get_single_excitations())}")
    print(f"Number of double excitations: {len(hamiltonian.get_double_excitations())}")
    print()

    # Get some target bitstrings for overlap estimation
    hf_bitstring = hamiltonian.get_hf_bitstring()
    single_excitations = hamiltonian.get_single_excitations()
    target_bitstrings = [hf_bitstring] + [ex.bitstring for ex in single_excitations[:5]]

    results = []

    # Test with Qiskit backend
    if test_qiskit:
        print("--- Benchmarking with Qiskit backend ---")
        for n_samples in n_samples_list:
            if n_samples % n_estimators != 0:
                print(f"Skipping N={n_samples} (not divisible by k={n_estimators})")
                continue

            print(f"N = {n_samples}...", end=" ", flush=True)

            protocol = ShadowProtocol(trial_state, use_qulacs=False)

            # Benchmark sample collection
            collection_stats = benchmark_sample_collection(
                protocol, n_samples, n_estimators, n_repeats
            )

            # Benchmark overlap estimation (need to collect samples once)
            protocol.collect_samples(n_samples, n_estimators, prediction='overlap')
            overlap_stats = benchmark_overlap_estimation(
                protocol, target_bitstrings, n_repeats=5
            )

            results.append({
                'system': system,
                'backend': 'Qiskit',
                'n_samples': n_samples,
                'n_qubits': n_qubits,
                'collection_time_mean': collection_stats['mean'],
                'collection_time_std': collection_stats['std'],
                'time_per_sample': collection_stats['mean'] / n_samples,
                'overlap_time_mean': overlap_stats['mean'],
                'overlap_time_std': overlap_stats['std'],
                'time_per_overlap': overlap_stats['per_overlap'],
            })

            print(f"✓ ({collection_stats['mean']:.3f}s collection, "
                  f"{collection_stats['mean']/n_samples*1000:.1f}ms/sample)")

        print()

    # Test with Qulacs backend
    if test_qulacs:
        if not HAS_QULACS:
            print("--- Qulacs not available, skipping ---")
            print("Install with: pip install qulacs")
            print()
        else:
            print("--- Benchmarking with Qulacs backend ---")
            for n_samples in n_samples_list:
                if n_samples % n_estimators != 0:
                    print(f"Skipping N={n_samples} (not divisible by k={n_estimators})")
                    continue

                print(f"N = {n_samples}...", end=" ", flush=True)

                protocol = ShadowProtocol(trial_state, use_qulacs=True)

                # Benchmark sample collection
                collection_stats = benchmark_sample_collection(
                    protocol, n_samples, n_estimators, n_repeats
                )

                # Benchmark overlap estimation
                protocol.collect_samples(n_samples, n_estimators, prediction='overlap')
                overlap_stats = benchmark_overlap_estimation(
                    protocol, target_bitstrings, n_repeats=5
                )

                results.append({
                    'system': system,
                    'backend': 'Qulacs',
                    'n_samples': n_samples,
                    'n_qubits': n_qubits,
                    'collection_time_mean': collection_stats['mean'],
                    'collection_time_std': collection_stats['std'],
                    'time_per_sample': collection_stats['mean'] / n_samples,
                    'overlap_time_mean': overlap_stats['mean'],
                    'overlap_time_std': overlap_stats['std'],
                    'time_per_overlap': overlap_stats['per_overlap'],
                })

                print(f"✓ ({collection_stats['mean']:.3f}s collection, "
                      f"{collection_stats['mean']/n_samples*1000:.1f}ms/sample)")

            print()

    df = pd.DataFrame(results)
    return df


def plot_results(df: pd.DataFrame, output_path: str = None):
    """Create visualization plots for benchmark results.

    Args:
        df: DataFrame with benchmark results
        output_path: Optional path to save figure
    """
    backends = df['backend'].unique()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Shadow Protocol Scaling Benchmark - {df['system'].iloc[0]}",
                 fontsize=16, fontweight='bold')

    # Plot 1: Total collection time vs N
    ax1 = axes[0, 0]
    for backend in backends:
        data = df[df['backend'] == backend]
        ax1.errorbar(
            data['n_samples'],
            data['collection_time_mean'],
            yerr=data['collection_time_std'],
            marker='o',
            label=backend,
            capsize=5,
            linewidth=2,
            markersize=8
        )
    ax1.set_xlabel('Number of Samples (N)', fontsize=12)
    ax1.set_ylabel('Collection Time (s)', fontsize=12)
    ax1.set_title('Sample Collection Time vs N', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Time per sample (should be roughly constant)
    ax2 = axes[0, 1]
    for backend in backends:
        data = df[df['backend'] == backend]
        ax2.plot(
            data['n_samples'],
            data['time_per_sample'] * 1000,  # Convert to ms
            marker='s',
            label=backend,
            linewidth=2,
            markersize=8
        )
    ax2.set_xlabel('Number of Samples (N)', fontsize=12)
    ax2.set_ylabel('Time per Sample (ms)', fontsize=12)
    ax2.set_title('Time per Sample (should be constant)', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=ax2.get_ylim()[0], color='gray', linestyle='--', alpha=0.5)

    # Plot 3: Overlap estimation time
    ax3 = axes[1, 0]
    for backend in backends:
        data = df[df['backend'] == backend]
        ax3.errorbar(
            data['n_samples'],
            data['overlap_time_mean'],
            yerr=data['overlap_time_std'],
            marker='^',
            label=backend,
            capsize=5,
            linewidth=2,
            markersize=8
        )
    ax3.set_xlabel('Number of Samples (N)', fontsize=12)
    ax3.set_ylabel('Overlap Estimation Time (s)', fontsize=12)
    ax3.set_title('Overlap Estimation Time (6 overlaps)', fontsize=13, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Speedup factor (if both backends available)
    ax4 = axes[1, 1]
    if len(backends) == 2:
        qiskit_data = df[df['backend'] == 'Qiskit'].sort_values('n_samples')
        qulacs_data = df[df['backend'] == 'Qulacs'].sort_values('n_samples')

        if len(qiskit_data) == len(qulacs_data):
            speedup = qiskit_data['collection_time_mean'].values / qulacs_data['collection_time_mean'].values
            ax4.plot(
                qiskit_data['n_samples'],
                speedup,
                marker='D',
                color='green',
                linewidth=2,
                markersize=8,
                label='Qulacs Speedup'
            )
            ax4.axhline(y=1, color='red', linestyle='--', linewidth=2, label='No Speedup')
            ax4.set_xlabel('Number of Samples (N)', fontsize=12)
            ax4.set_ylabel('Speedup Factor', fontsize=12)
            ax4.set_title('Qulacs Speedup over Qiskit', fontsize=13, fontweight='bold')
            ax4.legend()
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'Inconsistent data', ha='center', va='center', transform=ax4.transAxes)
    else:
        ax4.text(0.5, 0.5, 'Single backend tested', ha='center', va='center', transform=ax4.transAxes)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        plt.show()


def print_summary(df: pd.DataFrame):
    """Print a summary of benchmark results.

    Args:
        df: DataFrame with benchmark results
    """
    print("\n=== BENCHMARK SUMMARY ===\n")

    for backend in df['backend'].unique():
        data = df[df['backend'] == backend]
        print(f"--- {backend} Backend ---")
        print(f"Time per sample (mean): {data['time_per_sample'].mean()*1000:.2f} ms")
        print(f"Time per sample (std):  {data['time_per_sample'].std()*1000:.2f} ms")
        print(f"Time per overlap (mean): {data['time_per_overlap'].mean()*1000:.2f} ms")
        print(f"Sample collection scaling: O(N) = {np.polyfit(data['n_samples'], data['collection_time_mean'], 1)[0]:.6f} * N")
        print()

    # Compute speedup if both backends tested
    backends = df['backend'].unique()
    if len(backends) == 2:
        qiskit_data = df[df['backend'] == 'Qiskit'].sort_values('n_samples')
        qulacs_data = df[df['backend'] == 'Qulacs'].sort_values('n_samples')

        if len(qiskit_data) == len(qulacs_data):
            speedup_collection = qiskit_data['collection_time_mean'].values / qulacs_data['collection_time_mean'].values
            speedup_overlap = qiskit_data['time_per_overlap'].values / qulacs_data['time_per_overlap'].values

            print(f"--- Qulacs vs Qiskit Speedup ---")
            print(f"Sample collection speedup: {speedup_collection.mean():.2f}x (±{speedup_collection.std():.2f})")
            print(f"Overlap estimation speedup: {speedup_overlap.mean():.2f}x (±{speedup_overlap.std():.2f})")
            print()


def main():
    parser = argparse.ArgumentParser(description="Benchmark shadow protocol scaling")
    parser.add_argument(
        '--system',
        type=str,
        default='H2',
        choices=['H2', 'BeH2', 'H2O', 'LiH'],
        help='Molecular system to test'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        nargs='+',
        default=[10, 20, 50, 100, 200, 500, 1000],
        help='List of N values to test'
    )
    parser.add_argument(
        '--k-estimators',
        type=int,
        default=2,
        help='Number of k-estimators for median-of-means'
    )
    parser.add_argument(
        '--repeats',
        type=int,
        default=3,
        help='Number of timing repetitions per N'
    )
    parser.add_argument(
        '--no-qiskit',
        action='store_true',
        help='Skip Qiskit backend benchmarks'
    )
    parser.add_argument(
        '--no-qulacs',
        action='store_true',
        help='Skip Qulacs backend benchmarks'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output CSV file path for results'
    )
    parser.add_argument(
        '--plot',
        type=str,
        default=None,
        help='Output path for plot (e.g., scaling.png)'
    )

    args = parser.parse_args()

    # Run benchmark
    df = run_scaling_benchmark(
        system=args.system,
        n_samples_list=args.n_samples,
        n_estimators=args.k_estimators,
        n_repeats=args.repeats,
        test_qiskit=not args.no_qiskit,
        test_qulacs=not args.no_qulacs,
    )

    # Print summary
    print_summary(df)

    # Save results
    if args.output:
        df.to_csv(args.output, index=False)
        print(f"Results saved to: {args.output}")

    # Generate plots
    if args.plot or not args.output:
        plot_results(df, args.plot)

    # Print table
    print("\n=== DETAILED RESULTS ===")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
