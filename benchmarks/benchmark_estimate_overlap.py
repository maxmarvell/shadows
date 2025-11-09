"""Benchmark for ShadowProtocol.estimate_overlap() performance.

This benchmark specifically measures the cost of computing overlaps after shadow
samples have been collected. This is important because overlap estimation is called
many times (once for HF reference + once per excitation).

Tested dimensions:
1. System size: H2, H4, H6, H8 chains (4 to 16 qubits)
2. Shadow samples: 100, 500, 1000, 5000
3. Number of overlap calls: 1, 10, 100

Usage:
    # Quick test
    python benchmarks/benchmark_estimate_overlap.py --quick

    # Full benchmark
    python benchmarks/benchmark_estimate_overlap.py --full

    # Custom
    python benchmarks/benchmark_estimate_overlap.py \
        --hydrogen-atoms 2 4 6 \
        --n-samples 500 1000 \
        --n-overlaps 10 50 \
        --repeats 3
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
from shadow_ci.utils import make_hydrogen_chain, Bitstring


def setup_system_and_protocol(n_hydrogen: int, n_samples: int, n_estimators: int = 10):
    """Setup hydrogen system and collect shadow samples.

    Args:
        n_hydrogen: Number of hydrogen atoms
        n_samples: Number of shadow samples to collect
        n_estimators: Number of median-of-means estimators

    Returns:
        Tuple of (protocol, n_qubits, hamiltonian)
    """
    # Build molecule
    mol_string = make_hydrogen_chain(n_hydrogen, 0.74)
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

    # Create protocol and collect samples
    protocol = ShadowProtocol(trial_state, ensemble_type='clifford', use_qulacs=True, n_jobs=1)
    protocol.collect_samples(n_samples, n_estimators, prediction='overlap')

    return protocol, n_qubits, hamiltonian


def benchmark_single_overlap(protocol: ShadowProtocol, bitstring: Bitstring) -> float:
    """Benchmark a single overlap estimation.

    Args:
        protocol: ShadowProtocol with collected samples
        bitstring: Target bitstring for overlap

    Returns:
        Elapsed time in seconds
    """
    start_time = time.perf_counter()
    _ = protocol.estimate_overlap(bitstring)
    elapsed_time = time.perf_counter() - start_time
    return elapsed_time


def benchmark_multiple_overlaps(
    protocol: ShadowProtocol,
    bitstrings: list,
    repeats: int = 3
) -> tuple:
    """Benchmark multiple overlap estimations.

    Args:
        protocol: ShadowProtocol with collected samples
        bitstrings: List of target bitstrings
        repeats: Number of times to repeat the benchmark

    Returns:
        Tuple of (mean_time_total, mean_time_per_overlap, std_time_per_overlap)
    """
    times_total = []
    times_per_overlap = []

    for _ in range(repeats):
        start_time = time.perf_counter()
        for bs in bitstrings:
            _ = protocol.estimate_overlap(bs)
        elapsed = time.perf_counter() - start_time

        times_total.append(elapsed)
        times_per_overlap.append(elapsed / len(bitstrings))

    return np.mean(times_total), np.mean(times_per_overlap), np.std(times_per_overlap)


def run_benchmark_suite(
    hydrogen_atoms_list,
    n_samples_list,
    n_overlaps_list,
    n_estimators=10,
    repeats=3
):
    """Run comprehensive overlap estimation benchmark.

    Args:
        hydrogen_atoms_list: List of hydrogen chain lengths to test
        n_samples_list: List of shadow sample counts to test
        n_overlaps_list: List of overlap counts to test
        n_estimators: Number of median-of-means estimators (fixed)
        repeats: Number of times to repeat each configuration

    Returns:
        pandas DataFrame with benchmark results
    """
    results = []
    total_configs = (
        len(hydrogen_atoms_list) *
        len(n_samples_list) *
        len(n_overlaps_list)
    )

    config_idx = 0

    print("=" * 80)
    print("ShadowProtocol.estimate_overlap() Benchmark Suite")
    print("=" * 80)
    print(f"Total configurations to test: {total_configs}")
    print(f"Repeats per configuration: {repeats}")
    print("=" * 80)

    for n_hydrogen in hydrogen_atoms_list:
        print(f"\n{'=' * 80}")
        print(f"Testing H{n_hydrogen} chain...")
        print(f"{'=' * 80}")

        for n_samples in n_samples_list:
            # Check divisibility
            if n_samples % n_estimators != 0:
                print(f"  ⚠ Skipping n_samples={n_samples} (not divisible by {n_estimators})")
                continue

            print(f"\n  Setting up system with {n_samples} shadow samples...")
            t_setup_start = time.perf_counter()

            try:
                protocol, n_qubits, hamiltonian = setup_system_and_protocol(
                    n_hydrogen, n_samples, n_estimators
                )
                t_setup = time.perf_counter() - t_setup_start
                print(f"    ✓ Setup complete in {t_setup:.2f} s ({n_qubits} qubits)")

                # Generate random bitstrings for overlap estimation
                # Use HF reference + single excitations + random states
                bitstrings = []

                # Add HF reference
                bitstrings.append(hamiltonian.get_hf_bitstring())

                # Add some single excitations
                singles = hamiltonian.get_single_excitations()
                for ex in singles[:min(5, len(singles))]:
                    bitstrings.append(ex.bitstring)

                # Add random bitstrings
                rng = np.random.default_rng(42)
                for _ in range(10):
                    bitstrings.append(Bitstring.random(n_qubits, rng=rng))

            except Exception as e:
                print(f"    ✗ Failed to setup: {e}")
                continue

            for n_overlaps in n_overlaps_list:
                config_idx += 1
                progress = (config_idx / total_configs) * 100

                # Select subset of bitstrings
                test_bitstrings = bitstrings[:min(n_overlaps, len(bitstrings))]
                actual_n_overlaps = len(test_bitstrings)

                print(f"\n  [{config_idx}/{total_configs}] ({progress:.1f}%) "
                      f"n_samples={n_samples}, n_overlaps={actual_n_overlaps}")

                try:
                    mean_total, mean_per, std_per = benchmark_multiple_overlaps(
                        protocol, test_bitstrings, repeats=repeats
                    )

                    print(f"    Total time: {mean_total:.4f} s")
                    print(f"    Per overlap: {mean_per*1000:.2f} ± {std_per*1000:.2f} ms")
                    print(f"    Throughput: {1/mean_per:.1f} overlaps/s")

                    results.append({
                        'n_hydrogen': n_hydrogen,
                        'n_qubits': n_qubits,
                        'n_samples': n_samples,
                        'n_overlaps': actual_n_overlaps,
                        'mean_time_total_s': mean_total,
                        'mean_time_per_overlap_s': mean_per,
                        'std_time_per_overlap_s': std_per,
                        'ms_per_overlap': mean_per * 1000,
                        'overlaps_per_sec': 1 / mean_per,
                        'repeats': repeats
                    })

                except Exception as e:
                    print(f"    ✗ Benchmark failed: {e}")

    return pd.DataFrame(results)


def analyze_results(df: pd.DataFrame):
    """Print analysis summary of benchmark results."""
    print("\n" + "=" * 80)
    print("BENCHMARK ANALYSIS SUMMARY")
    print("=" * 80)

    # Scaling with sample count
    print("\n1. Scaling with Number of Shadow Samples")
    print("-" * 80)
    print("\nTime per overlap vs n_samples (averaged across system sizes):")
    scaling_samples = df.groupby('n_samples').agg({
        'ms_per_overlap': ['mean', 'std'],
        'overlaps_per_sec': 'mean'
    })
    print(scaling_samples)

    # Scaling with system size
    print("\n2. Scaling with System Size")
    print("-" * 80)
    print("\nTime per overlap vs n_qubits (averaged across sample counts):")
    scaling_size = df.groupby(['n_hydrogen', 'n_qubits']).agg({
        'ms_per_overlap': ['mean', 'std'],
        'overlaps_per_sec': 'mean'
    })
    print(scaling_size)

    # Batch size effects
    print("\n3. Batch Size Effects (if applicable)")
    print("-" * 80)
    # Check if there's variation in n_overlaps
    if df['n_overlaps'].nunique() > 1:
        print("\nTime per overlap vs n_overlaps:")
        batch_effects = df.groupby('n_overlaps').agg({
            'ms_per_overlap': ['mean', 'std']
        })
        print(batch_effects)
    else:
        print("Only one overlap count tested, no batch effects to analyze.")

    # Overall statistics
    print("\n4. Overall Statistics")
    print("-" * 80)
    print(f"Fastest overlap: {df['ms_per_overlap'].min():.2f} ms")
    print(f"Slowest overlap: {df['ms_per_overlap'].max():.2f} ms")
    print(f"Median overlap: {df['ms_per_overlap'].median():.2f} ms")
    print(f"Mean overlap: {df['ms_per_overlap'].mean():.2f} ms")

    # Cost estimate for typical ground state estimation
    print("\n5. Cost Estimates for Ground State Estimation")
    print("-" * 80)

    for n_h in sorted(df['n_hydrogen'].unique()):
        subset = df[df['n_hydrogen'] == n_h]
        if len(subset) == 0:
            continue

        # Use median time as estimate
        ms_per_overlap_median = subset['ms_per_overlap'].median()
        n_qubits = int(subset['n_qubits'].iloc[0])

        # Estimate number of excitations for this system
        # For RHF: nocc = nH, nvirt = (n_qubits/2 - nocc)
        nocc = n_h
        nvirt = n_qubits // 2 - nocc
        n_singles = nocc * nvirt  # per spin, RHF only measures alpha
        n_doubles = (nocc * (nocc - 1) // 2) * (nvirt * (nvirt - 1) // 2)  # alpha-beta only for RHF
        total_overlaps = 1 + n_singles + n_doubles  # HF + singles + doubles

        total_time = total_overlaps * ms_per_overlap_median / 1000

        print(f"\nH{n_h} ({n_qubits} qubits):")
        print(f"  Overlaps needed: {total_overlaps} (1 HF + {n_singles} singles + {n_doubles} doubles)")
        print(f"  Est. overlap time: {ms_per_overlap_median:.2f} ms each")
        print(f"  Total overlap est. time: {total_time:.2f} s ({total_time/60:.2f} min)")


def plot_results(df: pd.DataFrame, output_path: str):
    """Create plots of benchmark results."""
    import matplotlib.pyplot as plt
    import sys
    from pathlib import Path

    # Try to import plotting config
    scripts_path = Path(__file__).parent.parent / 'scripts'
    if scripts_path.exists():
        sys.path.insert(0, str(scripts_path))
        try:
            from plotting_config import setup_plotting_style
            setup_plotting_style()
        except ImportError:
            pass

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Plot 1: Time per overlap vs system size
    ax = axes[0, 0]
    for n_samples in sorted(df['n_samples'].unique()):
        subset = df[df['n_samples'] == n_samples]
        grouped = subset.groupby('n_qubits')['ms_per_overlap'].mean()
        ax.plot(grouped.index, grouped.values, 'o-',
                label=f'{n_samples} samples', linewidth=2, markersize=6)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Time per Overlap (ms)')
    ax.set_title('Overlap Cost vs System Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Time per overlap vs number of samples
    ax = axes[0, 1]
    for n_h in sorted(df['n_hydrogen'].unique()):
        subset = df[df['n_hydrogen'] == n_h]
        grouped = subset.groupby('n_samples')['ms_per_overlap'].mean()
        ax.semilogx(grouped.index, grouped.values, 'o-',
                   label=f'H{n_h}', linewidth=2, markersize=6)
    ax.set_xlabel('Number of Shadow Samples')
    ax.set_ylabel('Time per Overlap (ms)')
    ax.set_title('Overlap Cost vs Sample Count')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Throughput vs system size
    ax = axes[1, 0]
    for n_samples in sorted(df['n_samples'].unique()):
        subset = df[df['n_samples'] == n_samples]
        grouped = subset.groupby('n_qubits')['overlaps_per_sec'].mean()
        ax.plot(grouped.index, grouped.values, 'o-',
                label=f'{n_samples} samples', linewidth=2, markersize=6)
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Throughput (overlaps/s)')
    ax.set_title('Overlap Throughput vs System Size')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Scaling summary (log-log)
    ax = axes[1, 1]
    # Plot time vs n_samples for each system size
    for n_h in sorted(df['n_hydrogen'].unique()):
        subset = df[df['n_hydrogen'] == n_h]
        grouped = subset.groupby('n_samples')['ms_per_overlap'].mean()
        ax.loglog(grouped.index, grouped.values, 'o-',
                 label=f'H{n_h}', linewidth=2, markersize=6)

    # Add reference lines
    x_ref = np.array([100, 5000])
    # Linear scaling reference
    y_ref = df['ms_per_overlap'].median() * x_ref / 1000
    ax.loglog(x_ref, y_ref, '--', color='gray', alpha=0.5,
             label='Linear scaling', linewidth=1)

    ax.set_xlabel('Number of Shadow Samples')
    ax.set_ylabel('Time per Overlap (ms) [log scale]')
    ax.set_title('Scaling Analysis (Log-Log)')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Plot saved to: {output_path}")


def main():
    """Main benchmark execution."""
    parser = argparse.ArgumentParser(
        description='Benchmark ShadowProtocol.estimate_overlap() performance'
    )

    parser.add_argument('--hydrogen-atoms', nargs='+', type=int,
                        default=[2, 4],
                        help='List of hydrogen chain lengths (default: 2 4)')
    parser.add_argument('--n-samples', nargs='+', type=int,
                        default=[100, 500],
                        help='List of shadow sample counts (default: 100 500)')
    parser.add_argument('--n-overlaps', nargs='+', type=int,
                        default=[10],
                        help='List of overlap counts to test (default: 10)')
    parser.add_argument('--n-estimators', type=int, default=10,
                        help='Number of median-of-means bins (default: 10)')
    parser.add_argument('--repeats', type=int, default=3,
                        help='Number of repeats per config (default: 3)')
    parser.add_argument('--output', type=str, default='estimate_overlap_benchmark.csv',
                        help='Output CSV path')
    parser.add_argument('--plot', type=str, default='estimate_overlap_benchmark.png',
                        help='Output plot path')
    parser.add_argument('--quick', action='store_true',
                        help='Quick test: H2-H4, 100-500 samples')
    parser.add_argument('--full', action='store_true',
                        help='Full test: H2-H8, various sample counts')

    args = parser.parse_args()

    # Apply presets
    if args.quick:
        args.hydrogen_atoms = [2, 4]
        args.n_samples = [100, 500]
        args.n_overlaps = [10]
        args.repeats = 2
    elif args.full:
        args.hydrogen_atoms = [2, 4, 6, 8]
        args.n_samples = [100, 500, 1000, 5000]
        args.n_overlaps = [1, 10, 50]
        args.repeats = 3

    # Run benchmark
    df = run_benchmark_suite(
        hydrogen_atoms_list=args.hydrogen_atoms,
        n_samples_list=args.n_samples,
        n_overlaps_list=args.n_overlaps,
        n_estimators=args.n_estimators,
        repeats=args.repeats
    )

    # Save results
    df.to_csv(args.output, index=False)
    print(f"\n{'=' * 80}")
    print(f"Results saved to: {args.output}")
    print(f"{'=' * 80}")

    # Analyze
    analyze_results(df)

    # Plot
    plot_results(df, args.plot)

    print("\n" + "=" * 80)
    print("BENCHMARK COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
