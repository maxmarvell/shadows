"""Quick analysis tool for collect_samples benchmark results.

Usage:
    python benchmarks/analyze_collect_samples.py benchmarks/collect_samples_results.csv
"""

import sys
import pandas as pd
import numpy as np


def main():
    if len(sys.argv) < 2:
        print("Usage: python analyze_collect_samples.py <results.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    df = pd.read_csv(csv_path)

    print("=" * 80)
    print(f"ANALYSIS: {csv_path}")
    print("=" * 80)

    # Optimal configuration for each system size
    print("\nOPTIMAL CONFIGURATIONS:")
    print("-" * 80)

    for n_h in sorted(df['n_hydrogen'].unique()):
        subset = df[df['n_hydrogen'] == n_h]
        best_idx = subset['mean_time_s'].idxmin()
        best = subset.loc[best_idx]

        n_qubits = int(best['n_qubits'])
        backend = best['backend']
        n_jobs = int(best['n_jobs'])
        n_samples = int(best['n_samples'])
        mean_time = best['mean_time_s']
        throughput = best['samples_per_sec']

        print(f"\nH{n_h} chain ({n_qubits} qubits):")
        print(f"  Best config: {backend}, {n_jobs} threads, {n_samples} samples")
        print(f"  Time: {mean_time:.3f} s")
        print(f"  Throughput: {throughput:.0f} samples/s")

        # Estimate for different sample counts
        ms_per_sample = best['ms_per_sample']
        for target_samples in [1000, 5000, 10000]:
            est_time = (target_samples * ms_per_sample) / 1000
            print(f"  Est. time for {target_samples:,} samples: {est_time:.1f} s ({est_time/60:.1f} min)")

    # Speedup analysis
    print("\n\nPARALLELIZATION SPEEDUP:")
    print("-" * 80)

    for backend in df['backend'].unique():
        print(f"\n{backend}:")
        backend_df = df[df['backend'] == backend]

        for n_h in sorted(backend_df['n_hydrogen'].unique()):
            subset = backend_df[backend_df['n_hydrogen'] == n_h]

            # Get serial baseline
            serial = subset[subset['n_jobs'] == 1]
            if len(serial) == 0:
                continue

            serial_time = serial['mean_time_s'].mean()

            print(f"  H{n_h}:")
            for n_jobs in sorted(subset['n_jobs'].unique()):
                if n_jobs == 1:
                    print(f"    {n_jobs} thread:  baseline")
                else:
                    parallel = subset[subset['n_jobs'] == n_jobs]
                    if len(parallel) > 0:
                        parallel_time = parallel['mean_time_s'].mean()
                        speedup = serial_time / parallel_time
                        efficiency = (speedup / n_jobs) * 100
                        print(f"    {n_jobs} threads: {speedup:.2f}x speedup ({efficiency:.0f}% efficiency)")

    # System size scaling
    print("\n\nSYSTEM SIZE SCALING:")
    print("-" * 80)

    # Use serial, 100 samples as baseline
    baseline = df[(df['n_jobs'] == 1) & (df['n_samples'] == 100)]
    if len(baseline) > 0:
        print("\nTime per sample scaling (serial, 100 samples):")
        for n_h in sorted(baseline['n_hydrogen'].unique()):
            subset = baseline[baseline['n_hydrogen'] == n_h]
            n_qubits = int(subset['n_qubits'].iloc[0])
            ms_per_sample = subset['ms_per_sample'].mean()
            print(f"  H{n_h} ({n_qubits:2d} qubits): {ms_per_sample:.2f} ms/sample")

    print("\n" + "=" * 80)


if __name__ == '__main__':
    main()
