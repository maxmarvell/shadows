"""Quick analysis script for scaling benchmark results."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

def analyze_results(csv_path: str):
    """Analyze benchmark results from CSV file."""
    df = pd.read_csv(csv_path)

    print(f"\n{'='*60}")
    print(f"Analysis of: {csv_path}")
    print(f"{'='*60}\n")

    # Basic info
    print(f"System: {df['system'].iloc[0]}")
    print(f"Qubits: {df['n_qubits'].iloc[0]}")
    print(f"Backends tested: {', '.join(df['backend'].unique())}")
    print(f"N values: {sorted(df['n_samples'].unique())}")
    print()

    # Per-backend analysis
    for backend in df['backend'].unique():
        data = df[df['backend'] == backend]
        print(f"\n{'-'*40}")
        print(f"{backend} Backend")
        print(f"{'-'*40}")

        # Time per sample statistics
        print(f"Time per sample:")
        print(f"  Mean: {data['time_per_sample'].mean()*1000:.2f} ms")
        print(f"  Std:  {data['time_per_sample'].std()*1000:.2f} ms")
        print(f"  Min:  {data['time_per_sample'].min()*1000:.2f} ms")
        print(f"  Max:  {data['time_per_sample'].max()*1000:.2f} ms")

        # Linear fit for scaling
        coeffs = np.polyfit(data['n_samples'], data['collection_time_mean'], 1)
        print(f"\nLinear scaling: T(N) ≈ {coeffs[0]:.6f} × N + {coeffs[1]:.6f}")
        print(f"Time per overlap: {data['time_per_overlap'].mean()*1000:.2f} ms (avg)")

        # Total time for common N values
        print(f"\nTotal collection times:")
        for _, row in data.iterrows():
            print(f"  N={row['n_samples']:4d}: {row['collection_time_mean']:.3f} s "
                  f"(±{row['collection_time_std']:.3f} s)")

    # Speedup analysis
    backends = df['backend'].unique()
    if len(backends) == 2:
        print(f"\n{'-'*40}")
        print("Qulacs vs Qiskit Speedup")
        print(f"{'-'*40}")

        qiskit = df[df['backend'] == 'Qiskit'].sort_values('n_samples')
        qulacs = df[df['backend'] == 'Qulacs'].sort_values('n_samples')

        if len(qiskit) == len(qulacs):
            speedup_collection = qiskit['collection_time_mean'].values / qulacs['collection_time_mean'].values
            speedup_per_sample = qiskit['time_per_sample'].values / qulacs['time_per_sample'].values

            print(f"Collection speedup: {speedup_collection.mean():.2f}x (±{speedup_collection.std():.2f})")
            print(f"Per-sample speedup: {speedup_per_sample.mean():.2f}x (±{speedup_per_sample.std():.2f})")

            print("\nSpeedup by N:")
            for i, n in enumerate(qiskit['n_samples']):
                print(f"  N={n:4d}: {speedup_collection[i]:.2f}x")

    # Recommendations
    print(f"\n{'-'*40}")
    print("Recommendations")
    print(f"{'-'*40}")

    if 'Qulacs' in df['backend'].values:
        qulacs_data = df[df['backend'] == 'Qulacs']
        avg_time_per_sample = qulacs_data['time_per_sample'].mean()

        if avg_time_per_sample < 0.0005:  # < 0.5 ms
            print(f"✓ Qulacs provides excellent performance")
            print(f"  Recommended for N > 100")
        elif avg_time_per_sample < 0.001:  # < 1 ms
            print(f"✓ Qulacs provides good performance")
            print(f"  Recommended for N > 50")
        else:
            print(f"⚠ Performance is moderate")
            print(f"  Consider using more parallel workers for larger systems")

    if 'Qiskit' in df['backend'].values:
        qiskit_data = df[df['backend'] == 'Qiskit']
        avg_time_per_sample = qiskit_data['time_per_sample'].mean()

        if avg_time_per_sample > 0.002:  # > 2 ms
            print(f"⚠ Qiskit backend is slow for this system")
            print(f"  Strongly recommend installing Qulacs")

    print()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_results.py <results.csv>")
        print("\nExample:")
        print("  python analyze_results.py benchmarks/scaling_results.csv")
        sys.exit(1)

    analyze_results(sys.argv[1])
