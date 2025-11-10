# Shadow CI Benchmarks

This directory contains comprehensive benchmarks for all major components of the Shadow CI ground state estimation workflow.

## Quick Start

```bash
# Install benchmark dependencies (if not already installed)
pip install pytest-benchmark

# Run all benchmarks
pytest benchmarks/ --benchmark-only

# Run specific benchmark class
pytest benchmarks/benchmark_ground_state.py::TestCliffordOperations --benchmark-only

# Run with verbose output
pytest benchmarks/ --benchmark-only -v

# Run shadow scaling analysis
python benchmarks/benchmark_shadow_scaling.py
```

## Benchmark Organization

### `benchmark_ground_state.py`

Main pytest benchmark suite covering:

1. **TestHamiltonianConstruction**
   - PySCF Hamiltonian construction
   - Single and double excitation generation
   - Tests on H2, BeH2, and H2O molecules

2. **TestTrialStatePreparation**
   - Hartree-Fock state generation
   - Bitstring conversions

3. **TestShadowProtocol**
   - Protocol initialization
   - Sample collection (serial and parallel)
   - Overlap estimation (single and batch)

4. **TestAmplitudeTensorConstruction**
   - SingleAmplitudes tensor building
   - DoubleAmplitudes antisymmetrization
   - RHF symmetry expansion

5. **TestEnergyEvaluation**
   - Correlation energy computation
   - Fock matrix contractions

6. **TestEstimatorComponents**
   - End-to-end amplitude estimation
   - Component-level profiling

7. **TestCliffordOperations**
   - Random Clifford generation
   - Clifford inverse operations
   - Stabilizer canonicalization

### `benchmark_shadow_scaling.py`

Specialized benchmark for analyzing shadow protocol scaling with number of shots (N), comparing Qiskit vs Qulacs backends.

**Quick run:**
```bash
python benchmarks/benchmark_shadow_scaling.py
```

**Custom run:**
```bash
python benchmarks/benchmark_shadow_scaling.py \
    --system H2 \
    --n-samples 10 20 50 100 200 500 1000 \
    --k-estimators 2 \
    --repeats 3 \
    --output results.csv \
    --plot scaling.png
```

See [SCALING_ANALYSIS.md](SCALING_ANALYSIS.md) for detailed results and analysis.

### `analyze_results.py`

Quick analysis script for scaling benchmark CSV results:

```bash
python benchmarks/analyze_results.py benchmarks/scaling_results.csv
```

## Advanced Usage

### Save Results to File

```bash
# Save as JSON
pytest benchmarks/ --benchmark-only --benchmark-json=results.json

# Save as HTML (requires pytest-benchmark[histogram])
pytest benchmarks/ --benchmark-only --benchmark-histogram=histogram
```

### Compare with Previous Results

```bash
# Save baseline
pytest benchmarks/ --benchmark-only --benchmark-save=baseline

# Run comparison
pytest benchmarks/ --benchmark-only --benchmark-compare=baseline
```

### Sort and Filter Results

```bash
# Sort by mean time
pytest benchmarks/ --benchmark-only --benchmark-sort=mean

# Only show specific columns
pytest benchmarks/ --benchmark-only --benchmark-columns=min,max,mean,ops

# Disable outlier detection
pytest benchmarks/ --benchmark-only --benchmark-disable-gc
```

### Performance Profiling

For more detailed profiling, use `pytest-profiling`:

```bash
pip install pytest-profiling
pytest benchmarks/ --benchmark-only --profile
```

Or use `py-spy` for sampling profiler:

```bash
pip install py-spy
py-spy record -o profile.svg -- pytest benchmarks/benchmark_ground_state.py::TestShadowProtocol --benchmark-only
```

## Interpreting Results

Benchmark output includes:

- **Min/Max**: Minimum and maximum execution times
- **Mean**: Average execution time
- **StdDev**: Standard deviation (consistency measure)
- **Median**: Middle value (robust to outliers)
- **IQR**: Interquartile range
- **Outliers**: Number of outlier measurements
- **OPS**: Operations per second (1 / Mean)
- **Rounds**: Number of benchmark iterations

### What to Look For

- **High StdDev**: Indicates inconsistent performance (cache effects, GC, etc.)
- **Many Outliers**: May indicate system noise or JIT compilation
- **Low OPS**: Slow operations that need optimization
- **Scaling**: Compare times across different molecule sizes to identify scaling issues

## Benchmark Results

### Shadow Protocol Scaling (H2 Molecule)

**System**: H2, 4 qubits, STO-3G basis

**Key Findings:**
- **Qiskit**: 1.88 ms/sample (average)
- **Qulacs**: 0.31 ms/sample (average)
- **Speedup**: 5.95x ± 2.27x
- **Scaling**: Linear O(N) confirmed for both backends

| N | Qiskit Time | Qulacs Time | Speedup |
|---|-------------|-------------|---------|
| 10 | 0.041 s | 0.004 s | 10.2x |
| 100 | 0.168 s | 0.040 s | 4.2x |
| 1000 | 0.866 s | 0.197 s | 4.4x |

See [SCALING_ANALYSIS.md](SCALING_ANALYSIS.md) for detailed analysis and recommendations.

### Performance Recommendations

**For Small Molecules (< 6 qubits)**
- Qiskit alone is sufficient for prototyping
- Use N=100-500 for reasonable accuracy
- Expected time: 0.1-0.5 seconds with Qulacs

**For Medium Molecules (6-14 qubits)**
- **Strongly recommend Qulacs**
- Use N=500-2000 for accuracy
- Expected speedup: 5-10x over Qiskit

**For Large Molecules (> 14 qubits)**
- **Qulacs required** (Qiskit becomes impractical)
- Consider parallelization (`n_jobs > 1`)

**Installing Qulacs:**
```bash
pip install qulacs
```

See [BENCHMARK_RESULTS.md](../BENCHMARK_RESULTS.md) for additional analysis and optimization recommendations.

## Adding New Benchmarks

### Template

```python
class TestMyComponent:
    """Benchmark description."""

    @pytest.fixture
    def my_fixture(self):
        # Setup code
        return data

    def test_my_operation(self, benchmark, my_fixture):
        """Benchmark my operation."""
        benchmark(my_function, my_fixture)
```

### Best Practices

1. **Use fixtures for setup** - Don't include setup time in benchmarks
2. **Test realistic scenarios** - Use real molecule sizes and sample counts
3. **Include multiple scales** - Test small, medium, and large systems
4. **Document what you're measuring** - Clear docstrings
5. **Group related benchmarks** - Use test classes for organization

## Continuous Benchmarking

For CI/CD integration:

```bash
# Run benchmarks and fail if slower than baseline by 10%
pytest benchmarks/ --benchmark-only --benchmark-compare=baseline --benchmark-compare-fail=mean:10%

# Auto-save results with timestamp
pytest benchmarks/ --benchmark-only --benchmark-autosave
```

## Troubleshooting

### Benchmarks are too slow

- Run a subset: `pytest benchmarks/ --benchmark-only -k "test_single"`
- Reduce rounds: `pytest benchmarks/ --benchmark-min-rounds=1`
- Use faster systems: Test on H2 instead of H2O

### Inconsistent results

- Disable garbage collection: `--benchmark-disable-gc`
- Increase warm-up: `--benchmark-warmup=on`
- Close other applications to reduce system noise

### Out of memory

- Run tests separately by class
- Reduce sample sizes in fixtures
- Use module-scope fixtures to share setup
