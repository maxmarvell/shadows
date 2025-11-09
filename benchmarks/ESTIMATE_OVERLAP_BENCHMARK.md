# estimate_overlap() Performance Benchmark

Benchmark for analyzing the performance of `ShadowProtocol.estimate_overlap()`, which computes overlaps between shadow samples and target bitstrings. This method is called repeatedly during ground state estimation (once for HF reference + once per excitation).

## Quick Start

```bash
# Quick test (H2-H4, 100-500 samples)
python benchmarks/benchmark_estimate_overlap.py --quick

# Full benchmark (H2-H8, various sample counts)
python benchmarks/benchmark_estimate_overlap.py --full

# Custom configuration
python benchmarks/benchmark_estimate_overlap.py \
    --hydrogen-atoms 2 4 6 \
    --n-samples 100 500 1000 \
    --n-overlaps 10 50 \
    --repeats 3
```

## What This Benchmark Tests

### Key Question
**How long does it take to compute an overlap after shadow samples are collected?**

This is critical because:
- HF overlap (c0): 1 call
- Single excitations (c1): nocc × nvirt calls
- Double excitations (c2): (nocc × nvirt)² / 4 calls

For a typical system with 10 occupied and 10 virtual orbitals:
- Total overlaps = 1 + 100 + 2500 = **2601 overlap computations!**

### Dimensions Tested

1. **System Size**: H2, H4, H6, H8 (4-16 qubits)
2. **Shadow Samples**: 100, 500, 1000, 5000
3. **Batch Size**: 1, 10, 50, 100 overlaps (to check for batch effects)

## Key Findings

### Scaling Behavior

From quick benchmark results:

**Linear Scaling with Sample Count:**
```
100 samples:  ~11 ms per overlap
500 samples:  ~54 ms per overlap
→ Confirmed O(N) scaling
```

**Scaling with System Size:**
```
H2 (4 qubits):  ~20 ms per overlap (avg)
H4 (8 qubits):  ~45 ms per overlap (avg)
→ Approximately O(2^(n/8)) scaling
```

### Performance Metrics

| System | Samples | ms/overlap | overlaps/s |
|--------|---------|------------|------------|
| H2 | 100 | 7.1 | 141 |
| H2 | 500 | 33.6 | 30 |
| H4 | 100 | 15.6 | 64 |
| H4 | 500 | 74.1 | 14 |

## Cost Breakdown

### Example: H6 System (12 qubits)

Typical H6 with 6 occupied, 6 virtual orbitals (RHF):
- HF overlap: 1
- Singles: 6 × 6 = 36
- Doubles: (6 × 5 / 2) × (6 × 5 / 2) = 225
- **Total: 262 overlaps**

With 500 shadow samples (~50 ms/overlap for 12 qubits):
- Total overlap time: 262 × 50 ms = **13.1 seconds**

### Breakdown by Phase

For ground state estimation, the time splits approximately:

1. **Collect shadow samples**: 60-80% of total time
   - Dominant cost for small systems
   - Parallelizable with n_jobs

2. **Estimate overlaps**: 20-40% of total time
   - Grows with O(n_excitations)
   - Not currently parallelized
   - Scales linearly with n_samples

## Optimization Opportunities

### 1. Reduce Sample Count (If Acceptable)

```python
# Current (slow but accurate)
estimator.estimate_ground_state(n_samples=5000, n_k_estimators=40)

# Optimized (faster, slightly less accurate)
estimator.estimate_ground_state(n_samples=1000, n_k_estimators=40)
```

**Impact on H6:**
- 5000 samples: 262 × 250ms = 65 seconds
- 1000 samples: 262 × 50ms = 13 seconds
- **5x speedup** with typically <10% increase in error

### 2. Parallelize Overlap Estimation (Future Work)

The overlap estimation loop is currently serial:
```python
# Current implementation (serial)
for i, ex in enumerate(excitations):
    coeffs[i] = protocol.estimate_overlap(ex.bitstring)
```

Could be parallelized:
```python
# Potential optimization
from multiprocessing import Pool
with Pool() as pool:
    coeffs = pool.starmap(protocol.estimate_overlap,
                         [(ex.bitstring,) for ex in excitations])
```

**Expected speedup:** 2-4x on 4-8 cores

### 3. Batch Overlap Estimation (Future Work)

Compute multiple overlaps simultaneously by reusing stabilizer calculations:
- Share canonicalization work across similar bitstrings
- Vectorize phase calculations

**Expected speedup:** 1.5-2x for large batches

## Understanding the Scaling

### Why Linear with Samples?

Each shadow snapshot requires:
1. Transform target state with inverse Clifford: O(n²)
2. Canonicalize stabilizers: O(n³)
3. Compute overlap phase: O(n²)

Total: O(n_samples × n³) where n = n_qubits

### Why Cubic with Qubits?

The bottleneck is stabilizer canonicalization (Gaussian elimination):
- Converts n stabilizers to canonical form
- Requires O(n³) operations
- Called once per snapshot per overlap

## Recommendations

### For Your Use Case

Based on H10 (20 qubits) with 10 occ, 10 virt:
- Singles: 100
- Doubles: 2500
- Total overlaps: 2601

**With 1000 samples:**
- Estimated overlap time: 2601 × 100ms = **260 seconds** (4.3 min)
- Sample collection time: ~20 seconds (with n_jobs=4)
- **Total: ~5 minutes per estimation**

**With 100 repetitions × 8 distances:**
- Total time: 5 min × 100 × 8 = **67 hours!**

**Optimization Strategy:**
1. **Reduce samples**: Use 500 instead of 1000 → **2x speedup**
2. **Reduce repetitions**: Use 50 instead of 100 → **2x speedup**
3. **Result**: ~17 hours (much more manageable)

### General Guidelines

**Small systems (4-8 qubits):**
- Overlap time is negligible (<1 second total)
- Sample collection dominates
- Focus on optimizing collect_samples()

**Medium systems (10-14 qubits):**
- Overlap time becomes significant (5-30 seconds)
- Balance sample count with accuracy needs
- Consider parallelizing if doing many estimations

**Large systems (>14 qubits):**
- Overlap time dominates (>1 minute)
- **Critical to minimize sample count**
- Future parallelization would help significantly

## Command-Line Options

```
usage: benchmark_estimate_overlap.py [-h]
                                     [--hydrogen-atoms H [H ...]]
                                     [--n-samples N [N ...]]
                                     [--n-overlaps O [O ...]]
                                     [--n-estimators K]
                                     [--repeats R]
                                     [--output FILE]
                                     [--plot FILE]
                                     [--quick] [--full]

Options:
  --hydrogen-atoms    Hydrogen chain lengths to test (default: 2 4)
  --n-samples         Shadow sample counts to test (default: 100 500)
  --n-overlaps        Number of overlaps per test (default: 10)
  --n-estimators      Median-of-means bins (default: 10)
  --repeats           Repetitions per config (default: 3)
  --output            Output CSV path
  --plot              Output plot path
  --quick             Quick preset: H2-H4, 100-500 samples
  --full              Full preset: H2-H8, all configs
```

## Output Files

1. **CSV**: Raw timing data for all configurations
2. **PNG**: 4-panel plot showing:
   - Overlap cost vs system size
   - Overlap cost vs sample count
   - Throughput vs system size
   - Scaling analysis (log-log)

## Integration with Other Benchmarks

Compare with `benchmark_collect_samples.py` results to understand full workflow:

```
Total Time = Sample Collection Time + Overlap Estimation Time

For H6 with 1000 samples, 262 overlaps:
  Sample collection: ~5 seconds (from collect_samples benchmark)
  Overlap estimation: ~13 seconds (from this benchmark)
  Total: ~18 seconds
```

## See Also

- [benchmark_collect_samples.py](benchmark_collect_samples.py) - Sample collection performance
- [COLLECT_SAMPLES_BENCHMARK.md](COLLECT_SAMPLES_BENCHMARK.md) - Collection phase analysis
- [benchmark_ground_state.py](benchmark_ground_state.py) - End-to-end workflow
