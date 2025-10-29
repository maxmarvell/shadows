# Methodology Comparison: Your Implementation vs Reference

## Reference Implementation (excitation_amplitude_sampling)

### Key Formula (line 1355):
```python
overlap_contribution = 2 * magnitude² * (2^n + 1) * phase₁ * conj(phase₂)
```

Where:
- `magnitude = 1 / sqrt(2)^x_rank`
- `phase₁ = get_ovlp_phase_basis(meas_out, ovlp_state1, reduced_stab_matrix)`
- `phase₂ = get_ovlp_phase_basis(meas_out, ovlp_state2, reduced_stab_matrix)`

### Process Flow:
1. Measure tau state with random Clifford U → get measurement `b`
2. Create stabilizer from measurement: `|b⟩`
3. Apply U†: `stab_data.evolve(rand_clifford.adjoint())`
4. Reduce to canonical form
5. Compute x_rank from reduced matrix
6. **Measure the stabilizer state**: `meas_out = stab_data.measure()[0]`
7. Compute phase for overlap with target states

### Important: Line 1349
```python
meas_out = stab_data.measure()[0]
```
**This creates a NEW measurement from the evolved stabilizer state!**

## Your Implementation (shadow-ci)

### Key Formula (line 98):
```python
overlap_contribution = magnitude² * phase
```

Where:
- `magnitude = sqrt(2)^(-x_rank)`  ← **SAME**
- `phase = gaussian_elimination(stabilizers, bitstring, a) * conj(gaussian_elimination(stabilizers, bitstring, vacuum))`

### Process Flow (lines 73-98):
1. Get measurement `b` from snapshot
2. Create stabilizer from measurement: `tableau = snapshot.measurement.to_stabilizer()`
3. Apply U†: `tableau = U_inv * tableau`
4. Reduce to canonical form
5. Compute x_rank
6. **Re-measure the tableau**: `measurement = sim.measure_many(*range(snapshot.measurement.size))`
7. Create bitstring from this new measurement
8. Compute phase using gaussian_elimination

## CRITICAL DIFFERENCE IDENTIFIED!

### Line 90-91 in your code:
```python
measurement = sim.measure_many(*range(snapshot.measurement.size))
bitstring = Bitstring(measurement)
```

**You are RE-MEASURING the evolved stabilizer state**, just like the reference!

This is correct! But there's a potential issue...

## Issue: Line 96 - vacuum term

```python
vacuum = Bitstring([False] * self.n_qubits)
phase *= np.conj(gaussian_elimination(stabilizers, bitstring, vacuum))
```

### In reference code (line 1355):
For computing `⟨φ₁|ψ⟩`:
```python
phase₁ * conj(phase₂)
```

Where both phase₁ and phase₂ are computed from **THE SAME reference state** (meas_out).

### In your code:
You're computing:
```python
gaussian_elimination(stabilizers, bitstring, a) * conj(gaussian_elimination(stabilizers, bitstring, vacuum))
```

This computes `⟨bitstring|a⟩ * conj(⟨bitstring|vacuum⟩)` = `⟨a|bitstring⟩⟨bitstring|vacuum⟩`

## The Formula Should Be:

According to the paper (Eq S32 from Huggins et al.):

⟨φ|Ψ_T⟩ = 2(2^N + 1) E_k[⟨φ|U_k†|b_k⟩⟨b_k|U_k|0⟩]

For the tau protocol, estimating ⟨φ|ψ⟩:
- We sample from tau = (|0⟩ + |ψ⟩)/√2
- The estimator is: 2(2^N + 1) ⟨φ|U†|b⟩⟨b|U|0⟩

In stabilizer formalism:
- U†|b⟩ is a stabilizer state (from which we sample |b'⟩)
- ⟨φ|U†|b⟩ = ⟨φ|b'⟩ × magnitude
- ⟨b|U|0⟩ = ⟨b'|0⟩ × magnitude (since ⟨b|U = ⟨b'| up to phase)

## Potential Issues in Your Implementation:

### 1. The vacuum term (line 96)
You compute: `⟨a|b'⟩⟨b'|0⟩`

**Question**: Is this correct? Let's check:
- The reference computes: `⟨ovlp_state1|b'⟩ * conj(⟨ovlp_state2|b'⟩)`
- For computing ⟨a|ψ⟩, we need: ovlp_state1 = a, ovlp_state2 = 0

So your formula seems correct!

### 2. Normalization (line 100)
```python
return 2 * (2**self.n_qubits + 1) * np.mean(overlaps)
```

**Issue**: The reference code (line 1366) divides by `samples_per_estimator`:
```python
result_list.append(np.median(n_median)*(1/float(samples_per_estimator))*op_list[n][0])
```

**You are taking np.mean() which already divides by sample count!** But then you apply the normalization factor.

The reference accumulates the sum in line 1355 without dividing, then takes median of these sums, then divides by samples_per_estimator.

### 3. Imaginary Phase Accumulation

Looking at line 1355 in reference:
```python
phase₁ * np.conj(phase₂)
```

This product should be real if both states are computational basis states!

**In your code (line 92, 96):**
```python
phase = gaussian_elimination(stabilizers, bitstring, a)
phase *= np.conj(gaussian_elimination(stabilizers, bitstring, vacuum))
```

If `a` and `vacuum` are both real computational basis states, the product should be real.

**Source of imaginary phase**: The `gaussian_elimination` function might be returning complex phases that don't cancel when multiplied.

## Recommendations:

1. **Check gaussian_elimination output**: For computational basis states, it should return real phases (1, -1) or 0.

2. **Verify the normalization**: The factor `2(2^N + 1)` should be applied after averaging, and the averaging should be done correctly.

3. **Debug the phase**: Add logging to see what phases are being computed and whether they're real or complex.

4. **Match the reference exactly**: The reference accumulates raw overlap values, then takes median, then averages. You're averaging first, then taking median.
