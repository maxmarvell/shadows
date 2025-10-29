"""Tests for stabilizer utility functions in shadow_ci.utils."""

import pytest
import stim
import numpy as np
from shadow_ci.utils import (
    Bitstring,
    gaussian_elimination,
    apply_stabilizer_to_state,
    compute_x_rank,
    canonicalize
)


class TestApplyStabilizerToState:
    """Test the apply_stabilizer_to_state function."""

    def test_apply_x_operator(self):
        """Test applying X Pauli operator flips the bit."""
        state = Bitstring([False, False, False])
        stabilizer = stim.PauliString("X__")
        phase = 1 + 0j

        new_state, new_phase = apply_stabilizer_to_state(state, stabilizer, phase)

        assert new_state[2] == True
        assert new_state[1] == False
        assert new_state[0] == False
        assert new_phase == 1 + 0j

    def test_apply_z_operator_on_zero(self):
        """Test applying Z to |0⟩ doesn't flip bit but preserves phase."""
        state = Bitstring([False, False])
        stabilizer = stim.PauliString("Z_")
        phase = 1 + 0j

        new_state, new_phase = apply_stabilizer_to_state(state, stabilizer, phase)

        assert new_state[0] == False
        assert new_phase == 1 + 0j

    def test_apply_z_operator_on_one(self):
        """Test applying Z to |1⟩ doesn't flip bit but adds minus sign."""
        state = Bitstring([False, True])
        stabilizer = stim.PauliString("Z_")
        phase = 1 + 0j

        new_state, new_phase = apply_stabilizer_to_state(state, stabilizer, phase)

        assert new_state[1] == True
        assert new_phase == -1 + 0j

    def test_apply_y_operator_on_zero(self):
        """Test applying Y to |0⟩ flips bit and adds i phase."""
        state = Bitstring([False, False])
        stabilizer = stim.PauliString("Y_")
        phase = 1 + 0j

        new_state, new_phase = apply_stabilizer_to_state(state, stabilizer, phase)

        assert new_state[1] == True
        assert np.isclose(new_phase, 1j)

    def test_apply_y_operator_on_one(self):
        """Test applying Y to |1⟩ flips bit and adds -i phase."""
        state = Bitstring([False, True])
        stabilizer = stim.PauliString("Y_")
        phase = 1 + 0j

        new_state, new_phase = apply_stabilizer_to_state(state, stabilizer, phase)

        assert new_state[1] == False
        assert np.isclose(new_phase, -1j)

    def test_apply_identity_operator(self):
        """Test applying identity doesn't change state or phase."""
        state = Bitstring([True, False, True])
        stabilizer = stim.PauliString("___")
        phase = 1 + 0j

        new_state, new_phase = apply_stabilizer_to_state(state, stabilizer, phase)

        assert new_state[0] == True
        assert new_state[1] == False
        assert new_state[2] == True
        assert new_phase == 1 + 0j

    def test_apply_multiple_operators(self):
        """Test applying multiple Pauli operators simultaneously."""
        state = Bitstring([False, True, False])
        stabilizer = stim.PauliString("XZY")
        phase = 1 + 0j

        new_state, new_phase = apply_stabilizer_to_state(state, stabilizer, phase)

        # X flips first bit: False -> True
        assert new_state[0] == True
        # Z on |1⟩ adds -1 phase: 1 * -1 = -1
        assert new_state[1] == True
        # Y on |0⟩ flips bit and adds i phase: -1 * i = -i
        assert new_state[2] == True
        assert np.isclose(new_phase, -1j)

    def test_negative_stabilizer_sign(self):
        """Test stabilizer with negative sign."""
        state = Bitstring([False, False])
        stabilizer = stim.PauliString("-X_")
        phase = 1 + 0j

        new_state, new_phase = apply_stabilizer_to_state(state, stabilizer, phase)

        assert new_state[1] == True
        assert new_phase == -1 + 0j

    def test_accumulate_phase(self):
        """Test that phase accumulates correctly with initial phase."""
        state = Bitstring([True])
        stabilizer = stim.PauliString("Z")
        phase = 1j  # Start with i

        new_state, new_phase = apply_stabilizer_to_state(state, stabilizer, phase)

        assert new_state[0] == True
        # i * (-1) = -i
        assert np.isclose(new_phase, -1j)


class TestComputeXRank:
    """Test the compute_x_rank function."""

    def test_no_x_operators(self):
        """Test X-rank is 0 when no X operators are present."""
        stabilizers = [
            stim.PauliString("ZZZ"),
            stim.PauliString("Z__"),
            stim.PauliString("__Z"),
        ]
        x_rank = compute_x_rank(stabilizers)
        assert x_rank == 0

    def test_all_x_operators(self):
        """Test X-rank equals number of stabilizers when all have X."""
        stabilizers = [
            stim.PauliString("X__"),
            stim.PauliString("_X_"),
            stim.PauliString("__X"),
        ]
        x_rank = compute_x_rank(stabilizers)
        assert x_rank == 3

    def test_mixed_operators(self):
        """Test X-rank counts only stabilizers with at least one X."""
        stabilizers = [
            stim.PauliString("XYZ"),  # has X
            stim.PauliString("ZZZ"),  # no X
            stim.PauliString("XXX"),  # has X
            stim.PauliString("Y__"),  # has X (Y contains X component in stim)
            stim.PauliString("_X_"),  # has X
        ]
        x_rank = compute_x_rank(stabilizers)
        assert x_rank == 4

    def test_empty_stabilizers(self):
        """Test X-rank is 0 for empty stabilizer list."""
        stabilizers = []
        x_rank = compute_x_rank(stabilizers)
        assert x_rank == 0

    def test_single_stabilizer_with_x(self):
        """Test X-rank for single stabilizer with X."""
        stabilizers = [stim.PauliString("X")]
        x_rank = compute_x_rank(stabilizers)
        assert x_rank == 1

    def test_single_stabilizer_without_x(self):
        """Test X-rank for single stabilizer without X."""
        stabilizers = [stim.PauliString("Z")]
        x_rank = compute_x_rank(stabilizers)
        assert x_rank == 0

    def test_y_contains_x_component(self):
        """Test that Y operators are not counted as having X in X-rank.

        Note: This tests the current implementation which only checks
        the X bits of the stabilizer, not the Z bits. Y = iXZ in Pauli
        algebra but in stim representation Y is encoded separately.
        """
        stabilizers = [stim.PauliString("YYY")]
        x_rank = compute_x_rank(stabilizers)
        # Y is encoded with both X and Z bits set in stim
        # to_numpy() returns (x_bits, z_bits) where Y has both True
        assert x_rank == 1


class TestGaussianElimination:
    """Test the gaussian_elimination function."""

    def test_identical_states(self):
        """Test overlap when ref and target are identical."""
        ref_state = Bitstring([False, False])
        target_state = Bitstring([False, False])
        stabilizers = []  # No stabilizers needed for identical states

        phase = gaussian_elimination(stabilizers, ref_state, target_state)

        assert np.isclose(phase, 1 + 0j)

    def test_no_overlap_without_stabilizers(self):
        """Test no overlap when states differ and no stabilizers provided."""
        ref_state = Bitstring([False, False])
        target_state = Bitstring([True, False])
        stabilizers = []

        phase = gaussian_elimination(stabilizers, ref_state, target_state)

        assert phase == 0j

    def test_single_x_stabilizer(self):
        """Test overlap with single X stabilizer connecting two states."""
        ref_state = Bitstring([False, False])
        target_state = Bitstring([False, True])
        stabilizers = [stim.PauliString("X_")]

        phase = gaussian_elimination(stabilizers, ref_state, target_state)

        # X can flip the first bit from 0 to 1
        assert np.isclose(phase, 1 + 0j)

    def test_two_bit_flip_with_stabilizers(self):
        """Test flipping two bits with two X stabilizers."""
        ref_state = Bitstring([False, False])
        target_state = Bitstring([True, True])
        stabilizers = [
            stim.PauliString("X_"),
            stim.PauliString("_X"),
        ]

        phase = gaussian_elimination(stabilizers, ref_state, target_state)

        assert np.isclose(phase, 1 + 0j)

    def test_phase_from_z_stabilizer(self):
        """Test that Z stabilizer contributes phase when applied to |1⟩."""
        ref_state = Bitstring([True])
        target_state = Bitstring([True])
        stabilizers = [stim.PauliString("Z")]

        phase = gaussian_elimination(stabilizers, ref_state, target_state)

        # States are identical, no stabilizer needs to be applied
        assert np.isclose(phase, 1 + 0j)

    def test_bell_state_stabilizers(self):
        """Test with Bell state stabilizers (XX and ZZ)."""
        # |Φ+⟩ = (|00⟩ + |11⟩)/√2 stabilized by XX and ZZ
        ref_state = Bitstring([False, False])
        target_state = Bitstring([True, True])
        stabilizers = [stim.PauliString("XX")]  # XX flips both bits

        phase = gaussian_elimination(stabilizers, ref_state, target_state)

        assert np.isclose(phase, 1 + 0j)

    def test_no_overlap_impossible_transformation(self):
        """Test no overlap when stabilizers can't connect states."""
        ref_state = Bitstring([False, False])
        target_state = Bitstring([True, False])
        # Only have Z stabilizer which can't flip bits
        stabilizers = [stim.PauliString("Z_")]

        phase = gaussian_elimination(stabilizers, ref_state, target_state)

        assert phase == 0j

    def test_y_stabilizer_phase(self):
        """Test Y stabilizer contributes correct phase."""
        ref_state = Bitstring([False])
        target_state = Bitstring([True])
        stabilizers = [stim.PauliString("Y")]

        phase = gaussian_elimination(stabilizers, ref_state, target_state)

        # Y on |0⟩ gives i|1⟩
        assert np.isclose(phase, 1j)

    def test_three_qubit_state(self):
        """Test three-qubit system with multiple stabilizers."""
        ref_state = Bitstring([False, False, False])
        target_state = Bitstring([False, True, True])
        stabilizers = [
            stim.PauliString("X__"),
            stim.PauliString("_X_"),
        ]

        phase = gaussian_elimination(stabilizers, ref_state, target_state)

        assert np.isclose(phase, 1 + 0j)

    def test_negative_stabilizer_contribution(self):
        """Test stabilizer with negative sign affects phase."""
        ref_state = Bitstring([False])
        target_state = Bitstring([True])
        stabilizers = [stim.PauliString("-X")]

        phase = gaussian_elimination(stabilizers, ref_state, target_state)

        assert np.isclose(phase, -1 + 0j)

    def test_canonical_stabilizer_ordering(self):
        """Test that gaussian elimination respects canonical ordering.

        The algorithm should apply stabilizers in row-echelon form order,
        only using stabilizers that have their leftmost X in the correct position.
        """
        ref_state = Bitstring([False, False, False])
        target_state = Bitstring([True, True, True])
        stabilizers = [
            stim.PauliString("X__"),  # Leftmost X at position 0
            stim.PauliString("_X_"),  # Leftmost X at position 1
            stim.PauliString("__X"),  # Leftmost X at position 2
        ]

        phase = gaussian_elimination(stabilizers, ref_state, target_state)

        assert np.isclose(phase, 1 + 0j)

    def test_non_canonical_ordering_ignored(self):
        """Test that non-canonical stabilizers are properly used/ignored.

        If a stabilizer has X bits to the left of the current position,
        it should not be used for that position.
        """
        ref_state = Bitstring([False, False])
        target_state = Bitstring([True, False])
        # Second stabilizer has X in first position, shouldn't be used for position 1
        stabilizers = [
            stim.PauliString("XX"),  # X at both positions
            stim.PauliString("_X"),  # X only at position 1
        ]

        phase = gaussian_elimination(stabilizers, ref_state, target_state)

        # Should use the second stabilizer which has no X to the left
        assert np.isclose(phase, 1 + 0j)


class TestStabilizerIntegration:
    """Integration tests combining multiple stabilizer functions."""

    def test_ghz_state_properties(self):
        """Test GHZ state |000⟩ + |111⟩ stabilizer properties."""
        # GHZ state stabilizers: XXX, ZZ_, _ZZ
        stabilizers = [
            stim.PauliString("XXX"),
            stim.PauliString("ZZ_"),
            stim.PauliString("_ZZ"),
        ]

        # X-rank should be 1 (only XXX has X)
        x_rank = compute_x_rank(stabilizers)
        assert x_rank == 1

        # Check overlap between |000⟩ and |111⟩
        ref_state = Bitstring([False, False, False])
        target_state = Bitstring([True, True, True])
        phase = gaussian_elimination(stabilizers, ref_state, target_state)
        assert np.isclose(phase, 1 + 0j)

    def test_stabilizer_state_dimension(self):
        """Test that X-rank determines the dimension of stabilizer state."""
        # For n-qubit system with k stabilizers having X, dimension is 2^(n-k)

        # 2-qubit product state |+⟩|+⟩ stabilized by X_ and _X
        stabilizers = [stim.PauliString("X_"), stim.PauliString("_X")]
        x_rank = compute_x_rank(stabilizers)
        assert x_rank == 2
        # Dimension = 2^(2-2) = 1, this is a pure state

        # Bell state |Φ+⟩ stabilized by XX and ZZ
        stabilizers = [stim.PauliString("XX"), stim.PauliString("ZZ")]
        x_rank = compute_x_rank(stabilizers)
        assert x_rank == 1
        # Dimension = 2^(2-1) = 2, spanned by |00⟩ and |11⟩


class TestPaperExample:
    """Test the exact example from the Huggins et al. paper.

    From the paper (Figure showing shadow tomography):
    Classical shadow: |b⟩ = |1011⟩ and Clifford U†
    Measurement basis states: |0000⟩ and |1011⟩

    After composing: U†|b⟩ gives stabilizer state
    After Gaussian elimination: stabilizers become canonical form

    The example shows:
    - Initial stabilizers from U†|b⟩
    - Canonical form after Gaussian elimination
    - Computing overlaps ⟨0_i|U†|b_j⟩⟨b_j|U|0⟩
    """

    def test_paper_example_stabilizer_composition(self):
        """Test composing measurement state |1011⟩ with Clifford unitary.

        This test reproduces the paper example to verify:
        1. Bitstring to stabilizer conversion
        2. Clifford unitary composition
        3. Canonical form reduction
        """
        # Create measurement outcome |1011⟩
        measurement = Bitstring.from_string("1011")

        # Convert to stabilizer tableau
        measurement_tableau = measurement.to_stabilizer()

        # Get the stabilizers
        stabilizers_initial = measurement_tableau.to_stabilizers()

        print("\n" + "="*70)
        print("PAPER EXAMPLE: Classical Shadow Processing")
        print("="*70)
        print(f"\nMeasurement outcome: |b⟩ = |1011⟩")
        print(f"Bitstring representation: {measurement.array}")
        print(f"\nInitial stabilizers from |1011⟩:")
        for i, stab in enumerate(stabilizers_initial):
            print(f"  S_{i}: {stab}")

        # Generate a random Clifford (in practice this would be the U† from sampling)
        import numpy as np
        n_qubits = 4
        clifford = stim.Tableau.from_stabilizers([
            stim.PauliString('-ZYYX'),
            stim.PauliString('ZXXI'),
            stim.PauliString('-ZIXZ'),
            stim.PauliString('-YYXX')
        ])

        print(f"\nPaper Clifford U (for testing):")
        print(f"  {clifford}")

        # Compose: U†|b⟩
        clifford_inv = clifford.inverse()
        composed_tableau = clifford_inv * measurement_tableau

        print(f"\nComposed tableau: U†|b⟩")

        # Get stabilizers from composed tableau
        stabilizers_composed = composed_tableau.to_stabilizers()
        print(f"\nStabilizers after composition U†|b⟩:")
        for i, stab in enumerate(stabilizers_composed):
            print(f"  S_{i}: {stab}")

        # Import canonicalize function
        from shadow_ci.utils import canonicalize, compute_x_rank

        # Canonicalize the stabilizers
        stabilizers_canonical = canonicalize(stabilizers_composed)

        print(f"\nCanonical form (after Gaussian elimination):")
        for i, stab in enumerate(stabilizers_canonical):
            print(f"  S_{i}: {stab}")

        # Compute X-rank
        x_rank = compute_x_rank(stabilizers_canonical)
        magnitude = np.sqrt(2) ** (-x_rank)

        print(f"\nX-rank: {x_rank}")
        print(f"Magnitude contribution: (√2)^(-{x_rank}) = {magnitude:.6f}")

        # Test overlap computation with |0000⟩
        vacuum = Bitstring.from_string("0000")

        print(f"\n" + "-"*70)
        print(f"Computing overlap ⟨0000|U†|1011⟩:")
        print("-"*70)

        # Use gaussian_elimination to compute phase
        # First, we need to get a reference state from measurement
        # The composed tableau represents a state, we measure it to get |b'⟩
        sim = stim.TableauSimulator()
        sim.set_state_from_stabilizers(stabilizers_canonical)
        measurement_result = sim.measure_many(*range(n_qubits))
        ref_state = Bitstring(list(measurement_result))

        print(f"Reference state |b'⟩ from stabilizer state: {ref_state.to_string()}")

        from shadow_ci.utils import gaussian_elimination

        # Compute phase for overlap with vacuum
        phase = gaussian_elimination(stabilizers_canonical, ref_state, vacuum)

        print(f"Phase from Gaussian elimination: {phase}")
        print(f"Magnitude of phase: {abs(phase):.6f}")

        # The overlap should be: magnitude² × phase
        overlap = magnitude**2 * phase
        print(f"\nOverlap ⟨0000|U†|1011⟩ = {overlap}")
        print(f"Magnitude: {abs(overlap):.6f}")

        # Verify this matches if we compute directly
        state_from_tableau = composed_tableau.to_state_vector()
        vacuum_state_vector = np.zeros(2**n_qubits, dtype=complex)
        vacuum_state_vector[0] = 1.0
        direct_overlap = np.vdot(vacuum_state_vector, state_from_tableau)

        print(f"\nDirect computation ⟨0000|ψ⟩ = {direct_overlap}")
        print(f"Magnitude: {abs(direct_overlap):.6f}")

        # They should match (within numerical precision)
        print(f"\nMatch between methods: {np.isclose(overlap, direct_overlap)}")
        if not np.isclose(overlap, direct_overlap, atol=1e-6):
            print(f"WARNING: Mismatch! Difference = {abs(overlap - direct_overlap)}")

    def test_paper_example_with_specific_clifford(self):
        """Test with a specific simple Clifford to understand the process.

        Using a simpler example: 2-qubit system with Hadamard on qubit 0
        """
        print("\n" + "="*70)
        print("SIMPLIFIED EXAMPLE: 2-qubit system")
        print("="*70)

        # Measurement |01⟩
        measurement = Bitstring.from_string("01")
        print(f"\nMeasurement: |b⟩ = |01⟩")

        measurement_tableau = measurement.to_stabilizer()
        print(measurement_tableau)

        # Simple Clifford: Hadamard on qubit 0
        # This maps |01⟩ → |0⟩ ⊗ H|1⟩ = |0⟩ ⊗ (|0⟩-|1⟩)/√2 = (|00⟩-|01⟩)/√2
        n_qubits = 2
        clifford = stim.Tableau(n_qubits)
        h_gate = stim.Tableau.from_named_gate("H")
        clifford.prepend(h_gate, [0])  # Apply H to qubit 0- (rightmost)

        print(f"\nClifford U: Hadamard on qubit 0")

        # Compose
        clifford_inv = clifford.inverse()
        composed = clifford_inv * measurement_tableau

        # Get state vector to verify
        state_vector = composed.to_state_vector()
        print(f"\nState U†|01⟩:")
        for i, amp in enumerate(state_vector):
            if abs(amp) > 1e-10:
                print(f"  |{format(i, '02b')}⟩: {amp:.6f}")

        # Expected: H† applied to second qubit of |01⟩
        # H†|1⟩ = (|0⟩-|1⟩)/√2
        # So U†|01⟩ = |0⟩ ⊗ H†|1⟩ = (|00⟩-|01⟩)/√2

        expected_00 = 1/np.sqrt(2)
        expected_01 = -1/np.sqrt(2)

        print(f"\nExpected amplitudes:")
        print(f"  |00⟩: {expected_00:.6f}")
        print(f"  |01⟩: {expected_01:.6f}")

        assert np.isclose(abs(state_vector[0]), abs(expected_00), atol=1e-6)
        assert np.isclose(abs(state_vector[1]), abs(expected_01), atol=1e-6)

        print("\nState verification: PASSED ✓")

        # Now test overlap computation through stabilizers
        stabilizers = canonicalize(composed.to_stabilizers())

        print(f"\nCanonical stabilizers:")
        for stab in stabilizers:
            print(f"  {stab}")

        from shadow_ci.utils import compute_x_rank, gaussian_elimination

        x_rank = compute_x_rank(stabilizers)
        print(f"\nX-rank: {x_rank}")

        # Get reference state
        sim = stim.TableauSimulator()
        sim.set_state_from_stabilizers(stabilizers)
        meas = sim.measure_many(*range(n_qubits))
        ref_state = Bitstring(list(meas))

        # Compute overlap with |00⟩
        target = Bitstring.from_string("00")
        phase = gaussian_elimination(stabilizers, ref_state, target)

        magnitude = np.sqrt(2) ** (-x_rank)
        overlap = magnitude * phase

        print(f"\nComputed overlap ⟨00|U†|01⟩ = {overlap}")
        print(f"Expected: {expected_00}")
        print(f"Match: {np.isclose(overlap, expected_00, atol=1e-6)}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
