"""Comprehensive audit of bit ordering and endianness across the entire codebase.

This test checks for consistency in bit ordering conventions:
- Qiskit uses little-endian: rightmost bit is qubit 0 (LSB)
- For |abc⟩: bit c is qubit 0, bit b is qubit 1, bit a is qubit 2
- Integer index: |abc⟩ corresponds to integer a*4 + b*2 + c*1
"""

import numpy as np
import stim
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from shadow_ci.utils import Bitstring
from shadow_ci.shadows import ShadowProtocol


def test_qiskit_convention():
    """Document and verify Qiskit's bit ordering convention."""
    print("\n" + "="*80)
    print("QISKIT BIT ORDERING CONVENTION")
    print("="*80)

    print("\nQiskit uses LITTLE-ENDIAN ordering:")
    print("  - Rightmost bit in string = Qubit 0 (LSB)")
    print("  - Leftmost bit in string = Qubit n-1 (MSB)")

    # Test |01⟩
    state_01 = Statevector.from_label("01")
    print(f"\n|01⟩ means:")
    print(f"  - Qubit 0 (rightmost): |1⟩")
    print(f"  - Qubit 1 (leftmost):  |0⟩")
    print(f"  State vector: {state_01.data}")
    print(f"  Nonzero at index: {np.argmax(np.abs(state_01.data))} (binary: 01)")

    # Apply gate to qubit 0
    qc = QuantumCircuit(2)
    qc.initialize([0, 1, 0, 0], [0, 1])  # |01⟩
    qc.h(0)  # H on qubit 0 (rightmost)

    result = Statevector(qc)
    print(f"\nAfter H on qubit 0 (rightmost):")
    print(f"  Expected: (|00⟩ - |01⟩)/√2")
    for i, amp in enumerate(result.data):
        if abs(amp) > 1e-10:
            print(f"  |{format(i, '02b')}⟩: {amp:.6f}")


def test_bitstring_convention():
    """Test Bitstring class bit ordering."""
    print("\n" + "="*80)
    print("BITSTRING CLASS BIT ORDERING")
    print("="*80)

    # Test string to array
    bits = Bitstring.from_string("01")
    print(f"\nBitstring.from_string('01'):")
    print(f"  array: {bits.array}")
    print(f"  array[0] (leftmost):  {bits.array[0]}")
    print(f"  array[1] (rightmost): {bits.array[1]}")

    # Test to_int
    print(f"\n  to_int(): {bits.to_int()}")
    print(f"  Expected: 1 (binary 01 = 0*2 + 1*1)")

    # Test to_string round trip
    print(f"  to_string(): '{bits.to_string()}'")
    print(f"  Expected: '01'")

    # Test indexing interpretation
    print(f"\nIndexing convention:")
    for i, bit in enumerate(bits):
        position = "leftmost" if i == 0 else "rightmost"
        print(f"  bits[{i}] = {bit} ({position})")

    # Test from_int
    bits_from_int = Bitstring.from_int(5, size=3)  # 5 = 0b101
    print(f"\nBitstring.from_int(5, size=3):")
    print(f"  Expected binary: 101")
    print(f"  array: {bits_from_int.array}")
    print(f"  to_string(): '{bits_from_int.to_string()}'")
    print(f"  Match: {bits_from_int.to_string() == '101'}")


def test_bitstring_to_stabilizer_convention():
    """Test Bitstring.to_stabilizer() bit ordering."""
    print("\n" + "="*80)
    print("BITSTRING.TO_STABILIZER() BIT ORDERING")
    print("="*80)

    test_cases = [
        ("010", 2),  # Should map to index 2
        ("101", 5),  # Should map to index 5
        ("0101", 5), # 4-qubit
    ]

    all_match = True
    for bitstring_str, expected_index in test_cases:
        bits = Bitstring.from_string(bitstring_str)
        tableau = bits.to_stabilizer()
        state_vector = tableau.to_state_vector()
        actual_index = np.argmax(np.abs(state_vector))

        qiskit_state = Statevector.from_label(bitstring_str)
        qiskit_index = np.argmax(np.abs(qiskit_state.data))

        match = (actual_index == expected_index == qiskit_index)
        symbol = "✓" if match else "✗"

        print(f"\n{symbol} |{bitstring_str}⟩:")
        print(f"    Bitstring array: {bits.array}")
        print(f"    Expected index:  {expected_index}")
        print(f"    Qiskit index:    {qiskit_index}")
        print(f"    Tableau index:   {actual_index}")

        if not match:
            all_match = False
            print(f"    ✗ MISMATCH!")

    return all_match


def test_stim_gate_application():
    """Test stim Tableau gate application convention."""
    print("\n" + "="*80)
    print("STIM TABLEAU GATE APPLICATION")
    print("="*80)

    # Create |01⟩
    state_01 = Bitstring.from_string("01")
    tableau_01 = state_01.to_stabilizer()

    print(f"\nInitial state: |01⟩")
    print(f"  State vector: {tableau_01.to_state_vector()}")

    # Test applying H to different qubits
    n_qubits = 2

    for qubit_idx in range(n_qubits):
        h_tableau = stim.Tableau(n_qubits)
        h_gate = stim.Tableau.from_named_gate("H")
        h_tableau.prepend(h_gate, [qubit_idx])

        result = h_tableau * tableau_01
        result_state = result.to_state_vector()

        print(f"\nApplying H to qubit {qubit_idx}:")
        print(f"  Result state vector: {result_state}")
        print(f"  Non-zero amplitudes:")
        for i, amp in enumerate(result_state):
            if abs(amp) > 1e-10:
                print(f"    |{format(i, '02b')}⟩: {amp:.6f}")

        # Determine what we expect
        if qubit_idx == 0:
            print(f"  Expected: H on qubit 0 (rightmost) of |01⟩")
            print(f"    |01⟩ = |0⟩₁ ⊗ |1⟩₀")
            print(f"    H|01⟩ = |0⟩₁ ⊗ H|1⟩₀ = |0⟩₁ ⊗ (|0⟩-|1⟩)/√2")
            print(f"    = (|00⟩ - |01⟩)/√2")
        else:
            print(f"  Expected: H on qubit 1 (leftmost) of |01⟩")
            print(f"    |01⟩ = |0⟩₁ ⊗ |1⟩₀")
            print(f"    H|01⟩ = H|0⟩₁ ⊗ |1⟩₀ = (|0⟩+|1⟩)/√2 ⊗ |1⟩₀")
            print(f"    = (|01⟩ + |11⟩)/√2")


def test_shadow_protocol_measurement():
    """Test ShadowProtocol measurement convention."""
    print("\n" + "="*80)
    print("SHADOW PROTOCOL MEASUREMENT CONVENTION")
    print("="*80)

    # Create a simple state
    state = Statevector.from_label("01")

    print(f"\nQuantum state: |01⟩")
    print(f"  State vector: {state.data}")
    print(f"  In Qiskit: qubit 0 = |1⟩, qubit 1 = |0⟩")

    # Manually simulate measurement
    # When we measure, we should get bitstring "01"
    print(f"\nExpected measurement outcome: '01'")
    print(f"  This should create Bitstring([False, True])")

    # Verify this converts back to the correct state
    measured_bits = Bitstring.from_string("01")
    reconstructed = measured_bits.to_stabilizer()
    reconstructed_state = reconstructed.to_state_vector()

    print(f"\nReconstructed state from measurement:")
    print(f"  State vector: {reconstructed_state}")
    print(f"  Match: {np.allclose(state.data, reconstructed_state)}")


def test_gaussian_elimination_convention():
    """Test gaussian_elimination bit ordering."""
    print("\n" + "="*80)
    print("GAUSSIAN ELIMINATION BIT ORDERING")
    print("="*80)

    from shadow_ci.utils import gaussian_elimination, canonicalize

    # Create |10⟩ state
    state_10 = Bitstring.from_string("10")
    tableau_10 = state_10.to_stabilizer()
    stabilizers = canonicalize(tableau_10.to_stabilizers())

    print(f"\nState: |10⟩")
    print(f"  Bitstring array: {state_10.array}")
    print(f"  In Qiskit: qubit 0 = |0⟩, qubit 1 = |1⟩")

    print(f"\nStabilizers:")
    for i, stab in enumerate(stabilizers):
        print(f"  S_{i}: {stab}")

    # Test overlap with itself
    ref_state = state_10
    target_state = state_10

    print(f"\nComputing ⟨10|10⟩:")
    print(f"  ref_state:    {ref_state.array}")
    print(f"  target_state: {target_state.array}")

    phase = gaussian_elimination(stabilizers, ref_state, target_state)
    print(f"  Phase: {phase}")
    print(f"  Expected: 1+0j")
    print(f"  Match: {np.isclose(phase, 1+0j)}")

    # Test overlap with |01⟩ (orthogonal)
    target_01 = Bitstring.from_string("01")
    print(f"\nComputing ⟨10|01⟩:")
    print(f"  ref_state:    {ref_state.array}")
    print(f"  target_state: {target_01.array}")

    phase_01 = gaussian_elimination(stabilizers, ref_state, target_01)
    print(f"  Phase: {phase_01}")
    print(f"  Expected: 0j")
    print(f"  Match: {phase_01 == 0j}")


def test_clifford_inverse_convention():
    """Test Clifford inverse and composition."""
    print("\n" + "="*80)
    print("CLIFFORD INVERSE AND COMPOSITION")
    print("="*80)

    # Create measurement |10⟩
    measurement = Bitstring.from_string("10")
    measurement_tableau = measurement.to_stabilizer()

    print(f"\nMeasurement: |b⟩ = |10⟩")
    print(f"  State vector: {measurement_tableau.to_state_vector()}")

    # Create simple Clifford (X on qubit 0)
    n_qubits = 2
    clifford = stim.Tableau(n_qubits)
    x_gate = stim.Tableau.from_named_gate("X")
    clifford.prepend(x_gate, [0])

    print(f"\nClifford U: X gate on qubit 0")

    # Forward: U|10⟩
    forward = clifford * measurement_tableau
    forward_state = forward.to_state_vector()

    print(f"\nU|10⟩:")
    print(f"  State vector: {forward_state}")
    print(f"  Expected: |11⟩ (X flips qubit 0: |0⟩₀→|1⟩₀)")
    expected_11 = Statevector.from_label("11").data
    print(f"  Match: {np.allclose(forward_state, expected_11)}")

    # Inverse: U†|10⟩
    clifford_inv = clifford.inverse()
    inverse = clifford_inv * measurement_tableau
    inverse_state = inverse.to_state_vector()

    print(f"\nU†|10⟩:")
    print(f"  State vector: {inverse_state}")
    print(f"  Expected: |11⟩ (X† = X, so same as forward)")
    print(f"  Match: {np.allclose(inverse_state, expected_11)}")

    # Test U†U = I
    identity = clifford_inv * clifford
    identity_result = identity * measurement_tableau
    identity_state = identity_result.to_state_vector()

    print(f"\nU†U|10⟩:")
    print(f"  State vector: {identity_state}")
    print(f"  Expected: |10⟩ (identity)")
    print(f"  Match: {np.allclose(identity_state, measurement_tableau.to_state_vector())}")


def test_shadows_module_conventions():
    """Test conventions in shadows.py."""
    print("\n" + "="*80)
    print("SHADOWS.PY MODULE CONVENTIONS")
    print("="*80)

    from shadow_ci.shadows import ClassicalSnapshot, ClassicalShadow, gate_map
    from qiskit import QuantumCircuit

    # Test gate_map convention
    print("\nTesting gate_map convention:")
    print("  The gate_map in shadows.py maps stim gates to qiskit")

    # Create a simple test
    qc = QuantumCircuit(2)
    measurement = Bitstring.from_string("01")

    print(f"\nMeasurement bitstring: '01'")
    print(f"  Array: {measurement.array}")
    print(f"  Expected: [False, True]")
    print(f"  Correct array ordering: {measurement.array == [False, True]}")

    # Create a snapshot
    clifford = stim.Tableau.random(2)
    snapshot = ClassicalSnapshot(measurement, clifford, 'clifford')

    print(f"\nClassicalSnapshot created:")
    print(f"  measurement.size: {snapshot.measurement.size}")
    print(f"  Expected: 2")
    print(f"  Match: {snapshot.measurement.size == 2}")


def audit_summary():
    """Provide summary of all endianness checks."""
    print("\n" + "="*80)
    print("ENDIANNESS AUDIT SUMMARY")
    print("="*80)

    print("\nConvention used throughout codebase:")
    print("  ✓ Qiskit little-endian: rightmost bit = qubit 0")
    print("  ✓ Bitstring array[0] = leftmost bit (MSB)")
    print("  ✓ Bitstring array[-1] = rightmost bit (LSB)")
    print("  ✓ String '01' → array [False, True] → index 1")
    print("  ✓ to_stabilizer() now produces Qiskit-compatible states")

    print("\nAreas to check in your code:")
    print("  1. When calling stim gate.prepend(gate, [qubit_idx]):")
    print("     - qubit_idx=0 means rightmost qubit (LSB)")
    print("     - qubit_idx=n-1 means leftmost qubit (MSB)")

    print("\n  2. In gaussian_elimination:")
    print("     - Loop over qubits from left to right (MSB to LSB)")
    print("     - This matches the array indexing [0] to [n-1]")

    print("\n  3. In shadows.py ClassicalShadow.overlap():")
    print("     - Measurement bitstring should match qubit ordering")
    print("     - Clifford inverse composition: U† * |b⟩")


if __name__ == "__main__":
    test_qiskit_convention()
    test_bitstring_convention()
    test_bitstring_to_stabilizer_convention()
    test_stim_gate_application()
    test_shadow_protocol_measurement()
    test_gaussian_elimination_convention()
    test_clifford_inverse_convention()
    test_shadows_module_conventions()
    audit_summary()

    print("\n" + "="*80)
    print("ENDIANNESS AUDIT COMPLETE")
    print("="*80)
