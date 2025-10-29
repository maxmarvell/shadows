"""Debug script to understand tableau composition and bit ordering issues."""

import numpy as np
import stim
from qiskit.quantum_info import Statevector
from shadow_ci.utils import Bitstring


def test_bitstring_to_stabilizer_ordering():
    """Verify that Bitstring.to_stabilizer() now produces correct ordering."""
    print("\n" + "="*80)
    print("VERIFYING BITSTRING.TO_STABILIZER() ORDERING")
    print("="*80)

    test_cases = [
        ("00", 0),
        ("01", 1),
        ("10", 2),
        ("11", 3),
        ("001", 1),
        ("010", 2),
        ("100", 4),
        ("101", 5),
    ]

    all_correct = True
    for bitstring_str, expected_index in test_cases:
        bits = Bitstring.from_string(bitstring_str)
        tableau = bits.to_stabilizer()
        state_vector = tableau.to_state_vector()
        actual_index = np.argmax(np.abs(state_vector))

        expected_qiskit = Statevector.from_label(bitstring_str)
        expected_qiskit_index = np.argmax(np.abs(expected_qiskit.data))

        match = (actual_index == expected_index == expected_qiskit_index)
        symbol = "✓" if match else "✗"

        print(f"\n{symbol} |{bitstring_str}⟩:")
        print(f"    Bitstring array: {bits.array}")
        print(f"    Expected index: {expected_index}")
        print(f"    Qiskit index:   {expected_qiskit_index}")
        print(f"    Tableau index:  {actual_index}")
        print(f"    Match: {match}")

        if not match:
            all_correct = False
            print(f"    ERROR: Mismatch detected!")

    print("\n" + "-"*80)
    if all_correct:
        print("✓ All Bitstring.to_stabilizer() tests PASSED")
    else:
        print("✗ Some Bitstring.to_stabilizer() tests FAILED")

    return all_correct


def test_single_gate_application():
    """Test applying a single gate to understand composition."""
    print("\n" + "="*80)
    print("TESTING SINGLE GATE APPLICATION")
    print("="*80)

    # Test 1: Apply X gate to |0⟩
    print("\nTest 1: X gate on |0⟩")
    print("-" * 40)

    state_0 = Bitstring.from_string("0")
    tableau_0 = state_0.to_stabilizer()

    # Create X gate
    x_gate = stim.Tableau.from_named_gate("X")

    # Apply X to the state
    result = x_gate * tableau_0

    result_state = result.to_state_vector()
    print(f"Input: |0⟩")
    print(f"Gate: X")
    print(f"Result state vector: {result_state}")
    print(f"Expected: |1⟩ = [0, 1]")

    expected_1 = Statevector.from_label("1").data
    match = np.allclose(result_state, expected_1)
    print(f"Match: {match} {'✓' if match else '✗'}")

    # Test 2: Apply H gate to |0⟩
    print("\n\nTest 2: H gate on |0⟩")
    print("-" * 40)

    h_gate = stim.Tableau.from_named_gate("H")
    result_h = h_gate * tableau_0

    result_h_state = result_h.to_state_vector()
    print(f"Input: |0⟩")
    print(f"Gate: H")
    print(f"Result state vector: {result_h_state}")
    print(f"Expected: |+⟩ = [1/√2, 1/√2]")

    expected_plus = np.array([1/np.sqrt(2), 1/np.sqrt(2)])
    match_h = np.allclose(np.abs(result_h_state), np.abs(expected_plus))
    print(f"Match: {match_h} {'✓' if match_h else '✗'}")

    # Test 3: Apply H gate to |1⟩
    print("\n\nTest 3: H gate on |1⟩")
    print("-" * 40)

    state_1 = Bitstring.from_string("1")
    tableau_1 = state_1.to_stabilizer()

    result_h1 = h_gate * tableau_1
    result_h1_state = result_h1.to_state_vector()

    print(f"Input: |1⟩")
    print(f"Gate: H")
    print(f"Result state vector: {result_h1_state}")
    print(f"Expected: |-⟩ = [1/√2, -1/√2]")

    expected_minus = np.array([1/np.sqrt(2), -1/np.sqrt(2)])
    match_h1 = np.allclose(result_h1_state, expected_minus, atol=1e-6)
    print(f"Match: {match_h1} {'✓' if match_h1 else '✗'}")


def test_two_qubit_gate_application():
    """Test applying gates to two-qubit states."""
    print("\n" + "="*80)
    print("TESTING TWO-QUBIT GATE APPLICATION")
    print("="*80)

    # Create |01⟩
    state_01 = Bitstring.from_string("01")
    tableau_01 = state_01.to_stabilizer()

    print(f"\nInitial state: |01⟩")
    initial_state = tableau_01.to_state_vector()
    print(f"State vector: {initial_state}")

    # Verify it's correct
    expected_01 = Statevector.from_label("01").data
    print(f"Expected:     {expected_01}")
    print(f"Initial state correct: {np.allclose(initial_state, expected_01)}")

    # Apply H gate to qubit 0 (rightmost, least significant)
    print("\n" + "-"*80)
    print("Applying H to qubit 0 (rightmost)")
    print("-"*80)

    n_qubits = 2
    h_on_q0 = stim.Tableau(n_qubits)
    h_gate = stim.Tableau.from_named_gate("H")
    h_on_q0.prepend(h_gate, [0])

    result_q0 = h_on_q0 * tableau_01
    result_q0_state = result_q0.to_state_vector()

    print(f"Result state vector: {result_q0_state}")
    print(f"\nNon-zero amplitudes:")
    for i, amp in enumerate(result_q0_state):
        if abs(amp) > 1e-10:
            print(f"  |{format(i, '02b')}⟩: {amp:.6f}")

    # Expected: H acts on rightmost qubit of |01⟩
    # |01⟩ = |0⟩ ⊗ |1⟩
    # H|01⟩ = |0⟩ ⊗ H|1⟩ = |0⟩ ⊗ (|0⟩-|1⟩)/√2 = (|00⟩-|01⟩)/√2
    print(f"\nExpected: (|00⟩ - |01⟩)/√2")
    print(f"  |00⟩: {1/np.sqrt(2):.6f}")
    print(f"  |01⟩: {-1/np.sqrt(2):.6f}")

    # Apply H gate to qubit 1 (leftmost, most significant)
    print("\n" + "-"*80)
    print("Applying H to qubit 1 (leftmost)")
    print("-"*80)

    h_on_q1 = stim.Tableau(n_qubits)
    h_on_q1.prepend(h_gate, [1])

    result_q1 = h_on_q1 * tableau_01
    result_q1_state = result_q1.to_state_vector()

    print(f"Result state vector: {result_q1_state}")
    print(f"\nNon-zero amplitudes:")
    for i, amp in enumerate(result_q1_state):
        if abs(amp) > 1e-10:
            print(f"  |{format(i, '02b')}⟩: {amp:.6f}")

    # Expected: H acts on leftmost qubit of |01⟩
    # |01⟩ in Qiskit means bit 1=0, bit 0=1
    # H on qubit 1: H|0⟩ ⊗ |1⟩ = (|0⟩+|1⟩)/√2 ⊗ |1⟩ = (|01⟩+|11⟩)/√2
    print(f"\nExpected: (|01⟩ + |11⟩)/√2")
    print(f"  |01⟩: {1/np.sqrt(2):.6f}")
    print(f"  |11⟩: {1/np.sqrt(2):.6f}")


def test_inverse_composition():
    """Test U† composition with measurement states."""
    print("\n" + "="*80)
    print("TESTING INVERSE COMPOSITION")
    print("="*80)

    # Create measurement |01⟩
    measurement = Bitstring.from_string("01")
    measurement_tableau = measurement.to_stabilizer()

    print(f"\nMeasurement: |b⟩ = |01⟩")
    meas_state = measurement_tableau.to_state_vector()
    print(f"State vector: {meas_state}")

    # Create Hadamard on qubit 1
    n_qubits = 2
    clifford = stim.Tableau(n_qubits)
    h_gate = stim.Tableau.from_named_gate("H")
    clifford.prepend(h_gate, [1])

    print(f"\nClifford U: H on qubit 1")

    # Test U|01⟩
    print("\n" + "-"*40)
    print("Forward: U|01⟩")
    print("-"*40)
    forward = clifford * measurement_tableau
    forward_state = forward.to_state_vector()
    print(f"Result: {forward_state}")
    print(f"Expected: (|01⟩ + |11⟩)/√2")

    # Test U†|01⟩
    print("\n" + "-"*40)
    print("Inverse: U†|01⟩")
    print("-"*40)
    clifford_inv = clifford.inverse()
    inverse = clifford_inv * measurement_tableau
    inverse_state = inverse.to_state_vector()
    print(f"Result: {inverse_state}")
    print(f"Expected: (|01⟩ - |11⟩)/√2  (since H† = H and adds phase)")

    print("\n" + "-"*40)
    print("Checking if U†U = I")
    print("-"*40)
    identity_check = clifford_inv * clifford
    print(f"U†U applied to |01⟩:")
    identity_result = identity_check * measurement_tableau
    identity_state = identity_result.to_state_vector()
    print(f"Result: {identity_state}")
    print(f"Expected: {meas_state}")
    print(f"Match: {np.allclose(identity_state, meas_state)}")


def test_composition_order():
    """Test the order of tableau composition."""
    print("\n" + "="*80)
    print("TESTING COMPOSITION ORDER")
    print("="*80)

    # Create |0⟩
    state = Bitstring.from_string("0")
    tableau = state.to_stabilizer()

    print("\nInitial state: |0⟩")

    # Create X gate
    x_gate = stim.Tableau.from_named_gate("X")

    # Test both orders
    print("\n" + "-"*40)
    print("Order 1: X * tableau")
    result1 = x_gate * tableau
    state1 = result1.to_state_vector()
    print(f"Result: {state1}")

    print("\n" + "-"*40)
    print("Order 2: tableau * X")
    result2 = tableau.then(x_gate)
    state2 = result2.to_state_vector()
    print(f"Result: {state2}")

    print("\n" + "-"*40)
    print("Expected: |1⟩ = [0, 1]")
    print(f"Order 1 correct: {np.allclose(state1, [0, 1])}")
    print(f"Order 2 correct: {np.allclose(state2, [0, 1])}")


if __name__ == "__main__":
    test_bitstring_to_stabilizer_ordering()
    test_single_gate_application()
    test_two_qubit_gate_application()
    test_inverse_composition()
    test_composition_order()

    print("\n" + "="*80)
    print("DEBUG COMPLETE")
    print("="*80)
