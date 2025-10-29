"""Comprehensive test suite for the Bitstring class."""

import pytest
import numpy as np
import stim
from qiskit.quantum_info import Statevector
from shadow_ci.utils import Bitstring


class TestBitstringConstruction:
    """Test Bitstring construction and initialization."""

    def test_construct_from_list(self):
        """Test basic construction from list of bools."""
        bits = Bitstring([True, False, True, False])
        assert bits.size == 4
        assert bits[0] == True
        assert bits[1] == False
        assert bits[2] == True
        assert bits[3] == False

    def test_construct_empty(self):
        """Test construction of empty bitstring."""
        bits = Bitstring([])
        assert bits.size == 0

    def test_construct_all_zeros(self):
        """Test construction of all zeros."""
        bits = Bitstring([False, False, False])
        assert bits.size == 3
        assert all(not bit for bit in bits)

    def test_construct_all_ones(self):
        """Test construction of all ones."""
        bits = Bitstring([True, True, True])
        assert bits.size == 3
        assert all(bit for bit in bits)

    def test_from_string(self):
        """Test construction from string."""
        bits = Bitstring.from_string("0101")
        assert bits.size == 4
        assert bits[0] == False
        assert bits[1] == True
        assert bits[2] == False
        assert bits[3] == True

    def test_from_string_empty(self):
        """Test from_string with empty string."""
        bits = Bitstring.from_string("")
        assert bits.size == 0

    def test_from_string_all_zeros(self):
        """Test from_string with all zeros."""
        bits = Bitstring.from_string("0000")
        assert bits.size == 4
        assert all(not bit for bit in bits)

    def test_from_string_all_ones(self):
        """Test from_string with all ones."""
        bits = Bitstring.from_string("1111")
        assert bits.size == 4
        assert all(bit for bit in bits)

    def test_from_int(self):
        """Test construction from integer."""
        bits = Bitstring.from_int(5, size=4)  # 5 = 0b0101
        assert bits.size == 4
        assert bits.to_string() == "0101"

    def test_from_int_zero(self):
        """Test from_int with zero."""
        bits = Bitstring.from_int(0, size=4)
        assert bits.to_string() == "0000"

    def test_from_int_max(self):
        """Test from_int with max value."""
        bits = Bitstring.from_int(15, size=4)  # 15 = 0b1111
        assert bits.to_string() == "1111"

    def test_from_int_requires_padding(self):
        """Test from_int with value requiring leading zeros."""
        bits = Bitstring.from_int(1, size=5)  # Should be "00001"
        assert bits.to_string() == "00001"

    def test_random_bitstring(self):
        """Test random bitstring generation."""
        bits = Bitstring.random(10)
        assert bits.size == 10
        assert all(isinstance(bit, bool) for bit in bits)

    def test_random_bitstring_reproducible(self):
        """Test that random bitstring is reproducible with seed."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        bits1 = Bitstring.random(10, rng=rng1)
        bits2 = Bitstring.random(10, rng=rng2)

        assert bits1 == bits2

    def test_random_bitstring_different_seeds(self):
        """Test that different seeds give different results."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(123)

        bits1 = Bitstring.random(100, rng=rng1)
        bits2 = Bitstring.random(100, rng=rng2)

        # Extremely unlikely to be equal
        assert bits1 != bits2


class TestBitstringProperties:
    """Test Bitstring properties and methods."""

    def test_size_property(self):
        """Test size property."""
        bits = Bitstring([True, False, True])
        assert bits.size == 3

    def test_len(self):
        """Test len() function."""
        bits = Bitstring([True, False, True])
        assert len(bits) == 3

    def test_getitem(self):
        """Test indexing."""
        bits = Bitstring([True, False, True, False])
        assert bits[0] == True
        assert bits[1] == False
        assert bits[2] == True
        assert bits[3] == False

    def test_getitem_negative_index(self):
        """Test negative indexing."""
        bits = Bitstring([True, False, True, False])
        assert bits[-1] == False
        assert bits[-2] == True

    def test_setitem(self):
        """Test item assignment."""
        bits = Bitstring([True, False, True, False])
        bits[1] = True
        assert bits[1] == True
        assert bits.to_string() == "1110"

    def test_iteration(self):
        """Test that bitstring is iterable."""
        bits = Bitstring([True, False, True])
        result = [b for b in bits]
        assert result == [True, False, True]

    def test_iteration_enumerate(self):
        """Test enumerate works."""
        bits = Bitstring([True, False, True])
        for i, bit in enumerate(bits):
            assert bit == bits[i]


class TestBitstringEquality:
    """Test Bitstring equality comparisons."""

    def test_equality_same(self):
        """Test equality of identical bitstrings."""
        bits1 = Bitstring([True, False, True])
        bits2 = Bitstring([True, False, True])
        assert bits1 == bits2

    def test_equality_different(self):
        """Test inequality of different bitstrings."""
        bits1 = Bitstring([True, False, True])
        bits2 = Bitstring([True, True, True])
        assert bits1 != bits2

    def test_equality_different_length(self):
        """Test inequality of different length bitstrings."""
        bits1 = Bitstring([True, False])
        bits2 = Bitstring([True, False, True])
        assert bits1 != bits2

    def test_equality_empty(self):
        """Test equality of empty bitstrings."""
        bits1 = Bitstring([])
        bits2 = Bitstring([])
        assert bits1 == bits2

    def test_equality_with_non_bitstring(self):
        """Test equality with non-Bitstring returns False."""
        bits = Bitstring([True, False])
        assert bits != [True, False]
        assert bits != "10"
        assert bits != 2


class TestBitstringConversions:
    """Test Bitstring conversion methods."""

    def test_to_string(self):
        """Test conversion to string."""
        bits = Bitstring([True, False, True, False])
        assert bits.to_string() == "1010"

    def test_to_string_empty(self):
        """Test to_string with empty bitstring."""
        bits = Bitstring([])
        assert bits.to_string() == ""

    def test_to_string_all_zeros(self):
        """Test to_string with all zeros."""
        bits = Bitstring([False, False, False])
        assert bits.to_string() == "000"

    def test_to_string_all_ones(self):
        """Test to_string with all ones."""
        bits = Bitstring([True, True, True])
        assert bits.to_string() == "111"

    def test_to_int(self):
        """Test conversion to integer."""
        bits = Bitstring([True, False, True, False])  # "1010" = 10
        assert bits.to_int() == 10

    def test_to_int_zero(self):
        """Test to_int with all zeros."""
        bits = Bitstring([False, False, False])
        assert bits.to_int() == 0

    def test_to_int_max(self):
        """Test to_int with all ones."""
        bits = Bitstring([True, True, True])  # "111" = 7
        assert bits.to_int() == 7

    def test_round_trip_string(self):
        """Test round trip: Bitstring -> string -> Bitstring."""
        original = Bitstring([True, False, True, False])
        string = original.to_string()
        restored = Bitstring.from_string(string)
        assert original == restored

    def test_round_trip_int(self):
        """Test round trip: Bitstring -> int -> Bitstring."""
        original = Bitstring([True, False, True, False])
        value = original.to_int()
        restored = Bitstring.from_int(value, size=original.size)
        assert original == restored

    def test_multiple_conversions(self):
        """Test multiple conversions remain consistent."""
        bits = Bitstring([True, False, True])
        assert Bitstring.from_string(bits.to_string()) == bits
        assert Bitstring.from_int(bits.to_int(), size=bits.size) == bits


class TestBitstringStabilizer:
    """Test Bitstring to stabilizer tableau conversion."""

    def test_to_stabilizer_all_zeros(self):
        """Test stabilizer for |000⟩."""
        bits = Bitstring([False, False, False])
        tableau = bits.to_stabilizer()

        assert isinstance(tableau, stim.Tableau)
        assert len(tableau) == 3

        # Get the quantum state from the tableau
        state_from_tableau = tableau.to_state_vector()
        expected_state = Statevector.from_label("000")

        # Compare state vectors
        assert np.allclose(state_from_tableau, expected_state.data), \
            f"Tableau state doesn't match expected state for |000⟩"

    def test_to_stabilizer_all_ones(self):
        """Test stabilizer for |111⟩."""
        bits = Bitstring([True, True, True])
        tableau = bits.to_stabilizer()

        assert isinstance(tableau, stim.Tableau)
        assert len(tableau) == 3

        # Get the quantum state from the tableau
        state_from_tableau = tableau.to_state_vector()
        expected_state = Statevector.from_label("111")

        # Compare state vectors
        assert np.allclose(state_from_tableau, expected_state.data), \
            f"Tableau state doesn't match expected state for |111⟩"

    def test_to_stabilizer_single_bit(self):
        """Test stabilizer for single qubit states."""
        # |0⟩
        bits0 = Bitstring([False])
        tableau0 = bits0.to_stabilizer()
        assert len(tableau0) == 1

        state_from_tableau0 = tableau0.to_state_vector()
        expected_state0 = Statevector.from_label("0")
        assert np.allclose(state_from_tableau0, expected_state0.data), \
            f"Tableau state doesn't match expected state for |0⟩"

        # |1⟩
        bits1 = Bitstring([True])
        tableau1 = bits1.to_stabilizer()
        assert len(tableau1) == 1

        state_from_tableau1 = tableau1.to_state_vector()
        expected_state1 = Statevector.from_label("1")
        assert np.allclose(state_from_tableau1, expected_state1.data), \
            f"Tableau state doesn't match expected state for |1⟩"

    def test_to_stabilizer_mixed(self):
        """Test stabilizer for mixed state like |0101⟩."""
        bits = Bitstring([False, True, False, True])
        tableau = bits.to_stabilizer()

        assert isinstance(tableau, stim.Tableau)
        assert len(tableau) == 4

        # Get the quantum state from the tableau
        state_from_tableau = tableau.to_state_vector()
        expected_state = Statevector.from_label("0101")

        # Compare state vectors
        assert np.allclose(state_from_tableau, expected_state.data), \
            f"Tableau state doesn't match expected state for |0101⟩"

    def test_to_stabilizer_all_computational_basis_states(self):
        """Test that stabilizer correctly represents all computational basis states."""
        # Test all 2-qubit computational basis states
        for i in range(4):
            binary_str = format(i, '02b')
            bits = Bitstring.from_string(binary_str)
            tableau = bits.to_stabilizer()

            state_from_tableau = tableau.to_state_vector()
            expected_state = Statevector.from_label(binary_str)

            assert np.allclose(state_from_tableau, expected_state.data), \
                f"Tableau state doesn't match expected state for |{binary_str}⟩"

    def test_to_stabilizer_three_qubit_states(self):
        """Test stabilizer for all 3-qubit computational basis states."""
        for i in range(8):
            binary_str = format(i, '03b')
            bits = Bitstring.from_string(binary_str)
            tableau = bits.to_stabilizer()

            state_from_tableau = tableau.to_state_vector()
            expected_state = Statevector.from_label(binary_str)

            assert np.allclose(state_from_tableau, expected_state.data), \
                f"Tableau state doesn't match expected state for |{binary_str}⟩"


class TestBitstringEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_bit_false(self):
        """Test single bit bitstring with False."""
        bits = Bitstring([False])
        assert bits.size == 1
        assert bits[0] == False
        assert bits.to_string() == "0"
        assert bits.to_int() == 0

    def test_single_bit_true(self):
        """Test single bit bitstring with True."""
        bits = Bitstring([True])
        assert bits.size == 1
        assert bits[0] == True
        assert bits.to_string() == "1"
        assert bits.to_int() == 1

    def test_large_bitstring(self):
        """Test large bitstring."""
        size = 100
        bits = Bitstring([False] * size)
        assert bits.size == size
        assert bits.to_int() == 0

    def test_all_possible_2bit_states(self):
        """Test all possible 2-bit states."""
        states = [
            ([False, False], "00", 0),
            ([False, True], "01", 1),
            ([True, False], "10", 2),
            ([True, True], "11", 3),
        ]

        for bits_list, expected_str, expected_int in states:
            bits = Bitstring(bits_list)
            assert bits.to_string() == expected_str
            assert bits.to_int() == expected_int

    def test_all_possible_3bit_states(self):
        """Test all possible 3-bit states."""
        for i in range(8):
            bits = Bitstring.from_int(i, size=3)
            assert bits.to_int() == i
            assert bits.size == 3


class TestBitstringEquivalenceWithStatevector:
    """Test that Bitstring equivalence matches quantum state equivalence."""

    def test_computational_basis_equivalence(self):
        """Test that equivalent bitstrings represent same computational basis state."""
        # Create bitstring |101⟩
        bits1 = Bitstring([True, False, True])
        bits2 = Bitstring.from_string("101")
        bits3 = Bitstring.from_int(5, size=3)

        # All should be equal
        assert bits1 == bits2
        assert bits2 == bits3
        assert bits1 == bits3

        # Create corresponding Statevector
        state = Statevector.from_label("101")

        # Check that all bitstrings convert to same representation
        assert bits1.to_string() == "101"
        assert bits2.to_string() == "101"
        assert bits3.to_string() == "101"

        # Check integer conversion matches
        assert bits1.to_int() == 5
        assert bits2.to_int() == 5
        assert bits3.to_int() == 5

    def test_computational_basis_all_states(self):
        """Test that all computational basis states are unique and equivalent."""
        n_qubits = 3
        n_states = 2**n_qubits

        for i in range(n_states):
            # Create bitstring in different ways
            binary_str = format(i, f'0{n_qubits}b')
            bits_from_int = Bitstring.from_int(i, size=n_qubits)
            bits_from_string = Bitstring.from_string(binary_str)

            # Create corresponding quantum state
            state = Statevector.from_label(binary_str)

            # Check equivalence
            assert bits_from_int == bits_from_string
            assert bits_from_int.to_int() == i
            assert bits_from_string.to_int() == i

    def test_vacuum_state_equivalence(self):
        """Test that vacuum state |000...0⟩ is properly represented."""
        for n_qubits in [1, 2, 3, 4, 5]:
            bits1 = Bitstring([False] * n_qubits)
            bits2 = Bitstring.from_int(0, size=n_qubits)
            bits3 = Bitstring.from_string("0" * n_qubits)

            # All should be equivalent
            assert bits1 == bits2
            assert bits2 == bits3

            # Should all convert to integer 0
            assert bits1.to_int() == 0
            assert bits2.to_int() == 0
            assert bits3.to_int() == 0

            # Create corresponding quantum state
            state_label = "0" * n_qubits
            state = Statevector.from_label(state_label)

            # Bitstring should represent the vacuum
            assert bits1.to_string() == state_label

    def test_max_excitation_state_equivalence(self):
        """Test that max excitation state |111...1⟩ is properly represented."""
        for n_qubits in [1, 2, 3, 4, 5]:
            max_value = 2**n_qubits - 1
            bits1 = Bitstring([True] * n_qubits)
            bits2 = Bitstring.from_int(max_value, size=n_qubits)
            bits3 = Bitstring.from_string("1" * n_qubits)

            # All should be equivalent
            assert bits1 == bits2
            assert bits2 == bits3

            # Should all convert to max value
            assert bits1.to_int() == max_value
            assert bits2.to_int() == max_value
            assert bits3.to_int() == max_value

            # Create corresponding quantum state
            state_label = "1" * n_qubits
            state = Statevector.from_label(state_label)

            # Bitstring should represent the state
            assert bits1.to_string() == state_label

    def test_single_excitation_equivalence(self):
        """Test single excitation states are properly represented."""
        n_qubits = 4

        for i in range(n_qubits):
            # Create state with single bit set at position i
            bits_list = [False] * n_qubits
            bits_list[i] = True

            bits1 = Bitstring(bits_list)
            expected_int = 2**(n_qubits - 1 - i)  # Binary place value
            bits2 = Bitstring.from_int(expected_int, size=n_qubits)

            # Should be equivalent
            assert bits1 == bits2
            assert bits1.to_int() == expected_int


class TestBitstringMutability:
    """Test Bitstring mutability and modification."""

    def test_modify_single_bit(self):
        """Test modifying a single bit."""
        bits = Bitstring([True, False, True])
        original_string = bits.to_string()

        bits[1] = True
        assert bits.to_string() == "111"
        assert bits.to_string() != original_string

    def test_modification_creates_new_value(self):
        """Test that modification changes the value."""
        bits1 = Bitstring([True, False, True])
        bits2 = Bitstring([True, False, True])

        assert bits1 == bits2

        bits1[0] = False
        assert bits1 != bits2

    def test_multiple_modifications(self):
        """Test multiple modifications."""
        bits = Bitstring([False, False, False, False])

        bits[0] = True
        bits[2] = True

        assert bits.to_string() == "1010"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
