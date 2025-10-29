"""Tests for classical shadow and shadow protocol classes."""

import pytest
import stim
import numpy as np
from qiskit.quantum_info import Statevector
from shadow_ci.shadows import (
    ClassicalSnapshot,
    ClassicalShadow,
    ShadowProtocol,
    CliffordGroup,
)
from shadow_ci.utils import Bitstring


class TestCliffordGroup:
    """Test the CliffordGroup ensemble sampler."""

    def test_initialization(self):
        """Test CliffordGroup can be initialized with dimension."""
        d = 3
        ensemble = CliffordGroup(d)
        assert ensemble.d == 3

    def test_generate_sample_returns_tableau(self):
        """Test that generate_sample returns a stim Tableau."""
        ensemble = CliffordGroup(d=2)
        sample = ensemble.generate_sample()
        assert isinstance(sample, stim.Tableau)

    def test_generate_sample_correct_dimension(self):
        """Test that generated tableau has correct number of qubits."""
        d = 4
        ensemble = CliffordGroup(d)
        sample = ensemble.generate_sample()
        assert len(sample) == d  # Tableau uses len() for number of qubits

    def test_generate_sample_is_random(self):
        """Test that consecutive samples are different (with high probability)."""
        ensemble = CliffordGroup(d=3)
        sample1 = ensemble.generate_sample()
        sample2 = ensemble.generate_sample()
        # With high probability, two random Cliffords should be different
        assert sample1 != sample2


class TestClassicalSnapshot:
    """Test the ClassicalSnapshot dataclass."""

    def test_creation_with_tableau(self):
        """Test creating snapshot with Tableau unitary."""
        measurement = Bitstring([True, False, True])
        unitary = stim.Tableau.random(3)
        snapshot = ClassicalSnapshot(
            measurement=measurement,
            unitary=unitary,
            sampler="clifford"
        )
        assert snapshot.measurement == measurement
        assert snapshot.unitary == unitary
        assert snapshot.sampler == "clifford"

    def test_creation_with_pauli_string(self):
        """Test creating snapshot with PauliString unitary."""
        measurement = Bitstring([False, False])
        unitary = stim.PauliString("XZ")
        snapshot = ClassicalSnapshot(
            measurement=measurement,
            unitary=unitary,
            sampler="pauli"
        )
        assert isinstance(snapshot.unitary, stim.PauliString)

    def test_creation_with_numpy_array(self):
        """Test creating snapshot with numpy array unitary."""
        measurement = Bitstring([True])
        unitary = np.eye(2, dtype=complex)
        snapshot = ClassicalSnapshot(
            measurement=measurement,
            unitary=unitary,
            sampler="custom"
        )
        assert isinstance(snapshot.unitary, np.ndarray)


class TestClassicalShadow:
    """Test the ClassicalShadow class."""

    def test_initialization(self):
        """Test ClassicalShadow initialization."""
        snapshots = [
            ClassicalSnapshot(
                measurement=Bitstring([False, False]),
                unitary=stim.Tableau.random(2),
                sampler="clifford"
            )
        ]
        shadow = ClassicalShadow(shadow=snapshots, n_qubits=2)
        assert shadow.n_qubits == 2
        assert shadow.N == 1

    def test_N_property(self):
        """Test that N property returns correct number of snapshots."""
        snapshots = [
            ClassicalSnapshot(
                measurement=Bitstring([False]),
                unitary=stim.Tableau.random(1),
                sampler="clifford"
            )
            for _ in range(10)
        ]
        shadow = ClassicalShadow(shadow=snapshots, n_qubits=1)
        assert shadow.N == 10

    def test_empty_shadow(self):
        """Test shadow with no snapshots."""
        shadow = ClassicalShadow(shadow=[], n_qubits=2)
        assert shadow.N == 0

    def test_compute_overlap_raises_for_non_tableau(self):
        """Test that compute_overlap raises error for non-Tableau unitaries."""
        snapshots = [
            ClassicalSnapshot(
                measurement=Bitstring([False]),
                unitary=np.eye(2),  # Not a Tableau
                sampler="custom"
            )
        ]
        shadow = ClassicalShadow(shadow=snapshots, n_qubits=1)

        a = Bitstring([False])
        b = Bitstring([False])

        with pytest.raises(NotImplementedError, match="Only Clifford/Tableau"):
            shadow.compute_overlap(a, b)

class TestShadowProtocol:
    """Test the ShadowProtocol class with uniform random Clifford sampling."""

    def test_initialization_with_statevector(self):
        """Test ShadowProtocol initialization with a Statevector."""
        # Create |00⟩ state
        state = Statevector.from_label('00')
        protocol = ShadowProtocol(state, n_samples=20, n_k_estimators=4)

        assert protocol.n_k_estimators == 4
        assert len(protocol.k_estimators) == 4
        assert protocol.classical_shadow.n_qubits == 2
        assert protocol.classical_shadow.N == 20

    def test_initialization_validates_equal_split(self):
        """Test that error is raised if samples can't be split equally."""
        state = Statevector.from_label('0')

        with pytest.raises(ValueError, match="equally sized"):
            ShadowProtocol(state, n_samples=10, n_k_estimators=3)

    def test_generates_random_cliffords(self):
        """Test that protocol generates random Clifford unitaries."""
        state = Statevector.from_label('0')
        protocol = ShadowProtocol(state, n_samples=10, n_k_estimators=2)

        # Check that snapshots have Tableau unitaries
        for snapshot in protocol.classical_shadow.shadow:
            assert isinstance(snapshot.unitary, stim.
                              Tableau)
            assert snapshot.sampler == "clifford"

    def test_single_qubit_computational_basis_complete(self):
        """Test amplitude estimation for |Ψ⟩ = |1⟩ (orthogonal to vacuum).

        For |Ψ⟩ = |1⟩:
        - |τ⟩ = (|0⟩ + |1⟩)/√2 = |+⟩ (properly normalized)
        - ⟨0|Ψ⟩ = ⟨0|1⟩ = 0
        - ⟨1|Ψ⟩ = ⟨1|1⟩ = 1
        """
        np.random.seed(42)
        # Create |Ψ⟩ = |1⟩ (orthogonal to vacuum)
        psi = Statevector.from_label('1')
        # Create |τ⟩ = (|0⟩ + |Ψ⟩)/√2
        vacuum = np.zeros(2, dtype=complex); vacuum[0] = 1.0
        tau = (vacuum + psi.data) / np.sqrt(2)
        tau_state = Statevector(tau)

        protocol = ShadowProtocol(tau_state, n_samples=1000, n_k_estimators=100)

        # Vacuum bitstring (always second argument)
        vac = Bitstring([False])

        # Compute amplitudes ⟨a|Ψ⟩
        amp_0 = protocol.overlap(Bitstring([False]), vac)  # ⟨0|1⟩
        amp_1 = protocol.overlap(Bitstring([True]), vac)   # ⟨1|1⟩

        # Check normalization: amplitudes should be ≤ 1
        assert np.abs(amp_0) <= 1.0, f"|⟨0|1⟩| = {np.abs(amp_0)} > 1"
        assert np.abs(amp_1) <= 1.0, f"|⟨1|1⟩| = {np.abs(amp_1)} > 1"

        # Check expected values: ⟨0|1⟩ = 0, ⟨1|1⟩ = 1
        assert np.abs(amp_0) < 0.15, f"⟨0|1⟩ = {amp_0}, expected ≈ 0"
        assert np.abs(amp_1 - 1.0) < 0.15, f"⟨1|1⟩ = {amp_1}, expected ≈ 1"

    def test_two_qubit_state_11(self):
        """Test amplitude estimation for 2-qubit |Ψ⟩ = |11⟩ (orthogonal to vacuum).

        For |Ψ⟩ = |11⟩:
        - ⟨00|11⟩ = 0
        - ⟨11|11⟩ = 1
        - ⟨01|11⟩ = 0, ⟨10|11⟩ = 0
        """
        np.random.seed(45)
        psi = Statevector.from_label('11')
        vacuum = np.zeros(4, dtype=complex); vacuum[0] = 1.0
        tau = (vacuum + psi.data) / np.sqrt(2)
        tau_state = Statevector(tau)

        protocol = ShadowProtocol(tau_state, n_samples=800, n_k_estimators=10)

        vac = Bitstring([False, False])
        amp_00 = protocol.overlap(Bitstring([False, False]), vac)
        amp_01 = protocol.overlap(Bitstring([False, True]), vac)
        amp_10 = protocol.overlap(Bitstring([True, False]), vac)
        amp_11 = protocol.overlap(Bitstring([True, True]), vac)

        assert np.abs(amp_11) <= 1.0
        assert np.abs(amp_00) < 0.15, f"⟨00|11⟩ = {amp_00}, expected ≈ 0"
        assert np.abs(amp_01) < 0.15, f"⟨01|11⟩ = {amp_01}, expected ≈ 0"
        assert np.abs(amp_10) < 0.15, f"⟨10|11⟩ = {amp_10}, expected ≈ 0"
        assert np.abs(amp_11 - 1.0) < 0.15, f"⟨11|11⟩ = {amp_11}, expected ≈ 1"

    def test_bell_state_all_overlaps(self):
        """Test amplitude estimation for Bell state |Ψ⟩ = |Φ+⟩ = (|00⟩ + |11⟩)/√2.

        For |Ψ⟩ = |Φ+⟩:
        - ⟨00|Φ+⟩ = 1/√2 ≈ 0.707
        - ⟨11|Φ+⟩ = 1/√2 ≈ 0.707
        - ⟨01|Φ+⟩ = 0
        - ⟨10|Φ+⟩ = 0
        """
        np.random.seed(46)
        # Create Bell state |Φ+⟩
        from qiskit import QuantumCircuit
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        psi = Statevector.from_label('00').evolve(qc)

        vacuum = np.zeros(4, dtype=complex); vacuum[0] = 1.0
        tau = (vacuum + psi.data) / np.sqrt(2)
        tau_state = Statevector(tau)

        protocol = ShadowProtocol(tau_state, n_samples=1000, n_k_estimators=10)

        vac = Bitstring([False, False])
        amp_00 = protocol.overlap(Bitstring([False, False]), vac)
        amp_11 = protocol.overlap(Bitstring([True, True]), vac)
        amp_01 = protocol.overlap(Bitstring([False, True]), vac)
        amp_10 = protocol.overlap(Bitstring([True, False]), vac)

        # All should be ≤ 1
        assert np.abs(amp_00) <= 1.0
        assert np.abs(amp_11) <= 1.0
        assert np.abs(amp_01) <= 1.0
        assert np.abs(amp_10) <= 1.0

        # Amplitudes should be 1/√2 or 0
        expected = 1.0 / np.sqrt(2)
        assert np.abs(amp_00 - expected) < 0.15, f"⟨00|Φ+⟩ = {amp_00}, expected ≈ {expected}"
        assert np.abs(amp_11 - expected) < 0.15, f"⟨11|Φ+⟩ = {amp_11}, expected ≈ {expected}"
        assert np.abs(amp_01) < 0.15, f"⟨01|Φ+⟩ = {amp_01}, expected ≈ 0"
        assert np.abs(amp_10) < 0.15, f"⟨10|Φ+⟩ = {amp_10}, expected ≈ 0"

    def test_hermiticity_of_amplitudes(self):
        """Test that amplitudes are consistent - we only compute ⟨a|Ψ⟩ with vacuum as second arg.

        Note: The protocol computes ⟨a|Ψ⟩ = overlap(a, vacuum), not general ⟨a|ρ|b⟩.
        For a pure state |Ψ⟩, we have ⟨a|Ψ⟩ and ⟨b|Ψ⟩ which aren't necessarily related
        by Hermitian conjugation (that would be for density matrix elements).

        This test just verifies the amplitudes are bounded and consistent.
        """
        np.random.seed(47)
        psi = Statevector.from_label('+')
        vacuum = np.zeros(2, dtype=complex); vacuum[0] = 1.0
        tau = (vacuum + psi.data) / np.sqrt(2)
        tau_state = Statevector(tau)

        protocol = ShadowProtocol(tau_state, n_samples=500, n_k_estimators=10)

        vac = Bitstring([False])
        amp_0 = protocol.overlap(Bitstring([False]), vac)
        amp_1 = protocol.overlap(Bitstring([True]), vac)

        # Both amplitudes should be ≤ 1
        assert np.abs(amp_0) <= 1.0
        assert np.abs(amp_1) <= 1.0

        # For |+⟩, both should be approximately equal (both ≈ 1/√2)
        assert np.abs(np.abs(amp_0) - np.abs(amp_1)) < 0.2, \
            f"|⟨0|+⟩| = {np.abs(amp_0)}, |⟨1|+⟩| = {np.abs(amp_1)}"

    def test_normalization_of_amplitudes(self):
        """Test that |Ψ⟩ is normalized: Σᵢ |⟨i|Ψ⟩|² ≈ 1.

        For a normalized state |Ψ⟩, we should have Σᵢ |⟨i|Ψ⟩|² = 1.
        """
        np.random.seed(48)

        # Test for |Ψ⟩ = |1⟩
        psi = Statevector.from_label('1')
        vacuum = np.zeros(2, dtype=complex); vacuum[0] = 1.0
        tau = (vacuum + psi.data) / np.sqrt(2)
        tau_state = Statevector(tau)

        protocol = ShadowProtocol(tau_state, n_samples=500, n_k_estimators=10)

        vac = Bitstring([False])
        amp_0 = protocol.overlap(Bitstring([False]), vac)
        amp_1 = protocol.overlap(Bitstring([True]), vac)

        norm_sq = np.abs(amp_0)**2 + np.abs(amp_1)**2
        assert np.abs(norm_sq - 1.0) < 0.3, f"Σ|⟨i|1⟩|² = {norm_sq}, expected ≈ 1"

        # Test for |Ψ⟩ = |+⟩
        psi_plus = Statevector.from_label('+')
        tau_plus = (vacuum + psi_plus.data) / np.sqrt(2)
        tau_state_plus = Statevector(tau_plus)

        protocol_plus = ShadowProtocol(tau_state_plus, n_samples=500, n_k_estimators=10)

        amp_0_plus = protocol_plus.overlap(Bitstring([False]), vac)
        amp_1_plus = protocol_plus.overlap(Bitstring([True]), vac)

        norm_sq_plus = np.abs(amp_0_plus)**2 + np.abs(amp_1_plus)**2
        assert np.abs(norm_sq_plus - 1.0) < 0.3, f"Σ|⟨i|+⟩|² = {norm_sq_plus}, expected ≈ 1"

    def test_amplitude_magnitudes_bounded(self):
        """Test that amplitude magnitudes are properly bounded."""
        np.random.seed(49)
        psi = Statevector.from_label('+')
        vacuum = np.zeros(2, dtype=complex); vacuum[0] = 1.0
        tau = (vacuum + psi.data) / np.sqrt(2)
        tau_state = Statevector(tau)

        protocol = ShadowProtocol(tau_state, n_samples=500, n_k_estimators=10)

        vac = Bitstring([False])
        amp_0 = protocol.overlap(Bitstring([False]), vac)
        amp_1 = protocol.overlap(Bitstring([True]), vac)

        # Amplitude magnitudes should be ≤ 1
        assert np.abs(amp_0) <= 1.0, f"|⟨0|+⟩| = {np.abs(amp_0)} > 1"
        assert np.abs(amp_1) <= 1.0, f"|⟨1|+⟩| = {np.abs(amp_1)} > 1"

    def test_ensemble_type_validation(self):
        """Test that invalid ensemble types raise errors."""
        state = Statevector.from_label('0')

        with pytest.raises(ValueError, match="Unexpected ensemble"):
            ShadowProtocol(state, n_samples=10, n_k_estimators=2, ensemble_type="invalid")

    def test_pauli_ensemble_not_implemented(self):
        """Test that Pauli ensemble raises NotImplementedError."""
        state = Statevector.from_label('0')

        with pytest.raises(NotImplementedError, match="Not yet implemented"):
            ShadowProtocol(state, n_samples=10, n_k_estimators=2, ensemble_type="pauli")


class TestShadowProtocolConvergence:
    """Test convergence properties with increasing number of samples."""

    def test_convergence_with_more_samples(self):
        """Test that amplitude estimates improve with more samples."""
        np.random.seed(50)
        psi = Statevector.from_label('1')
        vacuum = np.zeros(2, dtype=complex); vacuum[0] = 1.0
        tau = (vacuum + psi.data) / np.sqrt(2)
        tau_state = Statevector(tau)

        vac = Bitstring([False])

        # Few samples
        protocol_small = ShadowProtocol(tau_state, n_samples=100, n_k_estimators=5)
        amp_small = protocol_small.overlap(Bitstring([True]), vac)

        # Many samples
        protocol_large = ShadowProtocol(tau_state, n_samples=1000, n_k_estimators=10)
        amp_large = protocol_large.overlap(Bitstring([True]), vac)

        # Both should be close to 1 (⟨1|1⟩ = 1)
        error_small = np.abs(amp_small - 1.0)
        error_large = np.abs(amp_large - 1.0)

        # With high probability, more samples gives better estimate
        # (allowing some slack for statistical variance)
        assert error_large <= error_small + 0.2, \
            f"More samples didn't improve: error_small={error_small}, error_large={error_large}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
