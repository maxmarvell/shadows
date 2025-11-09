"""Benchmark suite for ground state estimation components.

This benchmark suite profiles the major components of the shadow CI workflow:
1. Molecular Hamiltonian construction and excitation generation
2. VQE solver (trial state preparation)
3. Shadow protocol: sample collection and overlap estimation
4. Amplitude tensor construction
5. Energy evaluation

Run with: pytest benchmarks/ --benchmark-only
"""

import pytest
import numpy as np
from pyscf import gto, scf

from shadow_ci.hamiltonian import MolecularHamiltonian
from shadow_ci.shadows import ShadowProtocol, CliffordGroup
from shadow_ci.utils import SingleAmplitudes, DoubleAmplitudes


@pytest.fixture(scope="module")
def h2_molecule():
    """Small H2 molecule for quick benchmarks."""
    mol = gto.Mole()
    mol.build(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
    return mol


@pytest.fixture(scope="module")
def beh2_molecule():
    """Medium-sized BeH2 molecule."""
    mol = gto.Mole()
    mol.build(atom="Be 0 0 0; H 0 0 1.3; H 0 0 -1.3", basis="sto-3g", verbose=0)
    return mol


@pytest.fixture(scope="module")
def h2o_molecule():
    """Larger H2O molecule for stress testing."""
    mol = gto.Mole()
    mol.build(atom="O 0 0 0; H 0.757 0.586 0; H -0.757 0.586 0", basis="sto-3g", verbose=0)
    return mol


@pytest.fixture(scope="module")
def h2_hamiltonian(h2_molecule):
    """H2 Hamiltonian fixture."""
    mf = scf.RHF(h2_molecule)
    mf.verbose = 0
    return MolecularHamiltonian.from_pyscf(mf)


@pytest.fixture(scope="module")
def beh2_hamiltonian(beh2_molecule):
    """BeH2 Hamiltonian fixture."""
    mf = scf.RHF(beh2_molecule)
    mf.verbose = 0
    return MolecularHamiltonian.from_pyscf(mf)


@pytest.fixture(scope="module")
def h2o_hamiltonian(h2o_molecule):
    """H2O Hamiltonian fixture."""
    mf = scf.RHF(h2o_molecule)
    mf.verbose = 0
    return MolecularHamiltonian.from_pyscf(mf)


class TestHamiltonianConstruction:
    """Benchmark Hamiltonian construction and excitation generation."""

    def test_hamiltonian_from_pyscf_h2(self, benchmark, h2_molecule):
        """Benchmark Hamiltonian construction from PySCF for H2."""
        def setup():
            mf = scf.RHF(h2_molecule)
            mf.verbose = 0
            return (mf,), {}

        benchmark.pedantic(MolecularHamiltonian.from_pyscf, setup=setup, rounds=10)

    def test_hamiltonian_from_pyscf_beh2(self, benchmark, beh2_molecule):
        """Benchmark Hamiltonian construction from PySCF for BeH2."""
        def setup():
            mf = scf.RHF(beh2_molecule)
            mf.verbose = 0
            return (mf,), {}

        benchmark.pedantic(MolecularHamiltonian.from_pyscf, setup=setup, rounds=10)

    def test_single_excitations_h2(self, benchmark, h2_hamiltonian):
        """Benchmark single excitation generation for H2."""
        benchmark(h2_hamiltonian.get_single_excitations)

    def test_single_excitations_beh2(self, benchmark, beh2_hamiltonian):
        """Benchmark single excitation generation for BeH2."""
        benchmark(beh2_hamiltonian.get_single_excitations)

    def test_double_excitations_h2(self, benchmark, h2_hamiltonian):
        """Benchmark double excitation generation for H2."""
        benchmark(h2_hamiltonian.get_double_excitations)

    def test_double_excitations_beh2(self, benchmark, beh2_hamiltonian):
        """Benchmark double excitation generation for BeH2."""
        benchmark(beh2_hamiltonian.get_double_excitations)

    def test_double_excitations_h2o(self, benchmark, h2o_hamiltonian):
        """Benchmark double excitation generation for H2O (larger system)."""
        benchmark(h2o_hamiltonian.get_double_excitations)


class TestTrialStatePreparation:
    """Benchmark trial state preparation."""

    def test_hf_state_preparation_h2(self, benchmark, h2_hamiltonian):
        """Benchmark HF state preparation for H2."""
        benchmark(h2_hamiltonian.get_hf_state)

    def test_hf_state_preparation_beh2(self, benchmark, beh2_hamiltonian):
        """Benchmark HF state preparation for BeH2."""
        benchmark(beh2_hamiltonian.get_hf_state)

    def test_hf_bitstring_h2(self, benchmark, h2_hamiltonian):
        """Benchmark HF bitstring generation for H2."""
        benchmark(h2_hamiltonian.get_hf_bitstring)


class TestShadowProtocol:
    """Benchmark shadow protocol components."""

    @pytest.fixture
    def h2_trial_state(self, h2_hamiltonian):
        """Get H2 trial state from HF."""
        # For benchmarking, just use the HF state (VQE is too slow for many tests)
        return h2_hamiltonian.get_hf_state()

    @pytest.fixture
    def beh2_trial_state(self, beh2_hamiltonian):
        """Get BeH2 trial state from HF."""
        # For benchmarking, just use the HF state (VQE is too slow for many tests)
        return beh2_hamiltonian.get_hf_state()

    def test_protocol_initialization_h2(self, benchmark, h2_trial_state):
        """Benchmark ShadowProtocol initialization for H2."""
        benchmark(ShadowProtocol, h2_trial_state)

    def test_collect_samples_h2_tiny(self, benchmark, h2_trial_state):
        """Benchmark sample collection for H2 (n=5, k=1) - minimal."""
        protocol = ShadowProtocol(h2_trial_state)
        benchmark.pedantic(protocol.collect_samples, args=(5, 1), kwargs={'prediction': 'overlap'}, rounds=3, iterations=1)

    def test_collect_samples_h2_small(self, benchmark, h2_trial_state):
        """Benchmark sample collection for H2 (n=10, k=2)."""
        protocol = ShadowProtocol(h2_trial_state)
        benchmark.pedantic(protocol.collect_samples, args=(10, 2), kwargs={'prediction': 'overlap'}, rounds=3, iterations=1)

    def test_collect_samples_h2_medium(self, benchmark, h2_trial_state):
        """Benchmark sample collection for H2 (n=20, k=2)."""
        protocol = ShadowProtocol(h2_trial_state)
        benchmark.pedantic(protocol.collect_samples, args=(20, 2), kwargs={'prediction': 'overlap'}, rounds=2, iterations=1)

    def test_estimate_overlap_single_h2(self, benchmark, h2_trial_state, h2_hamiltonian):
        """Benchmark single overlap estimation for H2."""
        protocol = ShadowProtocol(h2_trial_state)
        protocol.collect_samples(20, 2, prediction='overlap')
        target = h2_hamiltonian.get_hf_bitstring()

        benchmark(protocol.estimate_overlap, target)

    def test_estimate_overlap_many_h2(self, benchmark, h2_trial_state, h2_hamiltonian):
        """Benchmark multiple overlap estimations for H2 (all singles)."""
        protocol = ShadowProtocol(h2_trial_state)
        protocol.collect_samples(20, 2, prediction='overlap')
        excitations = h2_hamiltonian.get_single_excitations()

        def estimate_all():
            return [protocol.estimate_overlap(ex.bitstring) for ex in excitations]

        benchmark(estimate_all)


class TestAmplitudeTensorConstruction:
    """Benchmark amplitude tensor construction from excitation lists."""

    def test_single_amplitudes_h2(self, benchmark, h2_hamiltonian):
        """Benchmark SingleAmplitudes construction for H2."""
        excitations = h2_hamiltonian.get_single_excitations()
        coeffs = np.random.randn(len(excitations)) + 1j * np.random.randn(len(excitations))

        benchmark(
            SingleAmplitudes.from_excitation_list,
            coeffs, excitations, h2_hamiltonian.nocc, h2_hamiltonian.nvirt, "RHF"
        )

    def test_single_amplitudes_beh2(self, benchmark, beh2_hamiltonian):
        """Benchmark SingleAmplitudes construction for BeH2."""
        excitations = beh2_hamiltonian.get_single_excitations()
        coeffs = np.random.randn(len(excitations)) + 1j * np.random.randn(len(excitations))

        benchmark(
            SingleAmplitudes.from_excitation_list,
            coeffs, excitations, beh2_hamiltonian.nocc, beh2_hamiltonian.nvirt, "RHF"
        )

    def test_double_amplitudes_h2(self, benchmark, h2_hamiltonian):
        """Benchmark DoubleAmplitudes construction for H2."""
        excitations = h2_hamiltonian.get_double_excitations()
        coeffs = np.random.randn(len(excitations)) + 1j * np.random.randn(len(excitations))

        benchmark(
            DoubleAmplitudes.from_excitation_list,
            coeffs, excitations, h2_hamiltonian.nocc, h2_hamiltonian.nvirt, "RHF"
        )

    def test_double_amplitudes_beh2(self, benchmark, beh2_hamiltonian):
        """Benchmark DoubleAmplitudes construction for BeH2."""
        excitations = beh2_hamiltonian.get_double_excitations()
        coeffs = np.random.randn(len(excitations)) + 1j * np.random.randn(len(excitations))

        benchmark(
            DoubleAmplitudes.from_excitation_list,
            coeffs, excitations, beh2_hamiltonian.nocc, beh2_hamiltonian.nvirt, "RHF"
        )

    def test_double_amplitudes_h2o(self, benchmark, h2o_hamiltonian):
        """Benchmark DoubleAmplitudes construction for H2O (larger system)."""
        excitations = h2o_hamiltonian.get_double_excitations()
        coeffs = np.random.randn(len(excitations)) + 1j * np.random.randn(len(excitations))

        benchmark(
            DoubleAmplitudes.from_excitation_list,
            coeffs, excitations, h2o_hamiltonian.nocc, h2o_hamiltonian.nvirt, "RHF"
        )


class TestEnergyEvaluation:
    """Benchmark energy evaluation from amplitudes."""

    def test_correlation_energy_h2(self, benchmark, h2_hamiltonian):
        """Benchmark correlation energy computation for H2."""
        singles = h2_hamiltonian.get_single_excitations()
        doubles = h2_hamiltonian.get_double_excitations()

        c1_coeffs = np.random.randn(len(singles)) + 1j * np.random.randn(len(singles))
        c2_coeffs = np.random.randn(len(doubles)) + 1j * np.random.randn(len(doubles))

        c0 = 1.0 + 0.0j
        c1 = SingleAmplitudes.from_excitation_list(
            c1_coeffs, singles, h2_hamiltonian.nocc, h2_hamiltonian.nvirt, "RHF"
        )
        c2 = DoubleAmplitudes.from_excitation_list(
            c2_coeffs, doubles, h2_hamiltonian.nocc, h2_hamiltonian.nvirt, "RHF"
        )

        benchmark(h2_hamiltonian.compute_correlation_energy, c0, c1, c2)

    def test_correlation_energy_beh2(self, benchmark, beh2_hamiltonian):
        """Benchmark correlation energy computation for BeH2."""
        singles = beh2_hamiltonian.get_single_excitations()
        doubles = beh2_hamiltonian.get_double_excitations()

        c1_coeffs = np.random.randn(len(singles)) + 1j * np.random.randn(len(singles))
        c2_coeffs = np.random.randn(len(doubles)) + 1j * np.random.randn(len(doubles))

        c0 = 1.0 + 0.0j
        c1 = SingleAmplitudes.from_excitation_list(
            c1_coeffs, singles, beh2_hamiltonian.nocc, beh2_hamiltonian.nvirt, "RHF"
        )
        c2 = DoubleAmplitudes.from_excitation_list(
            c2_coeffs, doubles, beh2_hamiltonian.nocc, beh2_hamiltonian.nvirt, "RHF"
        )

        benchmark(beh2_hamiltonian.compute_correlation_energy, c0, c1, c2)


class TestEstimatorComponents:
    """Benchmark GroundStateEstimator component methods."""

    @pytest.fixture
    def h2_protocol_with_samples(self, h2_hamiltonian):
        """H2 protocol with collected samples."""
        trial = h2_hamiltonian.get_hf_state()
        protocol = ShadowProtocol(trial)
        protocol.collect_samples(20, 2, prediction='overlap')
        return protocol, h2_hamiltonian

    def test_estimate_c0_h2(self, benchmark, h2_protocol_with_samples):
        """Benchmark c0 (HF overlap) estimation."""
        protocol, hamiltonian = h2_protocol_with_samples
        psi0 = hamiltonian.get_hf_bitstring()

        benchmark(protocol.estimate_overlap, psi0)

    def test_estimate_singles_h2(self, benchmark, h2_protocol_with_samples):
        """Benchmark all single excitation overlaps."""
        protocol, hamiltonian = h2_protocol_with_samples
        excitations = hamiltonian.get_single_excitations()

        def estimate_all_singles():
            coeffs = np.empty(len(excitations), dtype=complex)
            for i, ex in enumerate(excitations):
                coeffs[i] = protocol.estimate_overlap(ex.bitstring)
            return SingleAmplitudes.from_excitation_list(
                coeffs, excitations, hamiltonian.nocc, hamiltonian.nvirt, "RHF"
            )

        benchmark(estimate_all_singles)

    def test_estimate_doubles_h2(self, benchmark, h2_protocol_with_samples):
        """Benchmark all double excitation overlaps."""
        protocol, hamiltonian = h2_protocol_with_samples
        excitations = hamiltonian.get_double_excitations()

        def estimate_all_doubles():
            coeffs = np.empty(len(excitations), dtype=complex)
            for i, ex in enumerate(excitations):
                coeffs[i] = protocol.estimate_overlap(ex.bitstring)
            return DoubleAmplitudes.from_excitation_list(
                coeffs, excitations, hamiltonian.nocc, hamiltonian.nvirt, "RHF"
            )

        benchmark(estimate_all_doubles)


class TestCliffordOperations:
    """Benchmark low-level Clifford operations."""

    def test_clifford_generation_4qubits(self, benchmark):
        """Benchmark random Clifford generation for 4 qubits."""
        ensemble = CliffordGroup(4)
        benchmark(ensemble.generate_sample)

    def test_clifford_generation_8qubits(self, benchmark):
        """Benchmark random Clifford generation for 8 qubits."""
        ensemble = CliffordGroup(8)
        benchmark(ensemble.generate_sample)

    def test_clifford_generation_12qubits(self, benchmark):
        """Benchmark random Clifford generation for 12 qubits."""
        ensemble = CliffordGroup(12)
        benchmark(ensemble.generate_sample)

    def test_clifford_inverse_4qubits(self, benchmark):
        """Benchmark Clifford inverse operation for 4 qubits."""
        import stim
        tab = stim.Tableau.random(4)
        benchmark(tab.inverse)

    def test_clifford_inverse_8qubits(self, benchmark):
        """Benchmark Clifford inverse operation for 8 qubits."""
        import stim
        tab = stim.Tableau.random(8)
        benchmark(tab.inverse)

    def test_stabilizer_canonicalization_4qubits(self, benchmark):
        """Benchmark stabilizer canonicalization for 4 qubits."""
        from shadow_ci.utils import canonicalize, Bitstring

        bitstring = Bitstring.random(4)
        stabilizers = bitstring.to_stabilizers()

        benchmark(canonicalize, stabilizers)

    def test_stabilizer_canonicalization_8qubits(self, benchmark):
        """Benchmark stabilizer canonicalization for 8 qubits."""
        from shadow_ci.utils import canonicalize, Bitstring

        bitstring = Bitstring.random(8)
        stabilizers = bitstring.to_stabilizers()

        benchmark(canonicalize, stabilizers)


if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only", "-v"])
