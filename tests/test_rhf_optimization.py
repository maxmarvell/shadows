"""Tests for RHF symmetry optimization in excitation amplitude estimation.

This module tests that RHF systems automatically return only unique excitations
(alpha only for singles, alpha-alpha + alpha-beta for doubles) which exploits
spin symmetry to reduce shadow measurements.
"""

import pytest
import numpy as np
from pyscf import gto, scf
from shadow_ci.hamiltonian import MolecularHamiltonian
from shadow_ci.utils import SingleAmplitudes, DoubleAmplitudes


class TestRHFSymmetryOptimization:
    """Test that RHF symmetry optimization produces correct amplitudes."""

    @pytest.fixture
    def h2_hamiltonian(self):
        """Create a simple H2 Hamiltonian for testing."""
        mol = gto.Mole()
        mol.build(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mf.verbose = 0
        return MolecularHamiltonian.from_pyscf(mf)

    @pytest.fixture
    def beh2_hamiltonian(self):
        """Create a BeH2 Hamiltonian with more electrons for testing doubles."""
        mol = gto.Mole()
        mol.build(atom="Be 0 0 0; H 0 0 1.3; H 0 0 -1.3", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mf.verbose = 0
        return MolecularHamiltonian.from_pyscf(mf)

    def test_single_excitations_reduction(self, h2_hamiltonian):
        """Test that RHF automatically returns only alpha singles (50% reduction)."""
        singles = h2_hamiltonian.get_single_excitations()

        nocc = h2_hamiltonian.nocc
        nvirt = h2_hamiltonian.nvirt
        expected_count = nocc * nvirt  # Only alpha excitations

        assert len(singles) == expected_count

        # All excitations should be alpha for RHF
        assert all(ex.spin == 'alpha' for ex in singles)

    def test_double_excitations_reduction(self, beh2_hamiltonian):
        """Test that RHF automatically returns only alpha-alpha and alpha-beta doubles."""
        doubles = beh2_hamiltonian.get_double_excitations()

        # Count spin cases
        n_aa = sum(1 for ex in doubles if ex.spin_case == 'alpha-alpha')
        n_bb = sum(1 for ex in doubles if ex.spin_case == 'beta-beta')
        n_ab = sum(1 for ex in doubles if ex.spin_case == 'alpha-beta')

        nocc = beh2_hamiltonian.nocc
        nvirt = beh2_hamiltonian.nvirt

        # RHF should have alpha-alpha and alpha-beta (reduced sets), but no beta-beta
        # Alpha-alpha: unique pairs with i<j, a<b
        expected_aa = (nocc * (nocc - 1) // 2) * (nvirt * (nvirt - 1) // 2)

        # Alpha-beta: reduced set exploiting RHF symmetry t^{ab,αβ}_{ij} = t^{ba,βα}_{ji}
        # This gives roughly half: diagonal (i=j,a=b) plus upper triangle
        # Formula: sum over i,a of (number of j>=i times number of b>=a)
        expected_ab = sum(
            (nocc - i) * (nvirt - a)
            for i in range(nocc)
            for a in range(nvirt)
        )

        assert n_aa == expected_aa, f"Wrong number of alpha-alpha excitations: {n_aa} != {expected_aa}"
        assert n_ab == expected_ab, f"Wrong number of alpha-beta excitations: {n_ab} != {expected_ab}"
        assert n_bb == 0, "RHF should not return beta-beta excitations"

    def test_single_amplitudes_construction(self, h2_hamiltonian):
        """Test that SingleAmplitudes can be constructed from RHF unique excitations."""
        singles = h2_hamiltonian.get_single_excitations()

        # Create random coefficients for the unique (alpha) excitations
        n_singles = len(singles)
        coeffs = np.random.randn(n_singles) + 1j * np.random.randn(n_singles)

        # Build amplitudes
        t1 = SingleAmplitudes.from_excitation_list(
            coeffs, singles, h2_hamiltonian.nocc, h2_hamiltonian.nvirt, "RHF"
        )

        # Verify shape
        expected_shape = (h2_hamiltonian.nocc, h2_hamiltonian.nvirt)
        assert t1.amplitudes.shape == expected_shape

    def test_double_amplitudes_construction(self, beh2_hamiltonian):
        """Test that DoubleAmplitudes can be constructed from RHF unique excitations."""
        doubles = beh2_hamiltonian.get_double_excitations()

        # Create random coefficients
        n_doubles = len(doubles)
        coeffs = np.random.randn(n_doubles) + 1j * np.random.randn(n_doubles)

        # Build amplitudes
        t2 = DoubleAmplitudes.from_excitation_list(
            coeffs, doubles, beh2_hamiltonian.nocc, beh2_hamiltonian.nvirt, "RHF"
        )

        # Verify shape
        nocc = beh2_hamiltonian.nocc
        nvirt = beh2_hamiltonian.nvirt
        expected_shape = (nocc, nocc, nvirt, nvirt)
        assert t2.amplitudes.shape == expected_shape

    def test_rhf_excitations_are_unique(self, h2_hamiltonian):
        """Test that RHF automatically returns only unique excitations."""
        # For RHF systems, the class should automatically return unique excitations
        singles = h2_hamiltonian.get_single_excitations()
        doubles = h2_hamiltonian.get_double_excitations()

        # All singles should be alpha
        assert all(ex.spin == 'alpha' for ex in singles)

        # Doubles should have no beta-beta
        has_beta_beta = any(ex.spin_case == 'beta-beta' for ex in doubles)
        assert not has_beta_beta, "RHF should not include beta-beta excitations"
