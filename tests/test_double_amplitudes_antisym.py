"""Test antisymmetrization of double amplitudes for RHF.

This test verifies that the DoubleAmplitudes.from_excitation_list() method
correctly reconstructs the spatial amplitude tensor from alpha-alpha and
alpha-beta excitations.
"""

import pytest
import numpy as np
from shadow_ci.utils import DoubleExcitation, DoubleAmplitudes, Bitstring


class TestDoubleAmplitudesAntisymmetrization:
    """Test the antisymmetrization logic for RHF double amplitudes."""

    def test_alpha_alpha_antisymmetry(self):
        """Test that alpha-alpha excitations correctly fill antisymmetric tensor."""
        nocc, nvirt = 2, 2

        # Create a single alpha-alpha excitation: (0,1) -> (0,1)
        exc = DoubleExcitation(
            occ1=0, occ2=1,
            virt1=0, virt2=1,
            spin_case='alpha-alpha',
            bitstring=Bitstring([False, False, True, True], endianess='little'),
            n1=nocc, n2=nocc
        )

        coeff = 1.0 + 0.0j
        t2 = DoubleAmplitudes.from_excitation_list(
            np.array([coeff]), [exc], nocc, nvirt, spin_type="RHF"
        )

        # Check antisymmetry relations
        assert t2[0,1,0,1] == coeff
        assert t2[1,0,0,1] == -coeff
        assert t2[0,1,1,0] == -coeff
        assert t2[1,0,1,0] == coeff

        # All other elements should be zero
        assert t2[0,0,0,0] == 0.0
        assert t2[1,1,0,0] == 0.0

    def test_alpha_beta_no_antisymmetry(self):
        """Test that alpha-beta excitations are added directly without antisymmetrization."""
        nocc, nvirt = 2, 2

        # Create an alpha-beta excitation: (0_α, 0_β) -> (0_α, 0_β)
        exc = DoubleExcitation(
            occ1=0, occ2=0,
            virt1=0, virt2=0,
            spin_case='alpha-beta',
            bitstring=Bitstring([False, False, True, True], endianess='little'),
            n1=nocc, n2=nocc
        )

        coeff = 2.0 + 0.0j
        t2 = DoubleAmplitudes.from_excitation_list(
            np.array([coeff]), [exc], nocc, nvirt, spin_type="RHF"
        )

        # Only the specified element should be filled
        assert t2[0,0,0,0] == coeff

        # Check that no antisymmetric partners are filled
        assert t2[0,0,0,1] == 0.0
        assert t2[0,0,1,0] == 0.0
        assert t2[0,1,0,0] == 0.0
        assert t2[1,0,0,0] == 0.0

    def test_combined_alpha_alpha_and_alpha_beta(self):
        """Test that both spin cases can be combined in the same tensor."""
        nocc, nvirt = 2, 2

        excitations = [
            # Alpha-alpha: (0,1) -> (0,1)
            DoubleExcitation(
                occ1=0, occ2=1, virt1=0, virt2=1,
                spin_case='alpha-alpha',
                bitstring=Bitstring([False, False, True, True], endianess='little'),
                n1=nocc, n2=nocc
            ),
            # Alpha-beta: (0_α, 0_β) -> (0_α, 0_β)
            DoubleExcitation(
                occ1=0, occ2=0, virt1=0, virt2=0,
                spin_case='alpha-beta',
                bitstring=Bitstring([False, False, True, True], endianess='little'),
                n1=nocc, n2=nocc
            ),
        ]

        coeffs = np.array([1.0 + 0.0j, 3.0 + 0.0j])
        t2 = DoubleAmplitudes.from_excitation_list(
            coeffs, excitations, nocc, nvirt, spin_type="RHF"
        )

        # Check alpha-alpha contribution with antisymmetry
        assert t2[0,1,0,1] == 1.0
        assert t2[1,0,0,1] == -1.0

        # Check alpha-beta contribution (no antisymmetry)
        # Note: t2[0,0,0,0] might have contributions from multiple sources
        # In this case, only from alpha-beta
        assert t2[0,0,0,0] == 3.0

    def test_multiple_alpha_alpha_excitations(self):
        """Test that multiple alpha-alpha excitations accumulate correctly."""
        nocc, nvirt = 3, 2

        excitations = [
            DoubleExcitation(
                occ1=0, occ2=1, virt1=0, virt2=1,
                spin_case='alpha-alpha',
                bitstring=Bitstring([False]*nocc + [True]*nvirt, endianess='little'),
                n1=nocc, n2=nocc
            ),
            DoubleExcitation(
                occ1=0, occ2=2, virt1=0, virt2=1,
                spin_case='alpha-alpha',
                bitstring=Bitstring([False]*nocc + [True]*nvirt, endianess='little'),
                n1=nocc, n2=nocc
            ),
        ]

        coeffs = np.array([1.0, 2.0])
        t2 = DoubleAmplitudes.from_excitation_list(
            coeffs, excitations, nocc, nvirt, spin_type="RHF"
        )

        # First excitation
        assert t2[0,1,0,1] == 1.0
        assert t2[1,0,1,0] == 1.0

        # Second excitation
        assert t2[0,2,0,1] == 2.0
        assert t2[2,0,1,0] == 2.0

    def test_alpha_beta_symmetry_expansion(self):
        """Test that alpha-beta excitations correctly expand using RHF symmetry.

        For RHF: t^{ab,αβ}_{ij} = t^{ba,βα}_{ji}
        This means (i_α,j_β)->(a_α,b_β) has same amplitude as (j_α,i_β)->(b_α,a_β)
        """
        nocc, nvirt = 2, 2

        # Create off-diagonal alpha-beta excitation: (0_α, 1_β) -> (0_α, 1_β)
        exc = DoubleExcitation(
            occ1=0, occ2=1,  # i=0, j=1
            virt1=0, virt2=1,  # a=0, b=1
            spin_case='alpha-beta',
            bitstring=Bitstring([False, False, True, True], endianess='little'),
            n1=nocc, n2=nocc
        )

        coeff = 2.5 + 0.0j
        t2 = DoubleAmplitudes.from_excitation_list(
            np.array([coeff]), [exc], nocc, nvirt, spin_type="RHF"
        )

        # Original excitation
        assert t2[0,1,0,1] == coeff

        # Symmetric partner: (j,i,b,a) = (1,0,1,0)
        assert t2[1,0,1,0] == coeff

        # Other elements should be zero
        assert t2[0,0,0,0] == 0.0
        assert t2[1,1,1,1] == 0.0
        assert t2[0,1,1,0] == 0.0
        assert t2[1,0,0,1] == 0.0

    def test_alpha_beta_diagonal_no_duplicate(self):
        """Test that diagonal alpha-beta excitations (i=j, a=b) are not duplicated."""
        nocc, nvirt = 2, 2

        # Diagonal alpha-beta: (0_α, 0_β) -> (1_α, 1_β)
        exc = DoubleExcitation(
            occ1=0, occ2=0,  # i=j=0
            virt1=1, virt2=1,  # a=b=1
            spin_case='alpha-beta',
            bitstring=Bitstring([False, False, True, True], endianess='little'),
            n1=nocc, n2=nocc
        )

        coeff = 3.0 + 0.0j
        t2 = DoubleAmplitudes.from_excitation_list(
            np.array([coeff]), [exc], nocc, nvirt, spin_type="RHF"
        )

        # Diagonal element should only have the coefficient once (no duplication)
        assert t2[0,0,1,1] == coeff

        # Other elements should be zero
        assert t2[0,1,1,1] == 0.0
        assert t2[1,0,1,1] == 0.0
        assert t2[0,0,0,0] == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
