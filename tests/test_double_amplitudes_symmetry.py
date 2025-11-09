"""Test symmetry properties of double excitation amplitudes for RHF systems.

This test extracts exact FCI amplitudes from PySCF and examines their symmetry
properties in spatial orbital representation.
"""

import numpy as np
from pyscf import gto, scf, fci, cc
from shadow_ci.utils import make_hydrogen_chain


def test_rhf_doubles_symmetry():
    """Extract exact FCI/CCSD amplitudes from PySCF and examine symmetry.

    For RHF systems, we expect certain symmetry relations in the t2 amplitudes.
    This test extracts the exact amplitudes to understand the symmetry structure.
    """
    # Build a H6 chain
    mol = gto.Mole()
    atom = make_hydrogen_chain(4, 0.74)
    mol.build(atom=atom, basis='sto-3g', verbose=0)

    # Run RHF
    mf = scf.RHF(mol)
    mf.kernel()

    norb = mf.mo_coeff.shape[1]
    nelec = mol.nelectron
    nocc = nelec // 2  # For RHF
    nvirt = norb - nocc

    print("="*70)
    print("RHF Double Excitation Amplitude Symmetry Analysis")
    print("="*70)
    print(f"System: H6 chain (bond length = 0.74 Å)")
    print(f"Basis: sto-3g")
    print(f"Spatial orbitals: {norb}")
    print(f"Occupied orbitals: {nocc}")
    print(f"Virtual orbitals: {nvirt}")
    print(f"HF Energy: {mf.e_tot:.8f} Ha")

    # Method 1: Use CCSD to get t2 amplitudes directly
    print("\n" + "="*70)
    print("Extracting t2 amplitudes from CCSD")
    print("="*70)

    mycc = cc.CCSD(mf)
    mycc.kernel()

    # Get t2 amplitudes from CCSD
    # PySCF CCSD stores t2 in the format t2[i,j,a,b]
    # where i,j are occupied indices and a,b are virtual indices
    t2 = mycc.t2

    print(f"\nCCSD converged: {mycc.converged}")
    print(f"CCSD Energy: {mycc.e_tot:.8f} Ha")
    print(f"t2 shape: {t2.shape}")
    print(f"Expected shape: ({nocc}, {nocc}, {nvirt}, {nvirt})")

    # Print the full t2 tensor
    print("\n" + "="*70)
    print("Full t2 amplitude tensor:")
    print("="*70)
    print("\nFormat: t2[i,j,a,b] where i,j are occupied, a,b are virtual")
    print("-"*70)

    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvirt):
                for b in range(nvirt):
                    val = t2[i,j,a,b]
                    if abs(val) > 1e-8:  # Only print non-zero values
                        print(f"t2[{i},{j},{a},{b}] = {val:>12.8f}")

    # Analyze symmetry properties
    print("\n" + "="*70)
    print("Analyzing Symmetry Properties")
    print("="*70)

    # Check antisymmetry: t2[i,j,a,b] = -t2[j,i,a,b]
    print("\n1. Antisymmetry w.r.t. occupied indices: t2[i,j,a,b] = -t2[j,i,a,b]")
    print("-"*70)
    occ_antisym_violations = []
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvirt):
                for b in range(nvirt):
                    val1 = t2[i,j,a,b]
                    val2 = t2[j,i,a,b]
                    if abs(val1 + val2) > 1e-8:
                        occ_antisym_violations.append((i,j,a,b,val1,val2))
                        print(f"  t2[{i},{j},{a},{b}] = {val1:>10.6f},  t2[{j},{i},{a},{b}] = {val2:>10.6f}  (sum = {val1+val2:>10.6f})")

    if not occ_antisym_violations:
        print("  ✓ Antisymmetry satisfied for all elements!")

    # Check antisymmetry: t2[i,j,a,b] = -t2[i,j,b,a]
    print("\n2. Antisymmetry w.r.t. virtual indices: t2[i,j,a,b] = -t2[i,j,b,a]")
    print("-"*70)
    virt_antisym_violations = []
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvirt):
                for b in range(nvirt):
                    val1 = t2[i,j,a,b]
                    val2 = t2[i,j,b,a]
                    if abs(val1 + val2) > 1e-8:
                        virt_antisym_violations.append((i,j,a,b,val1,val2))
                        print(f"  t2[{i},{j},{a},{b}] = {val1:>10.6f},  t2[{i},{j},{b},{a}] = {val2:>10.6f}  (sum = {val1+val2:>10.6f})")

    if not virt_antisym_violations:
        print("  ✓ Antisymmetry satisfied for all elements!")

    # Check combined symmetry: t2[i,j,a,b] = t2[j,i,b,a]
    print("\n3. Combined symmetry: t2[i,j,a,b] = t2[j,i,b,a]")
    print("-"*70)
    combined_sym_violations = []
    for i in range(nocc):
        for j in range(nocc):
            for a in range(nvirt):
                for b in range(nvirt):
                    val1 = t2[i,j,a,b]
                    val2 = t2[j,i,b,a]
                    if abs(val1 - val2) > 1e-8:
                        combined_sym_violations.append((i,j,a,b,val1,val2))
                        print(f"  t2[{i},{j},{a},{b}] = {val1:>10.6f},  t2[{j},{i},{b},{a}] = {val2:>10.6f}  (diff = {val1-val2:>10.6f})")

    if not combined_sym_violations:
        print("  ✓ Combined symmetry satisfied for all elements!")

    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print(f"Total violations:")
    print(f"  - Occupied antisymmetry: {len(occ_antisym_violations)}")
    print(f"  - Virtual antisymmetry: {len(virt_antisym_violations)}")
    print(f"  - Combined symmetry: {len(combined_sym_violations)}")

    if (len(occ_antisym_violations) == 0 and
        len(virt_antisym_violations) == 0 and
        len(combined_sym_violations) == 0):
        print("\n✓ All symmetry relations satisfied!")
        print("="*70)
        return True
    else:
        print("\n✗ Some symmetry violations found!")
        print("="*70)
        return False


if __name__ == "__main__":
    success = test_rhf_doubles_symmetry()
    exit(0 if success else 1)
