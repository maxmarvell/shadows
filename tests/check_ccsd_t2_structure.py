"""Check the structure of PySCF's CCSD t2 amplitudes."""

import numpy as np
from pyscf import gto, scf, cc
from shadow_ci.utils import make_hydrogen_chain

# Build H4
mol = gto.Mole()
atom = make_hydrogen_chain(4, 0.74)
mol.build(atom=atom, basis='sto-3g', verbose=0)

mf = scf.RHF(mol)
mf.kernel()

mycc = cc.CCSD(mf)
mycc.kernel()

print("="*70)
print("Exploring PySCF CCSD t2 structure")
print("="*70)

# Check what attributes the CCSD object has
print("\nCCSD object attributes related to t2:")
for attr in dir(mycc):
    if 't2' in attr.lower():
        print(f"  - {attr}")

print(f"\nType of mycc.t2: {type(mycc.t2)}")
print(f"Shape of mycc.t2: {mycc.t2.shape}")

# Check if there's a way to get spin-separated amplitudes
print("\n" + "="*70)
print("Checking for spin-separated methods")
print("="*70)

# Try to find methods for spin-unrestricted
for attr in dir(mycc):
    if 'spin' in attr.lower() or 'unrestrict' in attr.lower():
        print(f"  - {attr}")

# The t2 for RHF is spin-integrated
# Let's manually compute what alpha-alpha and alpha-beta would be

nocc = 2
nvirt = 2
t2 = mycc.t2

print("\n" + "="*70)
print("Understanding the spin structure")
print("="*70)

print("\nFor RHF, t2[i,j,a,b] represents spatial orbital amplitudes.")
print("This is related to spin-orbital amplitudes by:")
print("  t2_aaaa[i,j,a,b] = t2[i,j,a,b] - t2[i,j,b,a]  (antisymmetric)")
print("  t2_bbbb[i,j,a,b] = t2[i,j,a,b] - t2[i,j,b,a]  (same as aaaa)")
print("  t2_abab[i,j,a,b] = t2[i,j,a,b]                 (not antisymmetrized)")

print("\nLet's verify this interpretation:")
print("-"*70)

for i in range(nocc):
    for j in range(nocc):
        for a in range(nvirt):
            for b in range(nvirt):
                if i < j and a < b:  # Unique alpha-alpha excitations
                    t2_spatial = t2[i,j,a,b]
                    t2_antisym = t2[i,j,a,b] - t2[i,j,b,a]

                    print(f"Excitation ({i},{j}) → ({a},{b}):")
                    print(f"  Spatial t2[{i},{j},{a},{b}]           = {t2_spatial:>10.6f}")
                    print(f"  Antisym t2[{i},{j},{a},{b}] - t2[{i},{j},{b},{a}] = {t2_antisym:>10.6f}")
                    print()
