"""Test the classical shadow protocol with a simple known state."""

from pyscf import gto, scf
from shadow_ci.chemistry import MolecularHamiltonian
from shadow_ci.solvers import VQESolver
from shadow_ci.estimator import GroundStateEstimator
import numpy as np

def main():
    # create a H2 mol
    mol = gto.Mole()
    mol.build(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")

    # create mean-field object
    mf = scf.RHF(mol)
    hamiltonian = MolecularHamiltonian.from_pyscf(mf)

    vqe = VQESolver(hamiltonian)
    state, E_vqe = vqe.solve()

    print(f"VQE Energy: {E_vqe}")
    print(f"HF Energy: {mf.e_tot}")
    print(f"Nuclear repulsion: {hamiltonian.nuclear_repulsion}")

    # Test with different sample sizes
    for n_samples in [1000, 5000, 10000, 50000]:
        n_k = min(100, n_samples // 100)
        estimator = GroundStateEstimator(hamiltonian, solver=vqe)
        e_corr, c0, c1, c2 = estimator.estimate_ground_state(n_samples, n_k)

        print(f"\nSamples: {n_samples}, K={n_k}")
        print(f"  e_singles: {e_corr[0]}")
        print(f"  e_doubles: {e_corr[1]}")
        print(f"  Total correlation: {e_corr}")
        print(f"  c0: {c0}")
        print(f"  Imaginary part: {np.imag(e_corr)}")

if __name__ == "__main__":
    main()
