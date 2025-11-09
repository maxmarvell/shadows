"""
In this script we use the methods developed in ____ to obtain
a good estimate for the ground state energy blah..
"""

from pyscf import gto, scf
import numpy as np
from shadow_ci.shadows import ShadowProtocol
from qiskit.quantum_info import Statevector
from shadow_ci.utils import make_hydrogen_chain
from shadow_ci.hamiltonian import MolecularHamiltonian
from shadow_ci.estimator import GroundStateEstimator
from shadow_ci.solvers import FCISolver

def main():

     # Build H2 molecule at equilibrium geometry
    mol = gto.Mole()
    atom = make_hydrogen_chain(6, 0.5)
    mol.build(atom=atom, basis="sto-3g")
    print(f"\nMolecule: H2 (bond length = 0.50 Å)")
    print(f"Basis set: sto-3g")

    # Compute Hartree-Fock reference
    mf = scf.RHF(mol)
    hamiltonian = MolecularHamiltonian.from_pyscf(mf)
    print(f"HF Energy: {hamiltonian.hf_energy:.8f} Ha")

    fci_solver = FCISolver(hamiltonian)
    estimator = GroundStateEstimator(hamiltonian, solver=fci_solver, verbose=4)
    estimator.estimate_ground_state(
        n_samples=1000,
        n_k_estimators=20,
        use_qualcs=True,
        n_jobs=2
    )
    

if __name__ == "__main__":
    main()