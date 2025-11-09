from pyscf import gto, scf
from shadow_ci.hamiltonian import MolecularHamiltonian
from shadow_ci.estimator import GroundStateEstimator
from shadow_ci.solvers import FCISolver
from shadow_ci.utils import make_hydrogen_chain
from shadow_ci.shadows import ShadowProtocol
import numpy as np
import matplotlib.pyplot as plt
from plotting_config import setup_plotting_style, save_figure

# Simulation parameters
N_SAMPLES = 100000          # Number of shadow measurement samples per estimation
N_ESTIMATORS = 40         # Number of median-of-means estimators (k in paper)
N_HYDROGEN = 6         
INTERATOMIC_DISTANCE = 0.5  

def main():

    # Build H2 molecule at current geometry
    mol_string = make_hydrogen_chain(N_HYDROGEN, INTERATOMIC_DISTANCE)
    mol = gto.Mole()
    mol.build(atom=mol_string, basis="sto-3g")

    # Compute Hartree-Fock reference state
    mf = scf.RHF(mol)
    hamiltonian = MolecularHamiltonian.from_pyscf(mf)
    
    fci_solver = FCISolver(hamiltonian)
    estimator = GroundStateEstimator(hamiltonian, solver=fci_solver, verbose=4)

    protocol = ShadowProtocol(estimator.trial, n_jobs=6, use_qulacs=True, verbose=3)
    protocol.collect_samples(N_SAMPLES, N_ESTIMATORS, prediction='overlap')

    res = estimator.estimate_first_order_interactions(protocol)

    print(res)


if __name__ == "__main__":
    main()