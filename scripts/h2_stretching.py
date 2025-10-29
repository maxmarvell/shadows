"""
In this script we use the methods developed in ____ to obtain
a good estimate for the ground state energy blah..
"""

from pyscf import ao2mo, gto, scf, fci
from shadow_ci.chemistry import MolecularHamiltonian
from shadow_ci.estimator import GroundStateEstimator
from shadow_ci.solvers import VQESolver
from qiskit_nature.second_q.mappers import JordanWignerMapper

from qiskit_nature.second_q.drivers import PySCFDriver

def main():

    # create a H2 mol
    mol = gto.Mole()
    mol.build(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")

    # create mean-field object
    mf = scf.RHF(mol)
    hamiltonian = MolecularHamiltonian.from_pyscf(mf)

    print(hamiltonian.get_hf_state())

    cisolver = fci.FCI(mf)
    e_fci, cvec = cisolver.kernel()
    print(f"Energy from FCI calculation: {e_fci}")

    driver = PySCFDriver(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
    mapper = JordanWignerMapper()
    vqe = VQESolver.from_pyscf_driver(driver, mapper=mapper)
    state, E = vqe.solve()
    print(f"Energy from VQE solver: {E}")

    estimator = GroundStateEstimator(hamiltonian, solver=vqe)
    E, c0, c1, c2 = estimator.estimate_ground_state(10000, 100)

    print(E)


if __name__ == "__main__":
    main()