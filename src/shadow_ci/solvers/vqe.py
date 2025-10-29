from typing import Tuple, Optional, Union
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import n_local as TwoLocal
from qiskit_nature.second_q.circuit.library.ansatzes import UCC
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer.primitives import EstimatorV2
from qiskit_aer import AerSimulator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import COBYLA, Optimizer, SLSQP
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.mappers import (
    QubitMapper,
    JordanWignerMapper,
    ParityMapper,
    BravyiKitaevMapper,
)
from qiskit_nature.second_q.circuit.library import HartreeFock

from shadow_ci.solvers.base import GroundStateSolver
from shadow_ci.chemistry.hamiltonian import MolecularHamiltonian

from qiskit_nature.second_q.drivers import PySCFDriver

from qiskit_aer.primitives import Estimator as AerEstimator


class VQESolver(GroundStateSolver):
    """VQE solver with configurable fermion-to-qubit mapping and Hartree-Fock initial state."""

    def __init__(
        self,
        hamiltonian: FermionicOp,
        num_particles: Tuple[int, int],
        num_orbitals: int,
        *,
        ansatz: Optional[QuantumCircuit] = None,
        mapper: Union[str, QubitMapper] = "jw",
        nuclear_repulsion_energy: float = 0.0,
    ):
        """
        Initialize VQE solver.

        Args:
            hamiltonian: MolecularHamiltonian instance
            optimizer: Qiskit optimizer (default: SLSQP)
            ansatz: Custom ansatz circuit (default: TwoLocal)
            mapper: Fermion-to-qubit mapping. Options:
                - "jw" or "jordan_wigner": Jordan-Wigner Transform
                - "parity": Parity encoding
                - "bravyi_kitaev" or "bk": Bravyi-Kitaev Transform
                - Or pass a QubitMapper instance directly
            max_iter: Maximum optimizer iterations
            nuclear_repulsion_energy: Nuclear repulsion energy to add to electronic energy
        """
        super().__init__(hamiltonian)
        self.ansatz = ansatz
        self.num_particles = num_particles
        self.num_orbitals = num_orbitals
        self.mapper = self._setup_mapper(mapper)
        self.nuclear_repulsion_energy = nuclear_repulsion_energy

    def _setup_mapper(self, mapper: Union[str, QubitMapper]) -> QubitMapper:
        """Set up the fermion-to-qubit mapper."""
        if isinstance(mapper, QubitMapper):
            return mapper

        mapper_dict = {
            "jw": JordanWignerMapper,
            "jordan_wigner": JordanWignerMapper,
            "parity": ParityMapper,
            "bravyi_kitaev": BravyiKitaevMapper,
            "bk": BravyiKitaevMapper,
        }

        mapper_lower = mapper.lower()
        if mapper_lower not in mapper_dict:
            raise ValueError(
                f"Unknown mapper '{mapper}'. Choose from: {list(mapper_dict.keys())}"
            )

        return mapper_dict[mapper_lower]()

    @classmethod
    def from_pyscf_driver(cls, driver: PySCFDriver, *, mapper: QubitMapper) -> 'VQESolver':
        problem = driver.run()
        num_particles = problem.num_particles        # (n_alpha, n_beta)
        num_spatial_orbitals = problem.num_spatial_orbitals
        hamiltonian = problem.second_q_ops()[0]
        nuclear_repulsion = problem.nuclear_repulsion_energy
        return cls(
            hamiltonian,
            num_particles,
            num_spatial_orbitals,
            mapper=mapper,
            nuclear_repulsion_energy=nuclear_repulsion
        )

    def solve(self, *, initial_state: Optional[Union[HartreeFock, QuantumCircuit]] = None, ) -> Tuple[Statevector, float]:
        """
        Solve for ground state using VQE.

        Returns:
            (optimal_circuit, energy) tuple
        """

        if initial_state is None:
            initial_state = HartreeFock(
                num_spatial_orbitals=self.num_orbitals,
                num_particles=self.num_particles,
                qubit_mapper=self.mapper,
            )
        
        ansatz = UCC(
            excitations="sd",
            num_spatial_orbitals=self.num_orbitals,
            num_particles=self.num_particles,
            qubit_mapper=self.mapper,
            initial_state=initial_state,            # start from mean-field |HF>
        )

        # Transpile the ansatz to basic gates that AerEstimator can handle
        # We need to transpile rather than just decompose to handle PauliEvolution gates
        backend = AerSimulator()
        ansatz = transpile(ansatz, backend, optimization_level=0)

        estimator = EstimatorV2()
        optimizer = SLSQP(maxiter=200)
        vqe = VQE(estimator, ansatz, optimizer)

        qubit_hamiltonian = self.mapper.map(self.hamiltonian)

        result = vqe.compute_minimum_eigenvalue(operator=qubit_hamiltonian)

        # Add nuclear repulsion energy to get total energy
        total_energy = result.optimal_value + self.nuclear_repulsion_energy

        return Statevector(ansatz.assign_parameters(result.optimal_parameters)), total_energy

if __name__ == "__main__":

    from pyscf import gto, scf

    # create a H2 mol
    mol = gto.Mole()
    mol.build(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")

    # create mean-field object
    mf = scf.RHF(mol)
    H = MolecularHamiltonian.from_pyscf(mf)

    vqe = VQESolver(H, mapper="bravyi_kitaev")
    state, gs = vqe.solve()

    from shadow_ci.solvers.fci import FCISolver
    fci = FCISolver(H)
    state_exact, gs_exact = fci.solve()

    print(gs, gs_exact)

    pass