from typing import Tuple, Optional, Union
from qiskit import QuantumCircuit, transpile
from qiskit_nature.second_q.circuit.library.ansatzes import UCC
from qiskit_aer.primitives import EstimatorV2
from qiskit_aer import AerSimulator
from qiskit_algorithms import VQE
from qiskit_algorithms.optimizers import SLSQP
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.mappers import (
    QubitMapper,
    JordanWignerMapper,
    ParityMapper,
    BravyiKitaevMapper,
)
from qiskit_nature.second_q.circuit.library import HartreeFock

from shadow_ci.solvers.base import GroundStateSolver

from qiskit_nature.second_q.drivers import PySCFDriver

from pyscf import scf


class VQESolver(GroundStateSolver):
    """VQE solver with configurable fermion-to-qubit mapping and Hartree-Fock initial state."""

    def __init__(
        self,
        mf: Union[scf.hf.RHF, scf.uhf.UHF],
        *,
        mapper: Union[str, QubitMapper] = "jw",
    ):
        """Initialize VQE solver from PySCF mean-field object.
        
        Args:
            mf: PySCF mean-field object (RHF or UHF)
            mapper: Fermion-to-qubit mapping (default: "jw")
        """
        super().__init__(mf)
        self.mapper = self._setup_mapper(mapper)
        self._setup_from_pyscf()

    def _setup_from_pyscf(self):
        """Extract problem data from PySCF via PySCFDriver."""
        
        atom_string = self.mf.mol.atom 
        basis = self.mf.mol.basis
        charge = self.mf.mol.charge
        spin = self.mf.mol.spin
        
        driver = PySCFDriver(
            atom=atom_string,
            basis=basis,
            charge=charge,
            spin=spin
        )
        
        problem = driver.run()
        self.num_particles = problem.num_particles
        self.num_orbitals = problem.num_spatial_orbitals
        self.hamiltonian = problem.second_q_ops()[0]
        self.nuclear_repulsion_energy = problem.nuclear_repulsion_energy

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
        num_particles = problem.num_particles
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

        backend = AerSimulator()
        ansatz = transpile(ansatz, backend, optimization_level=0)

        estimator = EstimatorV2()
        optimizer = SLSQP(maxiter=200)
        vqe = VQE(estimator, ansatz, optimizer)

        qubit_hamiltonian = self.mapper.map(self.hamiltonian)

        result = vqe.compute_minimum_eigenvalue(operator=qubit_hamiltonian)

        total_energy = result.optimal_value + self.nuclear_repulsion_energy

        return Statevector(ansatz.assign_parameters(result.optimal_parameters)), total_energy

if __name__ == "__main__":

    pass