from abc import ABC, abstractmethod
from typing import Tuple, Any
from shadow_ci.hamiltonian import MolecularHamiltonian
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import Statevector


class GroundStateSolver(ABC):
    """Abstract base class for quantum state solvers."""

    def __init__(self, hamiltonian: FermionicOp):
        self.hamiltonian = hamiltonian
        self.energy = None
        self.state = None

    @abstractmethod
    def solve(self, **options) -> Tuple[Statevector, float]:
        """
        Solve for ground state.

        Returns:
            (state, energy) tuple
        """
        pass
