from abc import ABC, abstractmethod
from typing import Tuple, Any
from shadow_ci.chemistry.hamiltonian import MolecularHamiltonian
from qiskit_nature.second_q.operators import FermionicOp


class GroundStateSolver(ABC):
    """Abstract base class for quantum state solvers."""

    def __init__(self, hamiltonian: FermionicOp):
        self.hamiltonian = hamiltonian
        self.energy = None
        self.state = None

    @abstractmethod
    def solve(self, **options) -> Tuple[Any, float]:
        """
        Solve for ground state.

        Returns:
            (state, energy) tuple
        """
        pass

    def get_statevector(self):
        """Return state as a statevector (if applicable)."""
        pass
