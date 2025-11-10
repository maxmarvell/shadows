from abc import ABC, abstractmethod
from typing import Tuple, Union
from qiskit_nature.second_q.operators import FermionicOp
from qiskit.quantum_info import Statevector
from pyscf import scf


class GroundStateSolver(ABC):
    """Abstract base class for quantum state solvers."""

    def __init__(self, mf: Union[scf.hf.RHF, scf.uhf.UHF]):
        self.mf = mf
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
