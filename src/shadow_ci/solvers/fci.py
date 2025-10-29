from shadow_ci.solvers.base import GroundStateSolver

from typing import Tuple
import numpy as np


class FCISolver(GroundStateSolver):

    def solve(self, **options) -> Tuple[np.ndarray, float]:
        import pyscf.fci

        # Check if spin_type is RHF or UHF and use appropriate solver
        if self.hamiltonian.spin_type == "RHF":
            # For RHF, use direct_spin1 (restricted) solver
            energy, civec = pyscf.fci.direct_spin1.kernel(
                self.hamiltonian.h1e,
                self.hamiltonian.h2e,
                self.hamiltonian.norb,
                self.hamiltonian.nelec,
            )
        else:
            # For UHF, h1e and h2e should be tuples/lists for (alpha, beta)
            energy, civec = pyscf.fci.direct_uhf.kernel(
                self.hamiltonian.h1e,
                self.hamiltonian.h2e,
                self.hamiltonian.norb,
                self.hamiltonian.nelec,
            )

        # Add nuclear repulsion energy
        self.energy = energy + self.hamiltonian.nuclear_repulsion
        self.civec = civec
        return civec, self.energy
    
    def _civec_to_statevector(civec):
        pass
