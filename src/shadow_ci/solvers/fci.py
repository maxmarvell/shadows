from shadow_ci.solvers.base import GroundStateSolver
from typing import Tuple
import numpy as np
import pyscf.fci
from qiskit.quantum_info import Statevector
from shadow_ci.hamiltonian import MolecularHamiltonian


class FCISolver(GroundStateSolver):

    def __init__(self, hamiltonian: MolecularHamiltonian):
        super().__init__(hamiltonian)
        self.hamiltonian = hamiltonian

    def solve(self, **options) -> Tuple[Statevector, float]:

        if self.hamiltonian.spin_type == "RHF":
            energy, civec = pyscf.fci.direct_spin0.kernel(
                self.hamiltonian.h1e,
                self.hamiltonian.h2e,
                self.hamiltonian.norb,
                self.hamiltonian.nelec,
            )
        else:
            energy, civec = pyscf.fci.direct_uhf.kernel(
                self.hamiltonian.h1e,
                self.hamiltonian.h2e,
                self.hamiltonian.norb,
                self.hamiltonian.nelec,
            )

        self.energy = energy + self.hamiltonian.nuclear_repulsion

        self.state = self._civec_to_statevector(civec)
        return self.state, self.energy

    def _civec_to_statevector(self, civec: np.ndarray) -> Statevector:
        """Convert PySCF FCI CI vector to Qiskit Statevector.

        PySCF FCI stores the wavefunction in a compressed format over
        determinants. This method expands it to the full 2^n qubit basis.

        Args:
            civec: CI vector from PySCF FCI solver

        Returns:
            Qiskit Statevector in computational basis
        """
        norb = self.hamiltonian.norb
        n_alpha, n_beta = self.hamiltonian.nelec
        n_qubits = 2 * norb

        # Initialize full statevector (2^n_qubits amplitudes)
        full_statevector = np.zeros(2**n_qubits, dtype=complex)

        if self.hamiltonian.spin_type == "RHF":
            # For RHF, use direct_spin1 addressing
            from pyscf.fci import cistring

            # Generate all alpha and beta string addresses
            alpha_strings = cistring.make_strings(range(norb), n_alpha)
            beta_strings = cistring.make_strings(range(norb), n_beta)

            # Map each FCI determinant to its corresponding qubit basis state
            for i_alpha, alpha_str in enumerate(alpha_strings):
                for i_beta, beta_str in enumerate(beta_strings):
                    # Get CI coefficient for this determinant
                    ci_coeff = civec[i_alpha, i_beta]

                    # Convert to qubit index (Jordan-Wigner encoding)
                    # In JW: |α₀α₁...α_{norb-1}β₀β₁...β_{norb-1}⟩
                    # PySCF strings are in occupation number representation
                    qubit_index = int(alpha_str) + (int(beta_str) << norb)

                    full_statevector[qubit_index] = ci_coeff
        else:
            # UHF case
            from pyscf.fci import cistring

            alpha_strings = cistring.make_strings(range(norb), n_alpha)
            beta_strings = cistring.make_strings(range(norb), n_beta)

            # civec might have different shape for UHF
            civec_flat = civec.ravel()

            for i_alpha, alpha_str in enumerate(alpha_strings):
                for i_beta, beta_str in enumerate(beta_strings):
                    idx = i_alpha * len(beta_strings) + i_beta
                    ci_coeff = civec_flat[idx]

                    qubit_index = int(alpha_str) + (int(beta_str) << norb)
                    full_statevector[qubit_index] = ci_coeff

        return Statevector(full_statevector)
