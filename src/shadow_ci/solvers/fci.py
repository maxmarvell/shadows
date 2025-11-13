from shadow_ci.solvers.base import GroundStateSolver
from typing import Tuple, Union
import numpy as np
import pyscf.fci
from qiskit.quantum_info import Statevector
from pyscf import scf, ao2mo


class FCISolver(GroundStateSolver):

    def __init__(self, mf: Union[scf.hf.RHF, scf.uhf.UHF]):
        super().__init__(mf)
        self._compute_integrals()

    def _compute_integrals(self):
        """Compute and cache integrals from mf object."""
        hcore = self.mf.get_hcore()
        self.h1e = self.mf.mo_coeff.T @ hcore @ self.mf.mo_coeff
        
        eri_ao = self.mf.mol.intor('int2e')
        self.h2e = ao2mo.full(eri_ao, self.mf.mo_coeff)
        
        self.norb = self.h1e.shape[0]
        self.nelec = self.mf.mol.nelec
        self.nuclear_repulsion = self.mf.mol.energy_nuc()

    def solve(self, **options):

        if isinstance(self.mf, scf.hf.RHF):
            energy, civec = pyscf.fci.FCI(self.mf).kernel()
            self.energy = energy
        else:
            energy, civec = pyscf.fci.direct_uhf.kernel(
                self.h1e,
                self.h2e,
                self.norb,
                self.nelec,
                **options
            )
            self.energy = energy + self.nuclear_repulsion

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

        n_alpha, n_beta = self.nelec
        norb = self.norb
        n_qubits = 2 * norb

        full_statevector = np.zeros(2**n_qubits, dtype=complex)

        if isinstance(self.mf, scf.hf.RHF):
            from pyscf.fci import cistring

            alpha_strings = cistring.make_strings(range(norb), n_alpha)
            beta_strings = cistring.make_strings(range(norb), n_beta)

            for i_alpha, alpha_str in enumerate(alpha_strings):
                for i_beta, beta_str in enumerate(beta_strings):
                    ci_coeff = civec[i_alpha, i_beta]

                    qubit_index = int(alpha_str) + (int(beta_str) << norb)

                    full_statevector[qubit_index] = ci_coeff
        else:
            from pyscf.fci import cistring

            alpha_strings = cistring.make_strings(range(norb), n_alpha)
            beta_strings = cistring.make_strings(range(norb), n_beta)

            civec_flat = civec.ravel()

            for i_alpha, alpha_str in enumerate(alpha_strings):
                for i_beta, beta_str in enumerate(beta_strings):
                    idx = i_alpha * len(beta_strings) + i_beta
                    ci_coeff = civec_flat[idx]

                    qubit_index = int(alpha_str) + (int(beta_str) << norb)
                    full_statevector[qubit_index] = ci_coeff

        return Statevector(full_statevector)
