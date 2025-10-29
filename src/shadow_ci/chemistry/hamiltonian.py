from typing import Tuple

import numpy as np
from pyscf import ao2mo
from pyscf.scf.hf import RHF
from pyscf.ci import cisd
from typing import List, Literal, NamedTuple
from numpy.typing import NDArray
from dataclasses import dataclass

from qiskit_nature.second_q.mappers import JordanWignerMapper, BravyiKitaevMapper
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit.quantum_info import Statevector
from shadow_ci.utils import Bitstring, SingleExcitation, DoubleExcitation

EncodingType = Literal["jordan_wigner", "bravyi_kitaev", "parity"]

class MolecularHamiltonian:

    def __init__(
        self,
        h1e: np.ndarray,
        h2e: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
        nuclear_repulsion: float = 0.0,
        spin_type: str = "RHF",
        encoding: EncodingType = "jordan_wigner"
    ):
        self.h1e = h1e
        self.h2e = h2e
        self.norb = norb
        self.nelec = nelec
        self.nuclear_repulsion = nuclear_repulsion
        self.spin_type = spin_type
        self.encoding = encoding

    @classmethod
    def from_pyscf(cls, mf: RHF):
        mf.kernel()
        hcore = mf.get_hcore()
        mo_coeff = mf.mo_coeff
        h1e = mo_coeff.T @ hcore @ mo_coeff
        eri_ao = mf.mol.intor("int2e")
        eri_mo = ao2mo.full(eri_ao, mo_coeff)
        nuclear_repulsion = mf.mol.energy_nuc()
        return cls(
            h1e,
            eri_mo,
            h1e.shape[0],
            nelec=mf.mol.nelec,
            nuclear_repulsion=nuclear_repulsion,
            spin_type="RHF"
        )
    
    def get_hf_state(self) -> Statevector:
        """Get HF single mean-field slater determinant as qubit state vector with proper encoding."""

        if self.encoding == "jordan_wigner":
            mapper = JordanWignerMapper()
        elif self.encoding == "bravyi_kitaev":
            raise NotImplementedError()
        elif self.encoding == "parity":
            raise NotImplementedError()
        else:
            raise ValueError(f"Unknown encoding: {self.encoding}")
        
        n_alpha, n_beta = self.nelec
        hf = HartreeFock(
            num_spatial_orbitals=self.norb,
            num_particles=(n_alpha, n_beta),
            qubit_mapper=mapper
        )

        return Statevector(hf)
    
    def get_hf_bitstring(self) -> Bitstring:
        n_alpha, n_beta = self.nelec
        alpha_string = [True] * n_alpha + [False] * (self.norb - n_alpha)
        beta_string = [True] * n_beta + [False] * (self.norb - n_beta)
        return Bitstring(alpha_string + beta_string)

    def get_single_excitations(self) -> List[SingleExcitation]:
        """Generate all single excitations from the HF reference in canonical order.

        Ordering convention:
        1. All alpha excitations: i=0..nocc_alpha, a=nocc_alpha..norb (lexicographic)
        2. All beta excitations: i=0..nocc_beta, a=nocc_beta..norb (lexicographic)

        Returns:
            List of SingleExcitation objects with indices and bitstrings
        """
        n_alpha, n_beta = self.nelec
        n_qubits = 2 * self.norb
        hf_bitstring = self.get_hf_bitstring()
        excitations = []

        occupied_alpha = list(range(n_alpha))
        virtual_alpha = list(range(n_alpha, self.norb))

        # Alpha excitations (ordered)
        for i in occupied_alpha:
            for a in virtual_alpha:
                excited_state = hf_bitstring.array.copy()
                excited_state[i] = False
                excited_state[a] = True
                excitations.append(SingleExcitation(
                    occ=i,
                    virt=a,
                    spin='alpha',
                    bitstring=Bitstring(excited_state)
                ))

        occupied_beta = list(range(self.norb, self.norb + n_beta))
        virtual_beta = list(range(self.norb + n_beta, n_qubits))

        # Beta excitations (ordered)
        for i in occupied_beta:
            for a in virtual_beta:
                excited_state = hf_bitstring.array.copy()
                excited_state[i] = False
                excited_state[a] = True
                # Store spatial orbital indices (subtract norb offset)
                excitations.append(SingleExcitation(
                    occ=i - self.norb,
                    virt=a - self.norb,
                    spin='beta',
                    bitstring=Bitstring(excited_state)
                ))

        return excitations

    def get_double_excitations(self) -> List[DoubleExcitation]:
        """Generate all double excitations from the HF reference in canonical order.

        Ordering convention:
        1. Alpha-Alpha: (i,j → a,b) with i < j, a < b (lexicographic)
        2. Beta-Beta: (i,j → a,b) with i < j, a < b (lexicographic)
        3. Alpha-Beta: (i_α,j_β → a_α,b_β) (lexicographic on all indices)

        Returns:
            List of DoubleExcitation objects with indices and bitstrings
        """
        n_alpha, n_beta = self.nelec
        n_qubits = 2 * self.norb
        hf_bitstring = self.get_hf_bitstring()
        excitations = []

        # Alpha spin indices
        occupied_alpha = list(range(n_alpha))
        virtual_alpha = list(range(n_alpha, self.norb))

        # Beta spin indices
        occupied_beta = list(range(self.norb, self.norb + n_beta))
        virtual_beta = list(range(self.norb + n_beta, n_qubits))

        # Alpha-Alpha double excitations
        for idx_i, i in enumerate(occupied_alpha):
            for j in occupied_alpha[idx_i + 1:]:
                for idx_a, a in enumerate(virtual_alpha):
                    for b in virtual_alpha[idx_a + 1:]:
                        excited_state = hf_bitstring.array.copy()
                        excited_state[i] = False
                        excited_state[j] = False
                        excited_state[a] = True
                        excited_state[b] = True
                        excitations.append(DoubleExcitation(
                            occ1=i, occ2=j,
                            virt1=a, virt2=b,
                            spin_case='alpha-alpha',
                            bitstring=Bitstring(excited_state)
                        ))

        # Beta-Beta double excitations
        for idx_i, i in enumerate(occupied_beta):
            for j in occupied_beta[idx_i + 1:]:
                for idx_a, a in enumerate(virtual_beta):
                    for b in virtual_beta[idx_a + 1:]:
                        excited_state = hf_bitstring.array.copy()
                        excited_state[i] = False
                        excited_state[j] = False
                        excited_state[a] = True
                        excited_state[b] = True
                        # Store spatial orbital indices
                        excitations.append(DoubleExcitation(
                            occ1=i - self.norb, occ2=j - self.norb,
                            virt1=a - self.norb, virt2=b - self.norb,
                            spin_case='beta-beta',
                            bitstring=Bitstring(excited_state)
                        ))

        # Alpha-Beta mixed excitations
        for i in occupied_alpha:
            for j in occupied_beta:
                for a in virtual_alpha:
                    for b in virtual_beta:
                        excited_state = hf_bitstring.array.copy()
                        excited_state[i] = False
                        excited_state[j] = False
                        excited_state[a] = True
                        excited_state[b] = True
                        excitations.append(DoubleExcitation(
                            occ1=i, occ2=j - self.norb,
                            virt1=a, virt2=b - self.norb,
                            spin_case='alpha-beta',
                            bitstring=Bitstring(excited_state)
                        ))

        return excitations

    def get_excitations(self, order: int = 2):
        """Generate all excitations up to a given order with canonical ordering.

        Args:
            order: Maximum excitation order (1 or 2)

        Returns:
            Tuple of (reference_bitstring, singles, doubles)
            - singles: List[SingleExcitation] (empty if order < 1)
            - doubles: List[DoubleExcitation] (empty if order < 2)
        """
        reference = self.get_hf_bitstring()
        singles = self.get_single_excitations() if order >= 1 else []
        doubles = self.get_double_excitations() if order >= 2 else []

        return reference, singles, doubles
    
    def get_antisymettry_excitation_signs(self):
        _, signs = cisd.tn_addrs_signs(self.norb, self.nelec, 1)
        return signs

    def get_fock_matrix(self) -> np.ndarray:
        """Compute the Fock matrix in the molecular orbital basis.

        For a restricted closed-shell system, the Fock matrix is:
        F_pq = h_pq + Σ_i [2(pq|ii) - (pi|qi)]

        where the sum runs over occupied orbitals and (pq|rs) are the
        two-electron integrals in chemist notation.

        Returns:
            np.ndarray: Fock matrix in MO basis (norb x norb)
        """
        norb = self.norb
        n_alpha, n_beta = self.nelec

        # Start with one-electron (core) Hamiltonian
        fock = self.h1e.copy()

        # Reshape two-electron integrals for easier indexing
        # h2e is stored as a flat array, reshape to (norb, norb, norb, norb)
        eri = self.h2e.reshape(norb, norb, norb, norb)

        # Add two-electron contributions
        # For restricted closed-shell, only use doubly occupied orbitals
        if self.spin_type == "RHF":
            nocc = n_alpha  # Number of doubly occupied orbitals
            for p in range(norb):
                for q in range(norb):
                    for i in range(nocc):
                        # Coulomb: 2 * (pq|ii)
                        fock[p, q] += 2.0 * eri[p, q, i, i]
                        # Exchange: -(pi|qi)
                        fock[p, q] -= eri[p, i, q, i]
        else:
            # For unrestricted calculations, would need separate alpha/beta treatment
            raise NotImplementedError("Fock matrix for UHF not yet implemented")

        return fock
    
    def get_g_ovvo(self):
        if self.spin_type == 'RHF':
            nocc, _ = self.nelec
            # h2e from ao2mo.full() is stored as flat array, need to reshape
            eri = self.h2e.reshape(self.norb, self.norb, self.norb, self.norb)
            return eri[:nocc, nocc:, nocc:, :nocc]
        else:
            raise NotImplementedError("Fock matrix for UHF not yet implemented")

if __name__ == "__main__":

    from pyscf import ao2mo, gto, scf

    # create a H2 mol
    mol = gto.Mole()
    mol.build(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")

    # create mean-field object
    mf = scf.RHF(mol)
    hamiltonian = MolecularHamiltonian.from_pyscf(mf)

    ref, single, double = hamiltonian.get_excitations()

    print(ref, single, double)
