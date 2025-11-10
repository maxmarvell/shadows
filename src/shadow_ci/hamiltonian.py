from typing import Tuple

import numpy as np
from pyscf import ao2mo
from pyscf.scf.hf import RHF
from typing import List, Literal, Optional

from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit.quantum_info import Statevector
from shadow_ci.utils import (
    Bitstring,
    SingleExcitation,
    DoubleExcitation,
    SingleAmplitudes,
    DoubleAmplitudes,
)

EncodingType = Literal["jordan_wigner", "bravyi_kitaev", "parity"]

class MolecularHamiltonian:
    """Quantum chemistry domain model for shadow tomography.

    This class serves as a bridge between PySCF quantum chemistry calculations
    and the shadow tomography protocol. It stores molecular integrals in the
    molecular orbital (MO) basis and provides domain-specific operations for
    generating excitations, computing energies, and converting between different
    quantum state representations.

    **Spin-Orbital Ordering Convention:**
    Spin orbitals are indexed as: [α₀, α₁, ..., α_{norb-1}, β₀, β₁, ..., β_{norb-1}]
    where α denotes spin-up and β denotes spin-down electrons.

    **Fermion-to-Qubit Encoding:**
    Supports multiple encodings (Jordan-Wigner, Bravyi-Kitaev, Parity).
    Default is Jordan-Wigner where qubit i represents spin-orbital i.

    **Bitstring Convention:**
    Bitstrings use little-endian convention: bit i (from right) = orbital i.
    This matches PySCF's occupation string representation.

    **RHF Optimization:**
    For RHF systems, excitation methods automatically return only unique
    excitations (exploiting spin symmetry), reducing shadow measurements by ~40%.

    **Integral Storage:**
    - h1e: One-electron integrals in MO basis (norb × norb)
    - h2e: Two-electron integrals in MO basis, stored as flat array from ao2mo.full()
    - Both integrals are in physicist's notation: (pq|rs)

    Attributes:
        h1e: One-electron Hamiltonian matrix (MO basis)
        h2e: Two-electron repulsion integrals (flattened, MO basis)
        norb: Number of spatial orbitals
        nelec: Tuple of (n_alpha, n_beta) electrons
        nuclear_repulsion: Nuclear repulsion energy in Hartrees
        spin_type: "RHF" or "UHF" (currently RHF-focused)
        encoding: Fermion-to-qubit encoding scheme
        nocc: Property returning number of occupied orbitals
        nvirt: Property returning number of virtual orbitals
        hf_energy: Property returning Hartree-Fock energy

    Example:
        >>> from pyscf import gto, scf
        >>> mol = gto.Mole()
        >>> mol.build(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g')
        >>> mf = scf.RHF(mol)
        >>> H = MolecularHamiltonian.from_pyscf(mf)
        >>> print(f"HF Energy: {H.hf_energy:.6f} Ha")
        >>> singles = H.get_single_excitations()  # Only alpha for RHF
        >>> doubles = H.get_double_excitations()  # α-α and α-β for RHF
    """

    def __init__(
        self,
        h1e: np.ndarray,
        h2e: np.ndarray,
        norb: int,
        nelec: Tuple[int, int],
        nuclear_repulsion: float = 0.0,
        spin_type: str = "RHF",
        encoding: EncodingType = "jordan_wigner",
        hf_energy: float = None
    ):
        self.h1e = h1e
        self.h2e = h2e
        self.norb = norb
        self.nelec = nelec
        self.nuclear_repulsion = nuclear_repulsion
        self.spin_type = spin_type
        self.encoding = encoding
        self._hf_energy = hf_energy

    @property
    def nocc(self) -> int:
        """Number of occupied spatial orbitals.

        For RHF, this is the number of doubly-occupied orbitals.
        For UHF, this returns the number of alpha electrons.

        Returns:
            Number of occupied spatial orbitals
        """
        return self.nelec[0]

    @property
    def nvirt(self) -> int:
        """Number of virtual (unoccupied) spatial orbitals.

        Returns:
            Number of virtual spatial orbitals
        """
        return self.norb - self.nocc

    @classmethod
    def from_pyscf(cls, mf: RHF):
        mf.kernel()
        hcore = mf.get_hcore()
        mo_coeff = mf.mo_coeff
        h1e = mo_coeff.T @ hcore @ mo_coeff
        eri_ao = mf.mol.intor("int2e")
        eri_mo = ao2mo.full(eri_ao, mo_coeff)
        nuclear_repulsion = mf.mol.energy_nuc()
        hf_energy = mf.e_tot
        return cls(
            h1e,
            eri_mo,
            h1e.shape[0],
            nelec=mf.mol.nelec,
            nuclear_repulsion=nuclear_repulsion,
            spin_type="RHF",
            hf_energy=hf_energy
        )
    
    def get_hf_state(self) -> Statevector:
        """Get HF single mean-field slater determinant as qubit state vector with proper encoding.

        Note: Currently only Jordan-Wigner encoding is fully supported.
        """

        if self.encoding == "jordan_wigner":
            mapper = JordanWignerMapper()
        elif self.encoding == "bravyi_kitaev":
            raise NotImplementedError("Bravyi-Kitaev encoding not yet supported")
        elif self.encoding == "parity":
            raise NotImplementedError("Parity encoding not yet supported")
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
        """Get the HF mean-field state as a bitstring. """
        n_alpha, n_beta = self.nelec
        alpha_string = [True] * n_alpha + [False] * (self.norb - n_alpha) 
        beta_string = [True] * n_beta + [False] * (self.norb - n_beta) 
        return Bitstring(alpha_string + beta_string, endianess='little')

    def get_single_excitations(self) -> List[SingleExcitation]:
        """Generate all unique single excitations from the HF reference in canonical order.

        For RHF systems, only alpha excitations are returned since beta excitations
        are equivalent by spin symmetry (reduces shadow measurements by 50%).

        For UHF systems, both alpha and beta excitations are returned.

        Returns:
            List of SingleExcitation objects with indices and bitstrings
        """
        n_alpha, n_beta = self.nelec
        n_qubits = 2 * self.norb
        hf_bitstring = self.get_hf_bitstring()
        excitations = []

        occupied_alpha = list(range(n_alpha))
        virtual_alpha = list(range(n_alpha, self.norb))

        # alpha excitations
        for i in occupied_alpha:
            for a in virtual_alpha:
                excited_state = hf_bitstring.copy()
                excited_state[i] = False
                excited_state[a] = True
                excitations.append(SingleExcitation(
                    occ=i,
                    virt=a - n_alpha,
                    spin='alpha',
                    bitstring=excited_state,
                    n=n_alpha
                ))

        if self.spin_type == "RHF": return excitations

        occupied_beta = list(range(self.norb, self.norb + n_beta))
        virtual_beta = list(range(self.norb + n_beta, n_qubits))

        # beta excitations
        for i in occupied_beta:
            for a in virtual_beta:
                excited_state = hf_bitstring.copy()
                excited_state[i] = False
                excited_state[a] = True
                excitations.append(SingleExcitation(
                    occ=i - self.norb,
                    virt=a - self.norb - n_beta,
                    spin='beta',
                    bitstring=excited_state, 
                    n=n_beta
                ))

        return excitations

    def get_double_excitations(self) -> List[DoubleExcitation]:
        """Generate unique double excitations from the HF reference.

        For RHF systems, only alpha-alpha and alpha-beta excitations are returned.
        Beta-beta excitations are equivalent to alpha-alpha by spin symmetry.
        This reduces the number of unique excitations by ~33%.

        For UHF systems, all three spin cases are returned: alpha-alpha, beta-beta, alpha-beta.

        Returns:
            List of DoubleExcitation objects with indices and bitstrings
        """

        n_alpha, n_beta = self.nelec
        n_qubits = 2 * self.norb
        reference = self.get_hf_bitstring()
        excitations = []

        # alpha spin indices
        occupied_alpha = list(range(n_alpha))
        virtual_alpha = list(range(n_alpha, self.norb))

        # beta spin indices
        occupied_beta = list(range(self.norb, self.norb + n_beta))
        virtual_beta = list(range(self.norb + n_beta, n_qubits))

        # alpha-beta mixed excitations
        for idx_i, i in enumerate(occupied_alpha):
            if self.spin_type != 'RHF': idx_i = 0
            else: idx_i += 1
            for j in occupied_beta[idx_i:]:
                for a in virtual_alpha:
                    for b in virtual_beta:
                        if a + self.norb == b: continue
                        excited_state = reference.copy()
                        excited_state[i] = False
                        excited_state[j] = False
                        excited_state[a] = True
                        excited_state[b] = True
                        excitations.append(DoubleExcitation(
                            occ1=i, occ2=j - self.norb,
                            virt1=a - n_alpha, virt2=b - self.norb - n_beta,
                            spin_case='alpha-beta',
                            bitstring=excited_state,
                            n1=n_alpha, n2=n_beta
                        ))

        # get double excitations from same orbital
        if self.spin_type == 'RHF':
            for i in occupied_alpha:
                for a in virtual_alpha:
                    excited_state = reference.copy()
                    excited_state[i] = False
                    excited_state[i+self.norb] = False
                    excited_state[a] = True
                    excited_state[a+self.norb] = True
                    excitations.append(DoubleExcitation(
                        occ1=i, occ2=i,
                        virt1=a-n_alpha, virt2=a-n_alpha,
                        spin_case='alpha-beta',
                        bitstring=excited_state,
                        n1=n_alpha, n2=n_alpha
                    ))

        # for RHF, skip alpha-alpha and beta-beta (equivalent to alpha-beta by symmetry)
        if self.spin_type == 'RHF':
            return excitations

        # alpha-alpha double excitations
        for idx_i, i in enumerate(occupied_alpha):
            for j in occupied_alpha[idx_i + 1:]:
                for idx_a, a in enumerate(virtual_alpha):
                    for b in virtual_alpha[idx_a + 1:]:
                        excited_state = reference.copy()
                        excited_state[i] = False
                        excited_state[j] = False
                        excited_state[a] = True
                        excited_state[b] = True
                        excitations.append(DoubleExcitation(
                            occ1=i, occ2=j,
                            virt1=a - n_alpha, virt2=b - n_alpha,
                            spin_case='alpha-alpha',
                            bitstring=excited_state,
                            n1=n_alpha, n2=n_alpha
                        ))

        # beta-beta double excitations (UHF only)
        for idx_i, i in enumerate(occupied_beta):
            for j in occupied_beta[idx_i + 1:]:
                for idx_a, a in enumerate(virtual_beta):
                    for b in virtual_beta[idx_a + 1:]:
                        excited_state = reference.copy()
                        excited_state[i] = False
                        excited_state[j] = False
                        excited_state[a] = True
                        excited_state[b] = True
                        excitations.append(DoubleExcitation(
                            occ1=i - self.norb, occ2=j - self.norb,
                            virt1=a - self.norb - n_beta, virt2=b - self.norb - n_beta,
                            spin_case='beta-beta',
                            bitstring=excited_state,
                            n1=n_beta, n2=n_beta
                        ))

        return excitations

    def get_fock_matrix(self) -> np.ndarray:
        """Compute the Fock matrix in the molecular orbital basis.

        For a restricted closed-shell system (RHF), the Fock matrix is:
        F_pq = h_pq + Σ_i [2(pq|ii) - (pi|qi)]

        where the sum runs over occupied orbitals and (pq|rs) are the
        two-electron integrals in chemist notation.

        Note: Only RHF is currently supported.

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
            raise NotImplementedError("Fock matrix only implemented for RHF (spin_type='RHF')")

        return fock

    def get_g_ovvo(self) -> np.ndarray:
        """Get occupied-virtual-virtual-occupied slice of two-electron integrals.

        Returns g[i,a,b,j] where i,j are occupied indices and a,b are virtual indices.
        This slice is used in MP2-like correlation energy formulas.

        Note: Only RHF is currently supported.

        Returns:
            Array of shape (nocc, nvirt, nvirt, nocc)
        """
        if self.spin_type == 'RHF':
            nocc, _ = self.nelec
            # h2e from ao2mo.full() is stored as flat array, need to reshape
            eri = self.h2e.reshape(self.norb, self.norb, self.norb, self.norb)
            return eri[:nocc, nocc:, nocc:, :nocc]
        else:
            raise NotImplementedError("g_ovvo only implemented for RHF (spin_type='RHF')")

    def compute_correlation_energy(
        self,
        c0: complex,
        c1: Optional[SingleAmplitudes],
        c2: DoubleAmplitudes
    ) -> float:
        """Compute correlation energy from shadow-estimated excitation amplitudes.

        Uses MP2-like formula to contract amplitudes with Fock matrix and ERIs:
            E_corr = E_singles + E_doubles
            E_singles = 2 * Σ_{ia} F_{ia} * c1_{ia}
            E_doubles = [2 * Σ_{ijab} t_{ijab} * g_{iabj} - Σ_{ijab} t_{ijab} * g_{ibaj}] / c0

        Args:
            c0: HF reference amplitude (normalization factor)
            c1: Single excitation amplitudes (SingleAmplitudes object)
            c2: Double excitation amplitudes (DoubleAmplitudes object)

        Returns:
            Correlation energy contribution in Hartrees
        """
        if self.spin_type != "RHF":
            raise NotImplementedError("Correlation energy formula only implemented for RHF")

        e_singles = 0
        if c1 is not None:
            fock = self.get_fock_matrix()
            f_ov = fock[:self.nocc, self.nocc:]
            e_singles = 2.0 * np.sum(f_ov * c1.amplitudes) / c0
        
        g_ovvo = self.get_g_ovvo()
        e_doubles = (
            2.0 * np.einsum("ijab,iabj->", c2, g_ovvo)
            - np.einsum("ijab,ibaj->", c2, g_ovvo)
        ) / c0

        return e_singles + e_doubles

if __name__ == "__main__":
    pass