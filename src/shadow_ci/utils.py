from dataclasses import dataclass
from typing import List
import stim

import numpy as np

@dataclass
class Bitstring():
    array: List[bool]

    @property
    def size(self) -> int:
        return len(self.array)

    def __getitem__(self, key):
        """Allow indexing as array[i]"""
        return self.array[key]

    def __setitem__(self, key, value):
        """Allow item assignment as array[i] = value"""
        self.array[key] = value

    def __eq__(self, other):
        """Check equality with another Bitstring."""
        if not isinstance(other, Bitstring):
            return False
        return self.array == other.array

    def __iter__(self):
        """Allow iteration over bits."""
        return iter(self.array)

    def __len__(self):
        """Allow len() to work."""
        return len(self.array)

    def to_string(self) -> str:
        return ''.join('1' if bit else '0' for bit in self.array)

    def to_stabilizer(self) -> stim.Tableau:
        n_qubits = len(self.array)
        tableau = stim.Tableau(n_qubits)

        for i, bit in enumerate(self.array):
            if bit:
                x_gate = stim.Tableau.from_named_gate("X")
                expanded_gate = stim.Tableau(n_qubits)
                expanded_gate.prepend(x_gate, [n_qubits - 1 - i])
                tableau = tableau.then(expanded_gate)

        return tableau

    def to_int(self) -> int:
        return int(self.to_string(), 2)

    @classmethod
    def from_string(cls, s: str) -> 'Bitstring':
        """Create a Bitstring from a string like '0101'."""
        return cls([c == '1' for c in s])

    @classmethod
    def from_int(cls, value: int, size: int) -> 'Bitstring':
        """Create a Bitstring from an integer with specified size."""
        binary = format(value, f'0{size}b')
        return cls.from_string(binary)

    @classmethod
    def random(cls, size: int, rng=None) -> 'Bitstring':
        """Create a random Bitstring of given size.

        Args:
            size: Number of bits
            rng: Optional numpy random generator (for reproducibility)

        Returns:
            Random Bitstring
        """
        if rng is None:
            rng = np.random.default_rng()
        bits = rng.choice([True, False], size=size).tolist()
        return cls(bits)
    
@dataclass
class SingleExcitation:
    """Represents a single excitation with its indices and bitstring."""
    occ: int  # occupied orbital index
    virt: int  # virtual orbital index
    spin: str  # 'alpha' or 'beta'
    bitstring: Bitstring

    def __repr__(self) -> str:
        spin_symbol = 'α' if self.spin == 'alpha' else 'β'
        return f"{self.occ}{spin_symbol} → {self.virt}{spin_symbol}"

@dataclass
class DoubleExcitation:
    """Represents a double excitation with its indices and bitstring."""
    occ1: int
    occ2: int
    virt1: int
    virt2: int
    spin_case: str  # 'alpha-alpha', 'beta-beta', or 'alpha-beta'
    bitstring: Bitstring

    def __repr__(self) -> str:
        if self.spin_case == 'alpha-alpha':
            return f"({self.occ1}α,{self.occ2}α) → ({self.virt1}α,{self.virt2}α)"
        elif self.spin_case == 'beta-beta':
            return f"({self.occ1}β,{self.occ2}β) → ({self.virt1}β,{self.virt2}β)"
        else:  # alpha-beta
            return f"({self.occ1}α,{self.occ2}β) → ({self.virt1}α,{self.virt2}β)"

@dataclass
class SingleAmplitudes:
    """Container for single excitation amplitudes in quantum chemistry convention.

    Stores amplitudes in the standard t1[i,a] format where:
    - i: occupied spatial orbital index (0 to nocc-1)
    - a: virtual spatial orbital index (0 to nvirt-1)

    For RHF, both alpha and beta have the same spatial amplitudes.
    For UHF, alpha and beta can differ.
    """
    amplitudes: np.ndarray  # shape: (nocc, nvirt) for RHF
    nocc: int
    nvirt: int
    spin_type: str = "RHF"

    def __post_init__(self):
        expected_shape = (self.nocc, self.nvirt)
        if self.amplitudes.shape != expected_shape:
            raise ValueError(
                f"Amplitudes shape {self.amplitudes.shape} doesn't match "
                f"expected shape {expected_shape} for nocc={self.nocc}, nvirt={self.nvirt}"
            )

    def __getitem__(self, key):
        """Allow indexing as t1[i, a]"""
        return self.amplitudes[key]

    def __repr__(self) -> str:
        return f"SingleAmplitudes(shape={self.amplitudes.shape}, nocc={self.nocc}, nvirt={self.nvirt})"

    @classmethod
    def from_excitation_list(cls, coefficients: np.ndarray,
                            excitations: List[SingleExcitation],
                            nocc: int, nvirt: int,
                            spin_type: str = "RHF") -> 'SingleAmplitudes':
        """Create SingleAmplitudes from flat coefficient array and excitation list.

        Args:
            coefficients: Flat array of coefficients for each excitation
            excitations: List of SingleExcitation objects defining the ordering
            nocc: Number of occupied spatial orbitals
            nvirt: Number of virtual spatial orbitals
            spin_type: "RHF" or "UHF"

        Returns:
            SingleAmplitudes object with properly shaped tensor
        """
        if len(coefficients) != len(excitations):
            raise ValueError(
                f"Number of coefficients ({len(coefficients)}) doesn't match "
                f"number of excitations ({len(excitations)})"
            )

        t1 = np.zeros((nocc, nvirt), dtype=complex)

        for coeff, exc in zip(coefficients, excitations):
            i = exc.occ
            a = exc.virt - nocc

            if spin_type == "RHF":
                t1[i, a] += coeff
            else:
                # for UHF would need separate alpha/beta tensors
                raise NotImplementedError("UHF singles amplitudes not yet implemented")

        if spin_type == "RHF":
            t1 /= 2.0

        return cls(amplitudes=t1, nocc=nocc, nvirt=nvirt, spin_type=spin_type)

@dataclass
class DoubleAmplitudes:
    """Container for double excitation amplitudes in quantum chemistry convention.

    Stores amplitudes in the standard t2[i,j,a,b] format where:
    - i,j: occupied spatial orbital indices (0 to nocc-1)
    - a,b: virtual spatial orbital indices (0 to nvirt-1)

    Following antisymmetry convention: t2[i,j,a,b] = -t2[j,i,a,b] = -t2[i,j,b,a]
    """
    amplitudes: np.ndarray  # shape: (nocc, nocc, nvirt, nvirt) for RHF
    nocc: int
    nvirt: int
    spin_type: str = "RHF"

    def __post_init__(self):
        expected_shape = (self.nocc, self.nocc, self.nvirt, self.nvirt)
        if self.amplitudes.shape != expected_shape:
            raise ValueError(
                f"Amplitudes shape {self.amplitudes.shape} doesn't match "
                f"expected shape {expected_shape} for nocc={self.nocc}, nvirt={self.nvirt}"
            )

    def __getitem__(self, key):
        """Allow indexing as t2[i, j, a, b]"""
        return self.amplitudes[key]

    def __repr__(self) -> str:
        return f"DoubleAmplitudes(shape={self.amplitudes.shape}, nocc={self.nocc}, nvirt={self.nvirt})"

    @classmethod
    def from_excitation_list(cls, coefficients: np.ndarray,
                            excitations: List[DoubleExcitation],
                            nocc: int, nvirt: int,
                            spin_type: str = "RHF") -> 'DoubleAmplitudes':
        """Create DoubleAmplitudes from flat coefficient array and excitation list.

        Args:
            coefficients: Flat array of coefficients for each excitation
            excitations: List of DoubleExcitation objects defining the ordering
            nocc: Number of occupied spatial orbitals
            nvirt: Number of virtual spatial orbitals
            spin_type: "RHF" or "UHF"

        Returns:
            DoubleAmplitudes object with properly shaped tensor
        """
        if len(coefficients) != len(excitations):
            raise ValueError(
                f"Number of coefficients ({len(coefficients)}) doesn't match "
                f"number of excitations ({len(excitations)})"
            )

        # Initialize amplitude tensor
        t2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=complex)

        # Map coefficients to tensor indices
        for coeff, exc in zip(coefficients, excitations):
            i = exc.occ1
            j = exc.occ2
            a = exc.virt1 - nocc  # virtual index (relative to nocc)
            b = exc.virt2 - nocc

            if spin_type == "RHF":
                # For RHF spatial orbitals, need to account for spin cases
                if exc.spin_case == "alpha-alpha":
                    # α-α excitation contributes to spatial t2[i,j,a,b]
                    t2[i, j, a, b] += coeff
                elif exc.spin_case == "beta-beta":
                    # β-β excitation contributes identically to spatial t2[i,j,a,b]
                    t2[i, j, a, b] += coeff
                elif exc.spin_case == "alpha-beta":
                    # α-β mixed spin excitation contributes to spatial t2[i,j,a,b]
                    # In spatial orbital formalism: accounts for different spin interactions
                    t2[i, j, a, b] += coeff
            else:
                raise NotImplementedError("UHF doubles amplitudes not yet implemented")

        return cls(amplitudes=t2, nocc=nocc, nvirt=nvirt, spin_type=spin_type)

def gaussian_elimination(
    stabilizers: List[stim.PauliString],
        ref_state: Bitstring,
        target_state: Bitstring
    ) -> complex:
    """Compute the phase of overlap between a stabilizer and a target basis state.

    If no overlap simply returns 0 + 0j

    Args:
        stabilizers: List of canonical stabilizers (reduced stabilizer matrix)
        ref_state: Reference computational basis state (measurement outcome)
        target_state: Target computational basis state for overlap computation

    Returns:
        Complex phase factor for the overlap
    """
    phase = 1 + 0j
    interm_state = ref_state
    target_str = target_state
    n_qubits = target_str.size

    for n in range(n_qubits):
        if target_str[n_qubits-1-n] != interm_state[n_qubits-1-n]:
            for m in range(len(stabilizers)):
                stabilizer = stabilizers[m]
                x_bits, _ = stabilizer.to_numpy()

                if not x_bits[n]: continue

                all_left_false = True
                for k in range(n):
                    if x_bits[k]:
                        all_left_false = False
                        break

                if all_left_false:
                    interm_state, phase = apply_stabilizer_to_state(interm_state, stabilizer, phase)
                    break

        # no overlap
        if target_str[n_qubits-1-n] != interm_state[n_qubits-1-n]: return 0j

    return phase

def apply_stabilizer_to_state(
        state: Bitstring,
        stabilizer: stim.PauliString,
        phase: complex
    ) -> tuple[Bitstring, complex]:
        """Apply a Pauli string (stabilizer) to a computational basis state.

        Args:
            state: Computational basis state as Bitstring object
            stabilizer: Pauli string to apply
            phase: Accumulated phase factor

        Returns:
            Tuple of (new_state, new_phase)
        """
        n_qubits = state.size
        post_state = state

        # Apply the stabilizer's sign/phase
        new_phase = phase * stabilizer.sign

        # Apply each Pauli operator
        for n in range(n_qubits):
            pauli = stabilizer[n]
            bit = post_state[n_qubits-n-1]

            if pauli == 1: # X
                post_state[n_qubits-n-1] = not bit
            elif pauli == 2: # Y
                post_state[n_qubits-n-1] = not bit
                new_phase = new_phase * (1j) * (-1) ** (int(bit))
            elif pauli == 3: # Z
                new_phase = new_phase * (-1) ** (int(bit))
            elif pauli != 0: # I
                raise ValueError(f"Unrecognized pauli operator {pauli}.")

        return post_state, new_phase

def compute_x_rank(canonical_stabilizers: List[stim.PauliString]) -> int:
    """Compute the X-rank given a set of canonical stabilizers."""
    x_rank = 0
    for stab in canonical_stabilizers:
        x_bits, _ = stab.to_numpy()
        for bit in x_bits:
            if bit: x_rank += 1; break
    return x_rank

def canonicalize(stabilizers: List[stim.PauliString]) -> List[stim.PauliString]:
    """Convert stim stabilizers to canonical form using Gaussian elimination.

    Takes a list of stim.PauliString stabilizers and reduces them to
    canonical row echelon form following the algorithm from
    https://arxiv.org/pdf/1711.07848

    Args:
        stabilizers: List of stim.PauliString objects representing stabilizers

    Returns:
        Canonical stabilizer matrix where each row is [pauli_ops..., phase]
        Example: [['X', 'I', 'I', 1], ['I', 'Z', 'I', -1]]

    Example:
        >>> # For Bell state |Φ⁺⟩ with stabilizers XX and ZZ
        >>> stabs = [stim.PauliString("XX"), stim.PauliString("ZZ")]
        >>> canonical = canonicalize(stabs)
        >>> # Returns: [['X', 'X', 1], ['Z', 'Z', 1]]
    """

    def rowswap(i: int, j: int):
        canonicalized[i], canonicalized[j] = canonicalized[j], canonicalized[i]

    def rowmult(i: int, j: int):
        canonicalized[j] = canonicalized[i] * canonicalized[j] 

    canonicalized = [s.copy() for s in stabilizers]
    nq = len(canonicalized[0])      # number of qubits = length of PauliString
    nr = len(canonicalized)         # number of generators (rows)

    # X-block
    i = 0
    for j in range(nq):
        k = next((k for k in range(i, nr) if canonicalized[k][j] in {1, 2}), None)
        if k is not None:
            rowswap(i, k)
            for m in range(nr):
                if m != i and canonicalized[m][j] in {1, 2}:
                    rowmult(i, m)
            i += 1

    # Z-block
    for j in range(nq):
        k = next((k for k in range(i, nr) if canonicalized[k][j] == 3), None)
        if k is not None:
            rowswap(i, k)
            for m in range(nr):
                if m != i and canonicalized[m][j] in {2, 3}:
                    rowmult(i, m)
            i += 1


    return canonicalized

if __name__ == "__main__":

    stabilizers = [
        stim.PauliString('-ZYYX'),
        stim.PauliString('ZXXI'),
        stim.PauliString('-ZIXZ'),
        stim.PauliString('-YYXX')
    ]
    U = stim.Tableau.from_stabilizers(stabilizers)
    b = Bitstring([True, False, True, True])

    # try composing tableaus directly
    b_tab = b.to_stabilizer()
    # tableau = b_tab.then(U.inverse())
    tableau = U.inverse() * b_tab

    print('='*50)
    print('Tableau from running U_inv * |b>')
    print('='*50)
    print(tableau.to_stabilizers())

    # import qiskit
    # from qiskit.quantum_info import Clifford, StabilizerState

    # # Build the same Clifford in Qiskit from Stim's tableau U
    # symp_mat = (
    #     [str(U.x_output(k)).replace("_", "I") for k in range(len(U))] +
    #     [str(s).replace("_", "I") for s in U.to_stabilizers()]
    # )
    # rand_clifford = Clifford(symp_mat)

    # # Correct encoding of the computational basis state |b>
    # # For each qubit i, generator is Z_i with sign (-) if b[i]==1 else (+)
    # n = b.size
    # labels = []
    # for i, bit in enumerate(b):
    #     s = ['I'] * n
    #     s[i] = 'Z'
    #     labels.append(('-' if bit else '+') + ''.join(s))

    # stab_data = StabilizerState.from_stabilizer_list(labels)     # this is |b>
    # print(stab_data)
    # stab_data = stab_data.evolve(rand_clifford)  # apply U^\dagger

    # # Extract stabilizer generators (sign in column 0)
    # stab_matrix = []
    # for lab in stab_data.clifford.to_labels(mode="S"):
    #     row = list(lab[1:]) + ([1] if lab[0] == '+' else [-1])
    #     stab_matrix.append(row)

    # print('='*50)
    # print('Tableau from running Qiskit (correct |b> encoding)')
    # print('='*50)
    # print(stab_matrix)

    
    # b_tab = b.to_stabilizer()
    # tableau = U * b_tab

    # print(tableau.to_stabilizers())

    # sim = stim.TableauSimulator()
    # sim.set_state_from_stabilizers(tableau.to_stabilizers())
    # stabilizers = canonicalize(tableau.to_stabilizers())

    # print(stabilizers)

    print('='*50)
    print('CHECKING CANONICALIZATION FORMULA')
    print('='*50)

    # CANONICALIZATION IS CORRECT AND MATCHES PAPER RESULT
    stabilizers = [
        stim.PauliString('-XZIY'),
        stim.PauliString('YXYY'),
        stim.PauliString('XIXY'),
        stim.PauliString('-YZIZ')
    ]
    canonical = canonicalize(stabilizers)
    print(canonical)