from dataclasses import dataclass
from typing import List
import stim

import numpy as np
from numpy.typing import NDArray

@dataclass
class Bitstring():

    array: List[bool]
    endianess: str = 'little'

    def __post_init__(self):
        """Validate endianness parameter."""
        if self.endianess not in ('big', 'little'):
            raise ValueError(f"endianess must be 'big' or 'little', got '{self.endianess}'")

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
        return self.array == other.array and self.endianess == other.endianess

    def __iter__(self):
        """Allow iteration over bits."""
        return iter(self.array)

    def __len__(self):
        """Allow len() to work."""
        return len(self.array)

    def to_string(self) -> str:
        """Convert bitstring to string representation."""
        if self.endianess == 'big':
            return ''.join('1' if bit else '0' for bit in self.array)
        else:
            return ''.join('1' if bit else '0' for bit in self.array[::-1])

    def to_stabilizers(self) -> List[stim.PauliString]:
        """Convert to list of stabilizer generators."""
        out = []
        n = self.size
        for i, bit in enumerate(self.array):
            label = ['I'] * n
            label[n - 1 - i] = 'Z'             
            p = stim.PauliString(''.join(label))
            if bit:
                p *= -1                  
            out.append(p)
        return out

    def to_int(self) -> int:
        """Convert bitstring to integer.

        For both endianness: to_string() already handles reversal, so we just convert.
        """
        if self.endianess == 'little':
            value = sum((1 << i) if bit else 0 for i, bit in enumerate(self.array))
        else:
            n = len(self.array)
            value = sum((1 << (n - 1 - i)) if bit else 0 for i, bit in enumerate(self.array))
        return value

    def to_array(self) -> NDArray[np.int_]:
        """Convert to numpy array of integers (0s and 1s)."""
        return np.array([int(i) for i in self.array])

    @classmethod
    def from_string(cls, s: str, endianess: str = 'little') -> 'Bitstring':
        """Create a Bitstring from a string like '0101'.

        Args:
            s: Binary string (e.g., '0101')
            endianess: 'little' (default) or 'big'
        """
        if endianess == "little":
            bits = [b == '1' for b in s[::-1]]
        else:
            bits = [b == '1' for b in s]
        return cls(bits, endianess=endianess)

    @classmethod
    def from_int(cls, value: int, size: int, endianess: str = 'little') -> 'Bitstring':
        """Create a Bitstring from an integer with specified size.

        Args:
            value: Non-negative integer to convert
            size: Number of bits in the output
            endianess: 'big' (default) or 'little'

        For big-endian: MSB is leftmost bit (standard binary)
        For little-endian: array is stored reversed
        """
        bits = [(value >> i) & 1 for i in range(size)]
        if endianess == "little":
            array = [bool(b) for b in bits]
        else:
            array = [bool(b) for b in bits[::-1]]
        return cls(array, endianess=endianess)

    @classmethod
    def random(cls, size: int, rng=None, endianess: str = 'little') -> 'Bitstring':
        """Create a random Bitstring of given size.

        Args:
            size: Number of bits
            rng: Optional numpy random generator (for reproducibility)
            endianess: 'big' (default) or 'little'

        Returns:
            Random Bitstring
        """
        if rng is None:
            rng = np.random.default_rng()
        bits = rng.choice([True, False], size=size).tolist()
        return cls(bits, endianess=endianess)

    def copy(self) -> 'Bitstring':
        """Return a deep copy of the Bitstring."""
        return Bitstring(self.array.copy(), endianess=self.endianess)
    
    def convert_endian(self) -> 'Bitstring':
        """Convert representation so integer value is preserved across endianness flip."""
        target = 'little' if self.endianess == 'big' else 'big'
        return Bitstring(self.array[::-1], endianess=target)
    
@dataclass
class SingleExcitation:
    """Represents a single excitation with its indices and bitstring."""
    occ: int  # occupied orbital index
    virt: int  # virtual orbital index
    spin: str  # 'alpha' or 'beta'
    bitstring: Bitstring
    n: int # number of occ alpha or beta orbitals

    def __repr__(self) -> str:
        spin_symbol = 'α' if self.spin == 'alpha' else 'β'
        return f"{self.occ}{spin_symbol} → {self.virt+self.n}{spin_symbol}"

@dataclass
class DoubleExcitation:
    """Represents a double excitation with its indices and bitstring."""
    occ1: int
    occ2: int
    virt1: int
    virt2: int
    spin_case: str  # 'alpha-alpha', 'beta-beta', or 'alpha-beta'
    bitstring: Bitstring
    n1: int # number of alpha or beta orbitals 
    n2: int # number of alpha or beta orbitals

    def __repr__(self) -> str:
        if self.spin_case == 'alpha-alpha':
            return f"({self.occ1}α,{self.occ2}α) → ({self.virt1 + self.n1}α,{self.virt2 + self.n2}α)"
        elif self.spin_case == 'beta-beta':
            return f"({self.occ1}β,{self.occ2}β) → ({self.virt1 + self.n1}β,{self.virt2 + self.n2}β)"
        else:  # alpha-beta
            return f"({self.occ1}α,{self.occ2}β) → ({self.virt1 + self.n1}α,{self.virt2 + self.n2}β)"

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
            a = exc.virt

            if spin_type == "RHF":
                t1[i, a] += coeff
            else:
                raise NotImplementedError("UHF singles amplitudes not yet implemented")

        return cls(amplitudes=t1, nocc=nocc, nvirt=nvirt, spin_type=spin_type)

@dataclass
class DoubleAmplitudes:
    """Container for double excitation amplitudes in quantum chemistry convention.

    Stores amplitudes in the standard t2[i,j,a,b] format where:
    - i,j: occupied spatial orbital indices (0 to nocc-1)
    - a,b: virtual spatial orbital indices (0 to nvirt-1)

    Following antisymmetry convention: t2[i,j,a,b] = -t2[j,i,a,b] = -t2[i,j,b,a] = t2[j,i,b,a]
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
            DoubleAmplitudes object
        """
        if len(coefficients) != len(excitations):
            raise ValueError(
                f"Number of coefficients ({len(coefficients)}) doesn't match "
                f"number of excitations ({len(excitations)})"
            )

        t2 = np.zeros((nocc, nocc, nvirt, nvirt), dtype=complex)

        for coeff, exc in zip(coefficients, excitations):
            if spin_type == "RHF":

                i = exc.occ1
                j = exc.occ2
                a = exc.virt1
                b = exc.virt2

                if exc.spin_case == 'alpha-beta':
                
                    if i == j and a == b:
                        t2[i,j,a,b] += coeff
                    else:
                        t2[i,j,a,b] -= coeff
                        t2[j,i,b,a] -= coeff

                # elif exc.spin_case == 'alpha-alpha':

                #     t2[i,j,a,b] += coeff
                #     t2[j,i,a,b] -= coeff
                #     t2[i,j,b,a] -= coeff
                #     t2[j,i,b,a] += coeff

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
    interm_state = ref_state.copy()
    target_str = target_state.copy()
    n_qubits = target_str.size

    for n in range(n_qubits):
        if target_str[n] != interm_state[n]:
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
        if target_str[n] != interm_state[n]: return 0j

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
            bit = post_state[n]

            if pauli == 1: # X
                post_state[n] = not bit
            elif pauli == 2: # Y
                post_state[n] = not bit
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

def make_hydrogen_chain(n_atoms: int, bond_length: float = 0.50) -> str:
    """Generate a linear hydrogen chain with fixed interatomic distance.

    Creates a string representation of N hydrogen atoms arranged linearly
    along the z-axis with equal spacing, suitable for PySCF molecule input.

    Args:
        n_atoms: Number of hydrogen atoms in the chain (must be >= 1)
        bond_length: Interatomic distance in Angstroms (default: 0.50)

    Returns:
        String in PySCF format: "H 0 0 0; H 0 0 d; H 0 0 2d; ..."

    Examples:
        >>> make_hydrogen_chain(2, 0.50)
        'H 0 0 0; H 0 0 0.50'

        >>> make_hydrogen_chain(4, 0.74)
        'H 0 0 0; H 0 0 0.74; H 0 0 1.48; H 0 0 2.22'
    """
    if n_atoms < 1:
        raise ValueError(f"n_atoms must be >= 1, got {n_atoms}")
    if bond_length <= 0:
        raise ValueError(f"bond_length must be positive, got {bond_length}")

    atoms = []
    for i in range(n_atoms):
        z_coord = i * bond_length
        atoms.append(f"H 0 0 {z_coord:.10g}")

    return "; ".join(atoms)

if __name__ == "__main__":

    # b = Bitstring.from_string('0001', endianess='big')
    # print(b.to_int())
    # print(b)

    b = Bitstring.from_int(5, size=8, endianess='big')
    print(b.to_int())
    print(b.to_string())
    print(b)

    b = Bitstring.from_int(5, size=8, endianess='little')
    print(b.to_int())
    print(b.to_string())
    print(b)