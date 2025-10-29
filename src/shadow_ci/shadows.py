from dataclasses import dataclass
from typing import Union, List
from numpy.typing import NDArray
import numpy as np
from itertools import batched
from qiskit.quantum_info import Statevector,Clifford, Operator
from qiskit import QuantumCircuit

from abc import ABC, abstractmethod
import stim
from typing import Callable, Dict, Union, List

from shadow_ci.utils import Bitstring, gaussian_elimination, compute_x_rank, canonicalize

GateFunction = Callable[[QuantumCircuit, Union[int, List[int]]], None]
gate_map: Dict[str, GateFunction] = {
    'H': lambda qc, q: qc.h(q),
    'CNOT': lambda qc, q: qc.cx(q[0], q[1]),
    'CX': lambda qc, q: qc.cx(q[0], q[1]),
    'S': lambda qc, q: qc.s(q),
    'X': lambda qc, q: qc.x(q),
    'Y': lambda qc, q: qc.y(q),
    'Z': lambda qc, q: qc.z(q),
}

class AbstractEnsemble(ABC):

    def __init__(self, d: int):
        self.d = d

    @abstractmethod
    def generate_sample(self) -> stim.Tableau | np.ndarray:
        """Gives a sample according to the ensemble"""

class CliffordGroup(AbstractEnsemble):
    def generate_sample(self) -> stim.Tableau:
        return stim.Tableau.random(self.d)

@dataclass
class ClassicalSnapshot:
    """A single measurement sample from shadow tomography."""
    measurement: Bitstring
    unitary: Union[
        stim.Tableau,
        stim.PauliString,
        stim.Circuit,
        np.ndarray
    ]
    sampler: str

@dataclass
class ClassicalShadow:

    shadow: list[ClassicalSnapshot]
    n_qubits: int

    @property
    def N(self) -> int:
        """Total number of shadow measurements."""
        return len(self.shadow)

    def overlap(self, a: Bitstring) -> np.complex128:
        """Compute overlap <a|psi> from classical shadow snapshots.

        Args:
            a: Target bitstring (bra state)

        Returns:
            Estimated overlap between states a and b
        """
        overlaps = []

        for snapshot in self.shadow:
            if not isinstance(snapshot.unitary, stim.Tableau):
                raise NotImplementedError("Only Clifford/Tableau unitaries supported")

            tableau = snapshot.measurement.to_stabilizer()
            U_inv = snapshot.unitary.inverse()
            tableau = U_inv * tableau

            sim = stim.TableauSimulator()
            sim.set_state_from_stabilizers(tableau.to_stabilizers())
            stabilizers = canonicalize(tableau.to_stabilizers())

            # get the magnitude
            x_rank = compute_x_rank(stabilizers)
            mag = np.sqrt(2) ** (-x_rank)

            # get the phase
            measurement = sim.measure_many(*range(snapshot.measurement.size))
            bitstring = Bitstring(measurement)
            phase = gaussian_elimination(stabilizers, bitstring, a)

            # Create vacuum as a Bitstring (all zeros)
            vacuum = Bitstring([False] * self.n_qubits)
            phase *= np.conj(gaussian_elimination(stabilizers, bitstring, vacuum))

            overlaps.append(mag**2 * phase)

        return 2 * (2**self.n_qubits + 1) * np.mean(overlaps)

def _tableau_to_qiskit_clifford(tab: stim.Tableau) -> Clifford:
    """Convert a stim.Tableau to a Qiskit Clifford via symplectic generators."""
    n = len(tab)
    xouts = [str(tab.x_output(k)).replace("_", "I") for k in range(n)]
    stabs = [str(p).replace("_", "I") for p in tab.to_stabilizers()]
    return Clifford(xouts + stabs)

class ShadowProtocol:
    """Protocol for processing classical shadows and computing estimators."""

    k_estimators: List[ClassicalShadow]

    def __init__(
            self, 
            state: Statevector, 
            *, 
            ensemble_type: str = 'clifford', 
        ):
        """Initialize with a ClassicalShadow object.

        Args:
            classical_shadow: The classical shadow data to process
        """

        self.ensemble_type = ensemble_type.lower()

        if ensemble_type == "clifford":
            self.ensemble = CliffordGroup(state.num_qubits)
        elif ensemble_type == "pauli":
            raise NotImplementedError("Not yet implemented the shadow protocol sampling from the Pauli group.")
        else:
            raise ValueError(f"Unexpected ensemble {ensemble_type}, choose wither 'clifford' or 'pauli'.")

        self.state = state
        self.k_estimators = None

    def tau(self) -> Statevector:
        if self.state.data[0] != 0:
            raise RuntimeError("tau() assumes ⟨0|ψ⟩ = 0 so that |τ⟩ is a valid equal superposition.")
        vacuum = np.zeros(2**self.state.num_qubits, dtype=complex); vacuum[0] = 1.0
        tau = (vacuum + self.state.data) / np.sqrt(2)
        return Statevector(tau)
    
    def collect_samples(self, n_samples: int, n_estimators: int, *, prediction: str = 'overlap'):

        if n_samples <= 0 or n_estimators <= 0:
            raise ValueError("n_samples and n_estimators must be positive.")
        if n_samples % n_estimators != 0:
            raise ValueError("The shadow must be split into K equally sized parts.")

        if prediction != "overlap":
            raise ValueError("Only 'overlap' prediction supported currently.")

        state = self.tau()
        
        snapshots = []
        for _ in range(n_samples):

            # generate rand clifford
            tab = self.ensemble.generate_sample()

            cliff = _tableau_to_qiskit_clifford(tab)

            evolved = state.evolve(Operator(cliff))

            probs = evolved.probabilities()
            idx = np.random.choice(1 << state.num_qubits, p=probs)

            bitstring = Bitstring.from_int(idx, size=state.num_qubits)

            snapshots.append(ClassicalSnapshot(bitstring, tab, self.ensemble_type))

        classical_shadow = ClassicalShadow(snapshots, n_qubits=state.num_qubits)

        batch = n_samples // n_estimators
        self.k_estimators = [
            ClassicalShadow(list(chunk), classical_shadow.n_qubits)
            for chunk in (classical_shadow.shadow[i:i+batch] for i in range(0, n_samples, batch))
        ]
        self.prediction = prediction
        

    def estimate_overlap(self, a: Bitstring):
        if self.k_estimators is None:
            raise ValueError("Must call collect_samples before estimating anything")

        means = []
        for estimator in self.k_estimators:
            means.append(estimator.overlap(a))

        return np.median(means)


if __name__ == "__main__":
    group = CliffordGroup(20)
    for _ in range(10):
        print(group.generate_sample())
    