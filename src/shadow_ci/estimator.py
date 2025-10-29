from shadow_ci.chemistry import MolecularHamiltonian
from shadow_ci.utils import SingleAmplitudes, DoubleAmplitudes
from shadow_ci.solvers import VQESolver
from shadow_ci.shadows import CliffordGroup, ClassicalSnapshot, ClassicalShadow, ShadowProtocol
from qiskit.quantum_info import Statevector
from qiskit import QuantumCircuit
from typing import Callable, Dict, Union, List
from shadow_ci.utils import Bitstring

import numpy as np

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

class GroundStateEstimator:
    """
    Use the 'mixed' energy estimator to approximate corrections to the ground state wavefunction
    of a HF mean-field Hamiltonian.
    """

    def __init__(self, hamiltonian: MolecularHamiltonian, solver: VQESolver):

        # use the solver to get the active quantum state
        self.trial, _ = solver.solve()
        self.hamiltonian = hamiltonian
        self.n_qubits = 2 * hamiltonian.norb
        

    def estimate_ground_state(self, n_samples: int, n_k_estimators: int):

        protocol = ShadowProtocol(self.trial)
        protocol.collect_samples(n_samples, n_k_estimators, prediction='overlap')

        c0 = self.estimate_c0(protocol)
        c1 = self.estimate_first_order_interactions(protocol)
        c2 = self.estimate_second_order_interaction(protocol)

        # Get integrals
        n_alpha, _ = self.hamiltonian.nelec
        nocc = n_alpha
        g_ovvo = self.hamiltonian.get_g_ovvo()  # shape: (nocc, nvirt, nvirt, nocc)
        fock = self.hamiltonian.get_fock_matrix()  # shape: (norb, norb)
        f_ov = fock[:nocc, nocc:]  # Extract occupied-virtual block (nocc, nvirt)

        # Now c1.amplitudes has shape (nocc, nvirt) and c2.amplitudes has shape (nocc, nvirt) and c2.amplitudes has shape (nocc, nocc, nvirt, nvirt)
        # These match the required shapes for the contractions
        # Take the real part to ensure energy is real (imaginary parts are statistical noise)
        e_singles = 2 * np.sum(f_ov * c1.amplitudes)
        e_doubles = 2 * (np.einsum("ijab,iabj->", c2.amplitudes, g_ovvo) / c0
                    - np.einsum("ijab,ibaj->", c2.amplitudes, g_ovvo)) / c0

        return e_singles + e_doubles, c0, c1, c2

    def estimate_c0(self, protocol: ShadowProtocol) -> np.complex128:
        psi0 = self.hamiltonian.get_hf_bitstring()
        return protocol.estimate_overlap(psi0)
    
    def estimate_first_order_interactions(self, protocol: ShadowProtocol) -> SingleAmplitudes:
        """Estimate single excitation amplitudes and return in tensor form.

        Returns:
            SingleAmplitudes: Amplitudes in t1[i,a] format (nocc, nvirt)
        """
        excitations = self.hamiltonian.get_single_excitations()
        coeffs = np.empty(len(excitations), dtype=complex)
        for i, ex in enumerate(excitations):
            coeffs[i] = protocol.estimate_overlap(ex.bitstring)

        # Convert to proper tensor format
        n_alpha, _ = self.hamiltonian.nelec
        nocc = n_alpha
        nvirt = self.hamiltonian.norb - nocc

        return SingleAmplitudes.from_excitation_list(
            coeffs, excitations, nocc, nvirt, self.hamiltonian.spin_type
        )

    def estimate_second_order_interaction(self, protocol: ShadowProtocol) -> DoubleAmplitudes:
        """Estimate double excitation amplitudes and return in tensor form.

        Returns:
            DoubleAmplitudes: Amplitudes in t2[i,j,a,b] format (nocc, nocc, nvirt, nvirt)
        """
        excitations = self.hamiltonian.get_double_excitations()
        coeffs = np.empty(len(excitations), dtype=complex)
        for i, ex in enumerate(excitations):
            coeffs[i] = protocol.estimate_overlap(ex.bitstring)

        # Convert to proper tensor format
        n_alpha, _ = self.hamiltonian.nelec
        nocc = n_alpha
        nvirt = self.hamiltonian.norb - nocc

        return DoubleAmplitudes.from_excitation_list(
            coeffs, excitations, nocc, nvirt, self.hamiltonian.spin_type
        )

if __name__ == "__main__":

    n_qubits = 4
    ensemble = CliffordGroup(n_qubits)
    state = Statevector.from_label("0101")

    clifford = ensemble.generate_sample()
    circuit = clifford.to_circuit()
    qc = QuantumCircuit(n_qubits)

    for instruction in circuit:
        gate_name = instruction.name
        targets = [t.value for t in instruction.targets_copy()]

        if gate_name in gate_map:
            gate_map[gate_name](qc, targets if len(targets) > 1 else targets[0])

    res, _ = state.evolve(qc).measure()