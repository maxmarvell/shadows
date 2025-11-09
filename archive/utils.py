
# def _qiskit_clifford_to_tableau(clifford: Clifford) -> stim.Tableau:
#     n = clifford.num_qubits
#     stabilizers = clifford.to_labels()
#     xs_qiskit = stabilizers[:n]
#     zs_qiskit = stabilizers[n:]
#     xs = [stim.PauliString(s) for s in xs_qiskit]
#     zs = [stim.PauliString(s) for s in zs_qiskit]
#     return stim.Tableau.from_conjugated_generators(xs=xs, zs=zs)
