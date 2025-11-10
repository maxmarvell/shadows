from dataclasses import dataclass
from typing import Union, List
import numpy as np
from qiskit.quantum_info import Statevector, Clifford
import qulacs

from abc import ABC, abstractmethod
import stim
from typing import Union, List
from multiprocessing import Pool, shared_memory

import time
import warnings

from shadow_ci.utils import Bitstring, gaussian_elimination, compute_x_rank, canonicalize

try:
    from qulacs import QuantumStateGpu
    GPU_AVAILABLE = True
except (ImportError, Exception):
    GPU_AVAILABLE = False
    QuantumStateGpu = None

_SHM = None
_ARR = None

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

            stabilizers = snapshot.measurement.to_stabilizers()

            U_inv = snapshot.unitary.inverse()
            transformed_stabilizers = [U_inv(s) for s in stabilizers]
            canonical_stabilizers = canonicalize(transformed_stabilizers)

            # get the magnitude
            x_rank = compute_x_rank(canonical_stabilizers)
            mag = np.sqrt(2) ** (-x_rank)

            # get the phase
            sim = stim.TableauSimulator()
            tab = stim.Tableau.from_stabilizers(canonical_stabilizers)
            sim.do_tableau(tab, targets=list(range(self.n_qubits)))
            measurement = sim.measure_many(*range(snapshot.measurement.size))
            bitstring = Bitstring(measurement, endianess='little')
            vacuum = Bitstring([False] * self.n_qubits, endianess='little')
            phase_a = gaussian_elimination(canonical_stabilizers, bitstring, a)
            phase_0 = gaussian_elimination(canonical_stabilizers, bitstring, vacuum)

            overlaps.append(mag**2 * phase_a * phase_0.conjugate())

        return 2 * (2**self.n_qubits + 1) * np.mean(overlaps)

def _compute_single_estimator_overlap(args):
    """Compute overlap for a single K-estimator (for parallelization).

    Args:
        args: Tuple of (estimator, target_bitstring)

    Returns:
        Complex overlap value
    """
    estimator, target = args
    return estimator.overlap(target)

def _tableau_to_qiskit_clifford(tab: stim.Tableau) -> Clifford:
    """Convert a stim.Tableau to a Qiskit Clifford via symplectic generators."""
    n = len(tab)
    destabs =  [str(x).replace("_","I") for x in [tab.x_output(k) for k in range(n)]]
    stabs = [str(x).replace("_","I") for x in tab.to_stabilizers()]
    return Clifford(destabs + stabs)

def _collect_sample_mp(args) -> int:
    tab, seed = args

    n_qubits = len(tab)

    cliff = _tableau_to_qiskit_clifford(tab)

    qulacs_state = qulacs.QuantumState(n_qubits)
    qulacs_state.load(_ARR)

    circuit = _clifford_to_qulacs_circuit(cliff, n_qubits)
    circuit.update_quantum_state(qulacs_state)

    return qulacs_state.sampling(1, random_seed=seed)[0]

def _init_worker(shm_name, shape, dtype_str):
    """Attach to the shared memory from each worker"""
    global _SHM, _ARR
    _SHM = shared_memory.SharedMemory(name=shm_name)
    _ARR = np.ndarray(shape, dtype=np.dtype(dtype_str), buffer=_SHM.buf)
    _ARR.setflags(write=False)

def _clifford_to_qulacs_circuit(cliff: Clifford, n_qubits: int) -> qulacs.QuantumCircuit:
    """Convert Qiskit Clifford to Qulacs circuit."""

    import textwrap
    from qiskit import qasm2
    from qulacs.converter import convert_QASM_to_qulacs_circuit

    circuit = qulacs.QuantumCircuit(n_qubits)

    # from c.lenihan
    qasm = qasm2.dumps(cliff.to_circuit())
    qasm = textwrap.dedent(qasm).strip()
    circuit = convert_QASM_to_qulacs_circuit(qasm.splitlines())

    return circuit

class ShadowProtocol:
    """Protocol for processing classical shadows and computing estimators."""

    k_estimators: List[ClassicalShadow]

    def __init__(
            self,
            state: Statevector,
            *,
            ensemble_type: str = 'clifford',
            use_qulacs: bool = True,
            use_gpu: bool = 'auto',  # 'auto', True, False
            n_jobs: int = 1,
            verbose: int = 0,
        ):
        """Initialize with a ClassicalShadow object.

        Args:
            state: Input quantum state
            ensemble_type: Type of ensemble ('clifford' or 'pauli')
            use_qulacs: Use Qulacs for faster simulation (default: True, auto-detects availability)
            n_jobs: Number of parallel processes (1 = no parallelization)
            verbose: Verbosity level (0=silent, 1=basic, 2=detailed, 3=debug)
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
        self.use_qulacs = use_qulacs
        self.n_jobs = n_jobs
        self.verbose = verbose

        if use_gpu == 'auto':
            self.use_gpu = GPU_AVAILABLE
        elif use_gpu and not GPU_AVAILABLE:
            warnings.warn("GPU requested but qulacs-gpu not available, falling back to CPU")
            self.use_gpu = False
        else:
            self.use_gpu = use_gpu

    def tau(self) -> Statevector:
        if not np.allclose(self.state.data[0], 0):
            raise RuntimeError("tau() assumes ⟨0|ψ⟩ = 0 so that |τ⟩ is a valid equal superposition.")
        vacuum = np.zeros(2**self.state.num_qubits, dtype=complex); vacuum[0] = 1.0
        tau = (vacuum + self.state.data) / np.sqrt(2)
        return Statevector(tau)
    
    def collect_samples(self, n_samples: int, n_estimators: int, *, prediction: str = 'overlap'):
        """Collect shadow samples by generating Clifford unitaries on-the-fly.

        This method generates random Clifford tableaus one at a time within the sampling
        loop, avoiding memory issues from pre-generating all tableaus upfront. This is
        especially important for large qubit systems and many samples.

        Args:
            n_samples: Total number of shadow measurements to collect
            n_estimators: Number of median-of-means estimators (k)
            prediction: Type of prediction to enable (default: 'overlap')

        Raises:
            ValueError: If n_samples or n_estimators is non-positive, or if n_samples
                       is not evenly divisible by n_estimators
        """
        import time

        if n_samples <= 0 or n_estimators <= 0:
            raise ValueError("n_samples and n_estimators must be positive.")
        if n_samples % n_estimators != 0:
            raise ValueError("The shadow must be split into K equally sized parts.")

        if prediction != "overlap":
            raise ValueError("Only 'overlap' prediction supported currently.")

        state = self.tau()
        n_qubits = state.num_qubits

        if self.verbose >= 2:
            print(f"  [ShadowProtocol] Collecting {n_samples:,} samples ({n_qubits} qubits)")
            print(f"  - Backend: {'Qulacs' if self.use_qulacs else 'Qiskit'}")
            print(f"  - Parallelization: {self.n_jobs} worker(s)")


        t_start = time.perf_counter()
        if self.n_jobs > 1 and not self.use_qulacs and state.num_qubits <= 8: # qulacs conflict parallelization deadlocks resources
            snapshots = self._collect_samples_parallel(state, n_samples)
        else:
            snapshots = self._collect_samples_serial(state, n_qubits, n_samples)
        t_elapsed = time.perf_counter() - t_start

        if self.verbose >= 1:
            throughput = n_samples / t_elapsed if t_elapsed > 0 else 0
            print(f"  [ShadowProtocol] Collected {n_samples:,} samples in {t_elapsed:.2f} s ({throughput:.0f} samples/s)")

        batch_size = n_samples // n_estimators
        self.k_estimators = [
            ClassicalShadow(snapshots[i:i+batch_size], n_qubits)
            for i in range(0, n_samples, batch_size)
        ]
        self.prediction = prediction

    def _collect_samples_serial(self, state: Statevector, n_qubits: int, n_samples: int) -> List[ClassicalSnapshot]:
        """Collect samples serially, generating tableaus on-the-fly.

        Args:
            state: Quantum state to measure
            n_qubits: Number of qubits
            n_samples: Number of samples to collect

        Returns:
            List of ClassicalSnapshot objects
        """

        snapshots = []

        if self.verbose >= 3:
            print(f"    [Serial Sampling] Starting collection of {n_samples} samples...")
            time_tableau_gen = 0.0
            time_sampling = 0.0

        for i in range(n_samples):
            if self.verbose >= 3:
                t0 = time.perf_counter()
            tab = self.ensemble.generate_sample()

            if self.verbose >= 3:
                time_tableau_gen += time.perf_counter() - t0

            if self.verbose >= 3:
                t0 = time.perf_counter()

            cliff = _tableau_to_qiskit_clifford(tab)
            if self.use_qulacs:
                qulacs_state = qulacs.QuantumState(n_qubits)
                qulacs_state.load(state)
                circuit = _clifford_to_qulacs_circuit(cliff, n_qubits)
                circuit.update_quantum_state(qulacs_state)
                sample = qulacs_state.sampling(1)[0]
                bitstring = Bitstring.from_int(sample, size=n_qubits, endianess='little')

            else:
                evolved = state.evolve(cliff)
                evolved.seed(np.random.randint(1e9))
                sample = evolved.sample_memory(shots=1)[0]
                bitstring = Bitstring.from_string(sample, endianess='little')

            if self.verbose >= 3:
                time_sampling += time.perf_counter() - t0

            snapshots.append(ClassicalSnapshot(bitstring, tab, self.ensemble_type))

            # Progress reporting
            if self.verbose >= 3 and (i + 1) % max(1, n_samples // 10) == 0:
                progress = (i + 1) / n_samples * 100
                print(f"      Progress: {i+1}/{n_samples} ({progress:.0f}%)")

        if self.verbose >= 3:
            print(f"    [Serial Sampling] Breakdown:")
            print(f"      - Tableau generation: {time_tableau_gen:.2f} s ({time_tableau_gen/n_samples*1000:.1f} ms/sample)")
            print(f"      - State evolution + sampling: {time_sampling:.2f} s ({time_sampling/n_samples*1000:.1f} ms/sample)")

        return snapshots

    def _collect_samples_parallel(self, state: Statevector, n_samples: int) -> List[ClassicalSnapshot]:
        """Collect samples in parallel by distributing tableaus to workers.

        Strategy:
        1. Generate random tableaus (cheap, fast)
        2. Distribute tableau batches to workers
        3. Workers convert tableaus → circuits locally and sample

        Workers do conversion locally since it's only ~1-2% overhead compared
        to circuit application, avoiding the memory cost of precomputing circuits.

        Args:
            state: Quantum state to measure
            n_samples: Number of samples to collect

        Returns:
            List of ClassicalSnapshot objects
        """

        if self.verbose >= 2:
            print(f"    [Parallel Sampling] Step 1: Generating {n_samples} random tableaus...")
            t0 = time.perf_counter()

        tableaus = [self.ensemble.generate_sample() for _ in range(n_samples)]

        if self.verbose >= 2:
            t_gen = time.perf_counter() - t0
            print(f"    [Parallel Sampling] Generated tableaus in {t_gen:.2f} s")
            print(f"    [Parallel Sampling] Step 2: Distributing tableaus to workers for sampling...")
            t0 = time.perf_counter()

        shm = shared_memory.SharedMemory(create=True, size=state.data.nbytes)
        shm_arr = np.ndarray(state.data.shape, dtype=state.data.dtype, buffer=shm.buf)
        shm_arr[:] = state.data

        args_list = [(tab, i+1) for i, tab in enumerate(tableaus)]
        
        try:
            with Pool(processes=self.n_jobs, initializer=_init_worker, initargs=(shm.name, state.data.shape, str(state.data.dtype))) as pool:
                bitstrings = pool.map(_collect_sample_mp, args_list)
        finally:
            shm.close()
            shm.unlink()   

        results = [(Bitstring.from_int(b, size=state.num_qubits), t) for b, t in zip(bitstrings, tableaus)]

        if self.verbose >= 2:
            t_sample = time.perf_counter() - t0
            print(f"    [Parallel Sampling] Sampled in {t_sample:.2f} s ({t_sample/n_samples*1000:.1f} ms/sample)")

        snapshots = [ClassicalSnapshot(bitstring, tab, self.ensemble_type)
                     for bitstring, tab in results]

        return snapshots

    def estimate_overlap(self, a: Bitstring):
        """Estimate overlap between shadow state and target bitstring.

        Parallelizes computation across K estimators when n_jobs > 1.

        Args:
            a: Target bitstring for overlap computation

        Returns:
            Median-of-means estimate of overlap <a|ψ>
        """
        import time

        if self.k_estimators is None:
            raise ValueError("Must call collect_samples before estimating anything")

        if self.verbose >= 3:
            t_start = time.perf_counter()

        if self.n_jobs > 1 and len(self.k_estimators) > 1:
            args_list = [(estimator, a) for estimator in self.k_estimators]
            with Pool(processes=self.n_jobs) as pool:
                means = pool.map(_compute_single_estimator_overlap, args_list)
        else:
            means = []
            for estimator in self.k_estimators:
                overlap_val = estimator.overlap(a)
                means.append(overlap_val)

        result = np.median(means)

        if self.verbose >= 3:
            t_elapsed = time.perf_counter() - t_start
            mode = "parallel" if self.n_jobs > 1 and len(self.k_estimators) > 1 else "serial"
            print(f"    [Overlap Estimation] Computed overlap in {t_elapsed*1000:.2f} ms ({len(self.k_estimators)} estimators, {mode})")

        return result


if __name__ == "__main__":
    pass
    