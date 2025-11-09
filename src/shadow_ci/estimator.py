from shadow_ci.hamiltonian import MolecularHamiltonian
from shadow_ci.utils import SingleAmplitudes, DoubleExcitation, SingleExcitation
from shadow_ci.solvers import GroundStateSolver
from shadow_ci.shadows import ShadowProtocol

from pyscf.ci.cisd import tn_addrs_signs
import numpy as np
import time

from typing import List, Tuple

class GroundStateEstimator:
    """Ground state energy estimator using shadow tomography.

    Uses the 'mixed' energy estimator to approximate corrections to the ground state
    wavefunction of a HF mean-field Hamiltonian via classical shadow tomography.

    Args:
        hamiltonian: Molecular Hamiltonian object
        solver: Ground state solver (e.g., FCISolver, VQESolver)
        verbose: Verbosity level (0=silent, 1=basic, 2=detailed, 3=debug)
    """

    def __init__(self, hamiltonian: MolecularHamiltonian, solver: GroundStateSolver, verbose: int = 0):
        self.trial, self.E_exact = solver.solve()
        self.E_hf = hamiltonian.hf_energy
        self.hamiltonian = hamiltonian
        self.n_qubits = 2 * hamiltonian.norb
        self.verbose = verbose

        if self.verbose >= 2:
            print(f"[GroundStateEstimator] Initialized:")
            print(f"  - System: {self.n_qubits} qubits ({hamiltonian.norb} orbitals)")
            print(f"  - HF Energy: {self.E_hf:.8f} Ha")
            print(f"  - FCI Energy: {self.E_exact:.8f} Ha")
            print(f"  - Correlation Energy: {self.E_exact - self.E_hf:.8f} Ha")

    def estimate_ground_state(self, n_samples: int, 
                              n_k_estimators: int, 
                              *, n_jobs: int = 1, 
                              use_qualcs: bool = True,
                              calc_c1: bool = False):
        """Estimate ground state energy and excitation amplitudes via shadow tomography.

        Args:
            n_samples: Total number of shadow measurements
            n_k_estimators: Number of median-of-means estimators
            n_jobs: Number of parallel workers (default: 1)
            use_qualcs: Use Qulacs backend for faster sampling (default: True)

        Returns:
            Tuple of (energy, c0_overlap, singles_amplitudes, doubles_amplitudes)
        """
        if self.verbose >= 2:
            print(f"\n{'=' * 70}")
            print(f"Shadow Tomography Ground State Estimation")
            print(f"{'=' * 70}")
            print(f"Configuration:")
            print(f"  - Shadow samples: {n_samples:,}")
            print(f"  - Median-of-means bins: {n_k_estimators}")
            print(f"  - Parallel workers: {n_jobs}")
            print(f"  - Backend: {'Qulacs' if use_qualcs else 'Qiskit'}")

        # Phase 1: Collect shadow samples
        if self.verbose >= 1:
            print(f"\n[Phase 1/4] Collecting shadow samples...")
            t_start = time.perf_counter()

        protocol = ShadowProtocol(self.trial, n_jobs=n_jobs, use_qulacs=use_qualcs, verbose=self.verbose-1)
        protocol.collect_samples(n_samples, n_k_estimators, prediction='overlap')

        if self.verbose >= 1:
            t_elapsed = time.perf_counter() - t_start
            throughput = n_samples / t_elapsed
            print(f"  ✓ Collected {n_samples:,} samples in {t_elapsed:.2f} s ({throughput:.0f} samples/s)")

        # Phase 2: Estimate HF overlap (c0)
        if self.verbose >= 1:
            print(f"\n[Phase 2/4] Estimating HF reference overlap (c0)...")
            t_start = time.perf_counter()

        c0 = self.estimate_c0(protocol)

        if calc_c1:
            t_elapsed = time.perf_counter() - t_start
            print(f"  ✓ c0 = {c0:.6f} (computed in {t_elapsed:.3f} s)")


            if self.verbose >= 1:
                n_singles = len(self.hamiltonian.get_single_excitations())
                print(f"\n[Phase 3/4] Estimating single excitation amplitudes (c1)...")
                print(f"  - Number of single excitations: {n_singles}")
                t_start = time.perf_counter()

            c1 = self.estimate_first_order_interactions(protocol)

            if self.verbose >= 1:
                t_elapsed = time.perf_counter() - t_start
                avg_time = t_elapsed / n_singles if n_singles > 0 else 0
                print(f"  ✓ Estimated {n_singles} amplitudes in {t_elapsed:.2f} s ({avg_time*1000:.1f} ms each)")
        else:
            c1 = None

        if self.verbose >= 1:
            n_doubles = len(self.hamiltonian.get_double_excitations())
            print(f"\n[Phase 4/4] Estimating double excitation amplitudes (c2)...")
            print(f"  - Number of double excitations: {n_doubles}")
            t_start = time.perf_counter()

        c2 = self.estimate_second_order_interaction(protocol)

        if self.verbose >= 1:
            t_elapsed = time.perf_counter() - t_start
            avg_time = t_elapsed / n_doubles if n_doubles > 0 else 0
            print(f"  ✓ Estimated {n_doubles} amplitudes in {t_elapsed:.2f} s ({avg_time*1000:.1f} ms each)")

        if self.verbose >= 2:
            print(f"\n[Computing correlation energy]...")

        e_corr = self.hamiltonian.compute_correlation_energy(c0, c1, c2)
        e_total = self.E_hf + e_corr

        if self.verbose >= 1:
            print(f"\n{'=' * 70}")
            print(f"Results:")
            print(f"  - HF Energy:          {self.E_hf:.8f} Ha")
            print(f"  - Correlation Energy: {e_corr:.8f} Ha")
            print(f"  - Total Energy:       {e_total:.8f} Ha")
            if hasattr(self, 'E_exact'):
                error = e_total - self.E_exact
                print(f"  - Exact FCI Energy:   {self.E_exact:.8f} Ha")
                print(f"  - Error:              {error:+.2e} Ha")
            print(f"{'=' * 70}\n")

        return e_total, c0, c1, c2

    def estimate_c0(self, protocol: ShadowProtocol) -> np.complex128:
        psi0 = self.hamiltonian.get_hf_bitstring()
        overlap = protocol.estimate_overlap(psi0)
        return overlap
    
    def estimate_first_order_interactions(self, protocol: ShadowProtocol) -> np.ndarray:
        """Estimate single excitation amplitudes and return in tensor form.

        For RHF systems, only unique excitations are measured (alpha only),
        reducing the number of shadow measurements by a factor of 2.

        Returns:
            SingleAmplitudes: Amplitudes in t1[i,a] format (nocc, nvirt)
        """

        excitations = self.hamiltonian.get_single_excitations()
        n_exc = len(excitations)

        _, t1sign = tn_addrs_signs(self.hamiltonian.norb, self.hamiltonian.nocc, 1)

        coeffs = np.empty(n_exc, dtype=complex)
        for i, ex in enumerate(excitations):
            coeffs[i] = protocol.estimate_overlap(ex.bitstring)

            if self.verbose >= 1 and (i + 1) % max(1, n_exc // 10) == 0:
                progress = (i + 1) / n_exc * 100
                print(f"    Progress: {i+1}/{n_exc} ({progress:.0f}%)")

        # t1 = self._construct_singles_tensor(coeffs, excitations)

        return (coeffs * t1sign).reshape(self.hamiltonian.nocc, self.hamiltonian.nvirt)

    def estimate_second_order_interaction(self, protocol: ShadowProtocol) -> np.ndarray:
        """Estimate double excitation amplitudes and return in tensor form.

        Returns:
            DoubleAmplitudes: Amplitudes in t2[i,j,a,b] format (nocc, nocc, nvirt, nvirt)
        """

        excitations = self.hamiltonian.get_double_excitations()
        n_exc = len(excitations)

        coeffs = np.empty(n_exc, dtype=complex)
        for i, ex in enumerate(excitations):
            coeffs[i] = protocol.estimate_overlap(ex.bitstring)

            # Progress reporting for verbose mode
            if self.verbose >= 1 and (i + 1) % max(1, n_exc // 10) == 0:
                progress = (i + 1) / n_exc * 100
                print(f"    Progress: {i+1}/{n_exc} ({progress:.0f}%)")

        return self._construct_doubles_tensor(coeffs, excitations)

    def _construct_singles_tensor(self, coefficients: np.ndarray, excitations: List[SingleExcitation]) -> np.ndarray:

        if len(coefficients) != len(excitations):
            raise ValueError(
                f"Number of coefficients ({len(coefficients)}) doesn't match "
                f"number of excitations ({len(excitations)})"
            )

        t1 = np.zeros((self.hamiltonian.nocc, self.hamiltonian.nvirt), dtype=complex)

        for coeff, exc in zip(coefficients, excitations):
            i = exc.occ
            a = exc.virt

            if self.hamiltonian.spin_type == "RHF": # we anticipate just alpha-alpha excitations here
                t1[i, a] += coeff
            else:
                raise NotImplementedError("UHF singles amplitudes not yet implemented")

        return t1
    
    def _construct_doubles_tensor(self, coefficients: np.ndarray, excitations: List[DoubleExcitation]):

        if len(coefficients) != len(excitations):
            raise ValueError(
                f"Number of coefficients ({len(coefficients)}) doesn't match "
                f"number of excitations ({len(excitations)})"
            )

        nocc, nvirt = self.hamiltonian.nocc, self.hamiltonian.nvirt
        t2 = np.zeros((nocc, nvirt, nocc, nvirt), dtype=complex)

        for coeff, exc in zip(coefficients, excitations):
            if self.hamiltonian.spin_type == "RHF":

                i = exc.occ1
                j = exc.occ2
                a = exc.virt1
                b = exc.virt2

                if exc.spin_case == 'alpha-beta':
                
                    if i == j and a == b:
                        t2[i,a,j,b] += coeff
                    else:
                        t2[i,a,j,b] += coeff
                        t2[j,b,i,a] += coeff

            else:
                raise NotImplementedError("UHF doubles amplitudes not yet implemented")

        _, t1sign = tn_addrs_signs(self.hamiltonian.norb, nocc, 1)
        antisymmetrised = np.einsum(
            "i,j,ij->ij", t1sign, t1sign, t2.reshape(nocc*nvirt, nocc*nvirt)
        ).reshape(nocc, nocc, nvirt, nvirt).transpose(0,2,1,3)

        return antisymmetrised 

    def _estimate_singles_doubles(self, protocol: ShadowProtocol) -> Tuple[np.ndarray, np.ndarray]:
        pass

if __name__ == "__main__":
    pass