from pyscf import scf, lib
from typing import Union, Optional
from shadow_ci.solvers import FCISolver, VQESolver
from shadow_ci.estimator import GroundStateEstimator
from shadow_ci.utils import get_hf_reference
from shadow_ci.utils import get_single_excitations
import numpy as np
import scipy.linalg

class BruecknerSolver:

    def __init__(
            self, 
            mf: Union[scf.hf.RHF, scf.uhf.UHF],
            solver_type: str = 'vqe', *,
            max_iter: int = 10,
            verbose: int = 1
        ):

        if isinstance(mf, scf.hf.RHF):
            self.spin_type = 'RHF'
        else:
            raise NotImplementedError()
        
        if solver_type in ['vqe', 'fci']:
            self.solver_type = solver_type
        else:
            raise NotImplementedError()

        self.mf = mf
        self.max_iter = max_iter
        self.verbose = verbose

    def solve(self, energy_tol: float = 1e-6):

        diis = None

        energies = []
        t1_norms = []
        energy_changes = []
        t1_norm_changes = []
        reference_overlaps = []
        exact_reference_overlaps = []

        for i in range(self.max_iter):

            self.mf.run()

            if self.solver_type == 'fci':
                solver = FCISolver(self.mf) 
            else:
                solver = VQESolver(self.mf)

            estimator = GroundStateEstimator(self.mf, solver, verbose=4)

            E, c0, c1, _ = estimator.estimate_ground_state(
                n_samples=1000,
                n_k_estimators=40,
                n_jobs=1,
                calc_c1=True
            )

            singles = get_single_excitations(self.mf)

            c1_exact = np.array([estimator.trial.data[s.bitstring.to_int()] for s in singles])
            nocc, _ = mf.mol.nelec
            norb = mf.mo_coeff.shape[0]
            nvirt = norb - nocc
            t1_exact = np.empty((nocc, nvirt), dtype=np.float64)
            for c, e in zip(c1_exact, singles):
                i = e.occ
                a = e.virt
                t1_exact[i,a] = c.real

            ref_idx = get_hf_reference(self.mf).to_int()
            exact_c0 = estimator.trial.data[ref_idx]

            t1 = t1_exact.real 

            norm = np.linalg.norm(t1)

            energies.append(E)
            t1_norms.append(norm)
            reference_overlaps.append(c0)


            exact_reference_overlaps.append(exact_c0)

            if i > 0:
                energy_changes.append(E - energies[i-1])
                t1_norm_changes.append(norm - t1_norms[i-1])

            # if i > 0:
            #     if np.abs(E - energies[i-1]) < energy_tol:
            #         history = {
            #             'energies': energies,
            #             't1_norms': t1_norms,
            #             'reference_overlaps': c0,
            #             'energy_changes': energy_changes,
            #             't1_norm_changes': t1_norm_changes,
            #             'converged': True,
            #             'n_iter': i + 1
            #         }
            #         return E, t1, history

            self._update_mf(t1, diis=diis, damping=0.0)

        history = {
            'energies': energies,
            't1_norms': t1_norms,
            'reference_overlaps': c0,
            'energy_changes': energy_changes,
            't1_norm_changes': t1_norm_changes,
            'converged': False,
            'n_iter': self.max_iter
        }

        print(exact_reference_overlaps)
        print(t1_norms)

        return E, t1, history


    def _update_mf(self, t1: np.ndarray, canonicalize=True, damping=0.0, diis: Optional[lib.diis.DIIS] = None):

        mo_coeff = self.mf.mo_coeff
        norb = mo_coeff.shape[-1]
        nocc, _ = self.mf.mol.nelec
        nvirt = norb - nocc

        if not t1.shape == (nocc, nvirt):
            raise ValueError("Incorrect shape for T1 amplitudes.")
        
        ovlp = self.mf.get_ovlp()
        if np.allclose(ovlp, np.eye(ovlp.shape[-1])):
            ovlp = None

        bmo_occ, bmo_vir = self._update_mo_coeff(mo_coeff, t1, ovlp, damping=damping, diis=diis)

        if canonicalize:
            if canonicalize == "hcore":
                h1e = self.mf.get_hcore()
            else:
                h1e = self.mf.get_fock()
            _, r = np.linalg.eigh(np.linalg.multi_dot((bmo_occ.T, h1e, bmo_occ)))
            bmo_occ = np.dot(bmo_occ, r)
            _, r = np.linalg.eigh(np.linalg.multi_dot((bmo_vir.T, h1e, bmo_vir)))
            bmo_vir = np.dot(bmo_vir, r)

        bmo = np.hstack((bmo_occ, bmo_vir))
        if ovlp is None:
            assert np.allclose(np.dot(bmo.T, bmo), np.eye(norb))
        else:
            assert np.allclose(np.linalg.multi_dot((bmo.T, ovlp, bmo)), np.eye(norb))

        if ovlp is None:
            assert np.allclose(np.dot(bmo.T, bmo), np.eye(norb))
        else:
            assert np.allclose(np.linalg.multi_dot((bmo.T, ovlp, bmo)), np.eye(norb))

        self.mf.mo_coeff = bmo
        self.mf.e_tot = self.mf.energy_tot()
        return self.mf

    def _update_mo_coeff(self, mo_coeff: np.ndarray, t1: np.ndarray, ovlp: Optional[np.ndarray] = None, damping: np.float64 = 0.0, diis: Optional[lib.diis.DIIS] = None):

        nocc, nvir = t1.shape
        norb = mo_coeff.shape[-1]

        if not nocc + nvir == norb:
            raise ValueError("Incorrect shape for T1 amplitudes.")

        delta_occ = (1 - damping) * np.dot(mo_coeff[:, nocc:], t1.T) # multiply virtuals by t1.T
        bmo_occ = mo_coeff[:, :nocc] + delta_occ

        if ovlp is None:
            bmo_occ = np.linalg.qr(bmo_occ)[0]
        else:
            dm_occ = np.dot(bmo_occ, bmo_occ.T)
            e, v = scipy.linalg.eigh(dm_occ, b=ovlp, type=2)
            bmo_occ = v[:, -nocc:]

        if diis: 
            dm_occ = np.dot(bmo_occ, bmo_occ.T)
            dm_occ = diis.update(dm_occ)
            e, v = scipy.linalg.eigh(dm_occ, b=ovlp, type=2)
            bmo_occ = v[:, -nocc:]
        
        # Virtual space
        if ovlp is None:
            dm_vir = np.eye(norb) - np.dot(bmo_occ, bmo_occ.T)
        else:
            dm_vir = np.linalg.inv(ovlp) - np.dot(bmo_occ, bmo_occ.T)

        _, v = scipy.linalg.eigh(dm_vir, b=ovlp, type=2)
        bmo_vir = v[:, -nvir:]

        assert bmo_occ.shape[-1] == nocc
        assert bmo_vir.shape[-1] == nvir
        if ovlp is None:
            ovlp = np.eye(norb)
        bmo = np.hstack((bmo_occ, bmo_vir))
        assert np.allclose(np.linalg.multi_dot((bmo.T, ovlp, bmo)) - np.eye(norb), 0)
        return bmo_occ, bmo_vir



if __name__ == "__main__":


    from shadow_ci.utils import make_hydrogen_chain
    from pyscf import gto

    atom = make_hydrogen_chain(4, bond_length=0.5)
    mol = gto.Mole()
    mol.build(atom=atom, basis="sto-3g")

    mf = scf.RHF(mol)
    mf.run()

    solver = BruecknerSolver(mf, solver_type='fci', max_iter=10)

    E, t1, history = solver.solve()

    # print(f"Final energy: {E}")
    # print(f"Converged: {history['converged']}")
    # print(f"Iterations: {history['n_iter']}")
    # print(f"Energy history: {history['energies']}")
    # print(f"T1 norm history: {history['t1_norms']}")
        

