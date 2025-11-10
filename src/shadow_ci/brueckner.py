from pyscf import scf, lib
from typing import Union, Optional
from shadow_ci.solvers import GroundStateSolver, FCISolver, VQESolver
from shadow_ci.estimator import GroundStateEstimator
import numpy as np
import scipy.linalg

from pyscf.ci.cisd import tn_addrs_signs

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

        diis = lib.diis.DIIS()

        for i in range(self.max_iter):

            if self.solver_type == 'fci':
                solver = FCISolver(self.mf) 
            else:
                solver = VQESolver(self.mf)

            estimator = GroundStateEstimator(self.mf, solver, verbose=4)

            E, c0, c1, _ = estimator.estimate_ground_state(
                n_samples=10000,
                n_k_estimators=20,
                n_jobs=4,
                calc_c1=True
            )

            nocc, _ = self.mf.mol.nelec
            norb = self.mf.mo_coeff.shape[0]
            _, t1sign = tn_addrs_signs(norb, nocc, 1)

            t1sign_2d = t1sign.reshape(nocc, norb - nocc)
            c1_raw = c1 / t1sign_2d

            t1 = c1_raw / c0

            if i > 1:
                if np.abs(E - E_prev) < energy_tol:
                    return E, t1
                
            self._update_mf(t1, diis=diis)

            E_prev = E


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

        self.mf.mo_coeff = bmo
        self.mf.e_tot = self.mf.energy_tot()
        return self.mf

    def _update_mo_coeff(self, mo_coeff: np.ndarray, t1: np.ndarray, ovlp: Optional[np.ndarray] = None, damping: np.float64 = 0.0, diis: Optional[lib.diis.DIIS] = None):

        nocc, nvir = t1.shape
        norb = mo_coeff.shape[-1]

        if not nocc + nvir == norb:
            raise ValueError("Incorrect shape for T1 amplitudes.")

        occ = np.s_[:nocc]
        vir = np.s_[nocc:]
        delta_occ = (1 - damping) * np.dot(mo_coeff[:, vir], t1.T)
        bmo_occ = mo_coeff[:, occ] + delta_occ

        # Orthogonalize occupied orbitals
        if ovlp is None:
            bmo_occ = np.linalg.qr(bmo_occ)[0]
        else:
            dm_occ = np.dot(bmo_occ, bmo_occ.T)
            if diis: dm_occ = diis.update(dm_occ)
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

    atom = make_hydrogen_chain(4)
    mol = gto.Mole()
    mol.build(atom=atom, basis="sto-3g")

    mf = scf.RHF(mol)
    mf.run()

    solver = BruecknerSolver(mf, solver_type='vqe')

    E = solver.solve()

    print(E)
        

