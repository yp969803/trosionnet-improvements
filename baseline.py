"""Step 4: baselines to beat: classical ETKDG and random torsion actions."""
import time

from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints

from env import gibbs


def etkdg_baseline(mol, e0, z0, tau, n=200, tfd_threshold=0.10, seeds=(100, 101, 102)):
    """Mean over `seeds` of the Gibbs score of n ETKDG+MMFF conformers after TFD dedup.

    Pass the env's e0/z0 so the score is on the env's reward scale. The default seeds
    differ from etkdg_normalizers' (0-3) so these are independent samples; averaging
    softens the effect of one lucky low-energy run.
    """
    runs = [_etkdg_run(mol, e0, z0, tau, n, tfd_threshold, s) for s in seeds]
    return {k: sum(r[k] for r in runs) / len(runs) for k in runs[0]}


def _etkdg_run(mol, e0, z0, tau, n, tfd_threshold, seed):
    t0 = time.time()
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    AllChem.EmbedMultipleConfs(m, n, randomSeed=seed)
    energies = [e for _ok, e in AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=2000)]
    tfd = TorsionFingerprints.GetTFDMatrix(m)  # flat lower triangle: (i, j<i) -> i*(i-1)/2 + j
    kept = []
    for i in range(len(energies)):  # arrival order, like the env
        if all(tfd[i * (i - 1) // 2 + j] > tfd_threshold for j in kept):
            kept.append(i)
    return {
        "score": sum(gibbs(energies[i], e0, z0, tau) for i in kept),
        "n_unique": len(kept),
        "seconds": time.time() - t0,
    }


def random_baseline(env, steps=None):
    """Total reward of uniformly random torsion actions over one episode."""
    env.reset()
    for _ in range(steps or env.horizon):
        env.step(env.action_space.sample())
    return env.score
