"""Step 3a: relax one conformer with MMFF94 and score it with the Gibbs reward."""
import math

from rdkit import Chem
from rdkit.Chem import AllChem

from molecules import set_torsions

K_BOLTZMANN = 0.0019872041  # kcal / (mol K), matches MMFF94 energy units


def mmff_energy(mol, conf_id=0):
    props = AllChem.MMFFGetMoleculeProperties(mol)
    return AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id).CalcEnergy()


def relax(mol, angles, max_iters=2000):
    """Set torsions (radians) on a copy of `mol`, minimize with MMFF94 (L-BFGS), return (mol, energy)."""
    m = Chem.Mol(mol)  # never mutate the caller's molecule
    set_torsions(m, angles)
    props = AllChem.MMFFGetMoleculeProperties(m)
    ff = AllChem.MMFFGetMoleculeForceField(m, props)
    ff.Minimize(maxIts=max_iters)
    return m, ff.CalcEnergy()


def gibbs(energy, e0, z0, tau):
    """Gibbs reward: exp(-(E - E0) / (k tau)) / Z0, energies in kcal/mol, tau in K."""
    return math.exp(-(energy - e0) / (K_BOLTZMANN * tau)) / z0


def etkdg_normalizers(mol, tau=500.0, n_confs=200, seed=0):
    """E0, Z0 from classical ETKDG conformers so that the baseline set scores ~1.

    shortcut: E0 = lowest energy found, Z0 = sum of Boltzmann weights over all
    ETKDG conformers without dedup; the paper does not spell out its recipe (Table 4).
    """
    m = Chem.Mol(mol)
    m.RemoveAllConformers()
    AllChem.EmbedMultipleConfs(m, n_confs, randomSeed=seed)
    res = AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=2000)
    energies = [e for _converged, e in res]
    e0 = min(energies)
    z0 = sum(math.exp(-(e - e0) / (K_BOLTZMANN * tau)) for e in energies)
    return e0, z0
