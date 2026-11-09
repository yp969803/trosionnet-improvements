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


def etkdg_normalizers(mol, tau=500.0, n_confs=200, seeds=(0, 1, 2, 3)):
    """E0, Z0 from classical ETKDG conformers, pooled over several seeds.

    One ETKDG run stays in one region, so its lowest energy varies a lot between seeds
    (about 11 kcal/mol on an 11-torsion alkane), and exp() blows that up. E0 is the lowest
    energy over all seeds; Z0 is the Boltzmann sum divided by len(seeds), so one run of
    n_confs scores about 1.

    shortcut: no dedup, and E0 can still be beaten by a lucky run; the paper does not
    spell out its recipe (Table 4). Upgrade to more seeds if scores are still erratic.
    """
    energies = []
    for seed in seeds:
        m = Chem.Mol(mol)
        m.RemoveAllConformers()
        AllChem.EmbedMultipleConfs(m, n_confs, randomSeed=seed)
        energies += [e for _converged, e in AllChem.MMFFOptimizeMoleculeConfs(m, maxIters=2000)]
    e0 = min(energies)
    z0 = sum(math.exp(-(e - e0) / (K_BOLTZMANN * tau)) for e in energies) / len(seeds)
    return e0, z0
