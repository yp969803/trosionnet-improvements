"""Step 2: random branched alkanes, torsion detection, torsion setting."""
import math
import random

from rdkit import Chem
from rdkit.Chem import AllChem, TorsionFingerprints, rdMolTransforms


def get_torsions(mol):
    """One 4-atom index tuple per rotatable torsion.

    Uses RDKit's TFD torsion list so the torsions match what TFD compares in step 3
    (terminal methyls and symmetry-equivalent duplicates are excluded).
    """
    tors, _rings = TorsionFingerprints.CalculateTorsionLists(mol)
    return [tuple(group[0][0]) for group in tors]


def set_torsions(mol, angles, conf_id=0):
    """Set each torsion (radians) in place. Alkanes are acyclic, so each is independent."""
    conf = mol.GetConformer(conf_id)
    for quad, angle in zip(get_torsions(mol), angles):
        rdMolTransforms.SetDihedralRad(conf, *quad, angle)


def get_torsion_angles(mol, conf_id=0):
    conf = mol.GetConformer(conf_id)
    return [rdMolTransforms.GetDihedralRad(conf, *q) for q in get_torsions(mol)]


def _random_tree(n_carbons, rng):
    rw = Chem.RWMol()
    rw.AddAtom(Chem.Atom(6))
    deg = [0]
    for i in range(1, n_carbons):
        parent = rng.choice([j for j in range(i) if deg[j] < 4])
        rw.AddAtom(Chem.Atom(6))
        rw.AddBond(parent, i, Chem.BondType.SINGLE)
        deg[parent] += 1
        deg.append(1)
    return rw.GetMol()


def random_branched_alkane(n_torsions, seed=None, max_tries=10000):
    """Random branched alkane with exactly `n_torsions` torsions, with H and a 3D conformer."""
    rng = random.Random(seed)
    for _ in range(max_tries):
        n_c = rng.randint(n_torsions + 3, 2 * n_torsions + 4)
        mol = _random_tree(n_c, rng)
        Chem.SanitizeMol(mol)
        mol = Chem.AddHs(mol)
        if len(get_torsions(mol)) == n_torsions:
            if AllChem.EmbedMolecule(mol, randomSeed=rng.randint(0, 2**31 - 1)) != 0:
                continue  # embedding failed, resample
            return mol
    raise RuntimeError(f"no alkane with {n_torsions} torsions in {max_tries} tries")


# Action space from the paper: 6 buckets per torsion, angle = k*pi/3, k=1..6.
N_BUCKETS = 6


def bucket_to_angle(k):
    return (k + 1) * math.pi / 3
