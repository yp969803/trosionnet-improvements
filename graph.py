"""Step 5a: RDKit molecule -> PyTorch Geometric graph."""
import numpy as np
import torch
from rdkit import Chem
from torch_geometric.data import Data

from molecules import get_torsions

ATOMS = {6: 0, 1: 1, 8: 2}  # C, H, O (the paper lists C and O; alkanes also need H)
BONDS = {Chem.BondType.SINGLE: 0, Chem.BondType.DOUBLE: 1,
         Chem.BondType.TRIPLE: 2, Chem.BondType.AROMATIC: 3}
NODE_DIM, EDGE_DIM = len(ATOMS) + 3, len(BONDS) + 2


def canonical_positions(coords):
    """Center on the origin, rotate onto principal axes, fix each axis sign by skewness.

    shortcut: the paper does not say how it fixes axis signs; this rule is my choice and
    can fail for symmetric molecules whose skewness is ~0 along an axis.
    """
    c = coords - coords.mean(axis=0)
    _vals, vecs = np.linalg.eigh(c.T @ c)
    c = c @ vecs[:, ::-1]  # largest-variance axis first
    sign = np.sign((c ** 3).sum(axis=0))
    sign[sign == 0] = 1.0
    return c * sign


def mol_to_graph(mol, conf_id=0):
    pos = canonical_positions(mol.GetConformer(conf_id).GetPositions())

    x = torch.zeros(mol.GetNumAtoms(), NODE_DIM)
    for a in mol.GetAtoms():
        x[a.GetIdx(), ATOMS[a.GetAtomicNum()]] = 1.0  # KeyError for unsupported elements
    x[:, len(ATOMS):] = torch.tensor(pos, dtype=torch.float32)

    src, dst, attr = [], [], []
    for b in mol.GetBonds():
        f = torch.zeros(EDGE_DIM)
        f[BONDS[b.GetBondType()]] = 1.0
        f[len(BONDS)] = float(b.GetIsConjugated())
        f[len(BONDS) + 1] = float(b.IsInRing())
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        src += [i, j]  # both directions: messages flow each way
        dst += [j, i]
        attr += [f, f]

    return Data(
        x=x,
        edge_index=torch.tensor([src, dst], dtype=torch.long),
        edge_attr=torch.stack(attr),
        torsions=torch.tensor(get_torsions(mol), dtype=torch.long),  # (n_torsions, 4)
    )
