import numpy as np
from rdkit.Chem import rdMolTransforms

from graph import EDGE_DIM, NODE_DIM, mol_to_graph
from molecules import random_branched_alkane


def test_shapes_and_centering():
    mol = random_branched_alkane(5, seed=0)
    g = mol_to_graph(mol)
    n, bonds = mol.GetNumAtoms(), mol.GetNumBonds()
    assert g.x.shape == (n, NODE_DIM)
    assert g.edge_index.shape == (2, 2 * bonds) and g.edge_attr.shape == (2 * bonds, EDGE_DIM)
    assert g.torsions.shape == (5, 4)
    assert g.x[:, :3].sum(1).eq(1).all()  # one-hot atom type
    assert np.allclose(g.x[:, 3:].mean(0).numpy(), 0, atol=1e-5)


def test_rotation_invariance():
    mol = random_branched_alkane(5, seed=0)
    before = mol_to_graph(mol).x
    q, _ = np.linalg.qr(np.random.default_rng(0).normal(size=(3, 3)))
    q *= np.sign(np.linalg.det(q))  # proper rotation
    m = np.eye(4)
    m[:3, :3] = q
    m[:3, 3] = [5.0, -2.0, 1.0]  # plus a translation
    rdMolTransforms.TransformConformer(mol.GetConformer(), m)
    after = mol_to_graph(mol).x
    assert np.allclose(before.numpy(), after.numpy(), atol=1e-4)
