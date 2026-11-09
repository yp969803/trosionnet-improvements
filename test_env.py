import random

from rdkit import Chem

from env import etkdg_normalizers, gibbs, mmff_energy, relax
from molecules import (bucket_to_angle, get_torsion_angles,
                       random_branched_alkane, set_torsions)


def test_relax_lowers_energy_and_keeps_input():
    mol = random_branched_alkane(5, seed=0)
    before = get_torsion_angles(mol)
    rng = random.Random(0)
    angles = [bucket_to_angle(rng.randrange(6)) for _ in range(5)]
    raw = Chem.Mol(mol)
    set_torsions(raw, angles)
    relaxed, e = relax(mol, angles)
    assert e <= mmff_energy(raw)  # minimization never raises energy
    assert get_torsion_angles(mol) == before  # input untouched
    # relaxed energy matches a fresh evaluation
    assert abs(mmff_energy(relaxed) - e) < 1e-6


def test_gibbs_monotone_in_energy():
    assert gibbs(10.0, 10.0, 1.0, 500) == 1.0
    assert gibbs(12.0, 10.0, 1.0, 500) < gibbs(11.0, 10.0, 1.0, 500)


def test_normalizers():
    mol = random_branched_alkane(4, seed=2)
    e0, z0 = etkdg_normalizers(mol, n_confs=30, seeds=(0, 1))
    assert z0 >= 0.5  # the lowest conformer contributes exp(0) = 1, divided by 2 seeds
