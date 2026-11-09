import math

import pytest

from molecules import (bucket_to_angle, get_torsion_angles, get_torsions,
                       random_branched_alkane, set_torsions)


@pytest.mark.parametrize("n", [1, 2, 5, 11])
def test_torsion_count(n):
    mol = random_branched_alkane(n, seed=0)
    assert len(get_torsions(mol)) == n
    assert mol.GetNumConformers() == 1


def test_set_torsions_roundtrip():
    mol = random_branched_alkane(6, seed=1)
    target = [bucket_to_angle(k) for k in (0, 1, 2, 3, 4, 5)]
    set_torsions(mol, target)
    got = get_torsion_angles(mol)
    for t, g in zip(target, got):
        d = (t - g + math.pi) % (2 * math.pi) - math.pi  # wrap-around difference
        assert abs(d) < 1e-3
