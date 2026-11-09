from baseline import etkdg_baseline, random_baseline
from molecules import random_branched_alkane
from torsion_env import TorsionEnv


def test_baselines():
    env = TorsionEnv(random_branched_alkane(4, seed=0), horizon=10)
    r = etkdg_baseline(env.mol, env.e0, env.z0, env.tau, n=30)
    assert r["score"] > 0 and 1 <= r["n_unique"] <= 30
    assert random_baseline(env) >= 0
