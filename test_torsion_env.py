from molecules import random_branched_alkane
from torsion_env import TorsionEnv


def test_duplicate_gets_zero_reward_and_episode_ends():
    env = TorsionEnv(random_branched_alkane(4, seed=0), horizon=6)
    env.reset(seed=0)
    a = env.action_space.sample()
    _, r1, _, _, _ = env.step(a)
    _, r2, _, _, _ = env.step(a)  # identical action -> identical conformer
    assert r1 > 0 and r2 == 0.0
    assert len(env.found.GetConformers()) == 1  # duplicate not stored
    done = False
    for _ in range(4):
        _, r, _, done, _ = env.step(env.action_space.sample())
        assert r >= 0
    assert done  # truncated at the horizon
