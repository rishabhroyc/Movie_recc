import numpy as np
from scripts.ab_sim import two_prop_ztest, sample_size_per_arm, aa_false_positive_rate


def test_ztest_detects_a_big_difference_and_not_a_tiny_one():
    a = np.array([1] * 300 + [0] * 700, dtype=float)
    b = np.array([1] * 400 + [0] * 600, dtype=float)
    r = two_prop_ztest(a, b)
    assert r["p_value"] < 0.001 and r["ci_low"] > 0
    same = two_prop_ztest(a, a.copy())
    assert same["p_value"] > 0.99 and abs(same["lift"]) < 1e-12


def test_sample_size_matches_textbook_value():
    # baseline 10%, detect +2pp at 80% power, alpha 0.05 -> about 3,800 per arm
    assert 3700 < sample_size_per_arm(0.10, 0.02) < 3900


def test_aa_false_positive_rate_is_near_five_percent():
    x = (np.random.default_rng(0).random(4000) < 0.3).astype(float)
    assert 0.02 < aa_false_positive_rate(x, n_rep=400) < 0.09
