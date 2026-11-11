"""Simulated A/B test (offline replay, NOT real traffic): popularity (A) vs hybrid (B).
  python -m scripts.ab_sim      needs results/test_topk.npz and artifacts/bundle.pkl from scripts.train"""
import json
import numpy as np
from scipy.stats import norm
from src.config import RES_DIR, ART_DIR, GENRES, SEED
from src.models import load_bundle


def two_prop_ztest(x_a, x_b):
    """Compare two hit rates. Returns lift (B - A), 95% CI for the lift, two-sided p-value."""
    n_a, n_b = len(x_a), len(x_b)
    p_a, p_b = x_a.mean(), x_b.mean()
    pooled = (x_a.sum() + x_b.sum()) / (n_a + n_b)
    se0 = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se0 if se0 > 0 else 0.0
    se = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    lift = p_b - p_a
    return {"p_a": float(p_a), "p_b": float(p_b), "lift": float(lift),
            "ci_low": float(lift - 1.96 * se), "ci_high": float(lift + 1.96 * se),
            "p_value": float(2 * (1 - norm.cdf(abs(z))))}


def sample_size_per_arm(p_base, mde, alpha=0.05, power=0.8):
    """Users needed in EACH arm to detect an absolute lift of `mde` over baseline rate p_base."""
    p2 = p_base + mde
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    pbar = (p_base + p2) / 2
    num = (za * np.sqrt(2 * pbar * (1 - pbar)) + zb * np.sqrt(p_base * (1 - p_base) + p2 * (1 - p2))) ** 2
    return int(np.ceil(num / mde ** 2))


def aa_false_positive_rate(outcome, n_rep=1000, seed=SEED):
    """Split the SAME outcomes into two random arms many times. Should be significant about 5% of the time."""
    rng = np.random.default_rng(seed)
    n, sig = len(outcome), 0
    for _ in range(n_rep):
        perm = rng.permutation(n)
        a, b = outcome[perm[: n // 2]], outcome[perm[n // 2: 2 * (n // 2)]]
        sig += two_prop_ztest(a, b)["p_value"] < 0.05
    return sig / n_rep


def distinct_genres(topk, G):
    return G[topk].any(axis=1).sum(axis=1).mean()      # avg number of different genres in a top-10 list


def main():
    z = np.load(RES_DIR / "test_topk.npz")
    bundle = load_bundle(ART_DIR / "bundle.pkl")
    gi = {g: j for j, g in enumerate(GENRES)}
    G = np.zeros((len(bundle["movies"]), len(GENRES)), dtype=bool)
    for i, gs in enumerate(bundle["movies"]["genres"]):
        for g in gs:
            if g in gi:
                G[i, gi[g]] = True

    n = len(z["users"])
    rng = np.random.default_rng(SEED)
    arm = rng.permutation(n) % 2                      # random 50/50 assignment: 0 = A, 1 = B
    a_idx, b_idx = np.flatnonzero(arm == 0), np.flatnonzero(arm == 1)
    x_a, x_b = z["pop_hit"][a_idx].astype(float), z["hyb_hit"][b_idx].astype(float)
    result = two_prop_ztest(x_a, x_b)
    result.update({"n_a": len(a_idx), "n_b": len(b_idx),
                   "n_needed_per_arm_for_2pp_lift": sample_size_per_arm(result["p_a"], 0.02),
                   "diversity_A_distinct_genres": float(distinct_genres(z["pop_topk"][a_idx], G)),
                   "diversity_B_distinct_genres": float(distinct_genres(z["hyb_topk"][b_idx], G)),
                   "aa_false_positive_rate": aa_false_positive_rate(z["pop_hit"].astype(float)),
                   "label": "SIMULATED: offline replay of held-out ratings, not real traffic"})
    (RES_DIR / "ab_results.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
