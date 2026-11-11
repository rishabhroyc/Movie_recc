"""Train + evaluate everything.
  python -m scripts.train --dataset ml-25m --eval-users 10000
Writes results/results.csv, results/slices.csv, results/test_topk.npz and artifacts/bundle.pkl."""
import argparse
import time
import numpy as np
import pandas as pd
from src.config import K, SEED, RES_DIR, ART_DIR, DATA_DIR, HEAD_FRACTION
from src.data import load_dataset
from src.split import make_split
from src.models import Popularity, ALS, Content, Hybrid, build_item_vecs, save_bundle
from src.metrics import topk_from_model, per_user_metrics, coverage, tail_share


def evaluate(model, users, history, target, tail_mask):
    topk = topk_from_model(model, users, history, K)
    pm = per_user_metrics(topk, target, users)
    row = {"n_users": len(users), "recall": pm["recall"].mean(), "ndcg": pm["ndcg"].mean(),
           "hit_rate": pm["hit"].mean(), "coverage": coverage(topk, len(tail_mask)),
           "tail_share": tail_share(topk, tail_mask)}
    return row, topk, pm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ml-25m", choices=["ml-25m", "ml-1m"])
    ap.add_argument("--max-users", type=int, default=None, help="sample users to save RAM, e.g. 50000")
    ap.add_argument("--eval-users", type=int, default=10000)
    ap.add_argument("--no-grid", action="store_true", help="skip the ALS grid, use factors=64 reg=0.1")
    ap.add_argument("--no-genome", action="store_true", help="ignore the ml-25m tag genome files")
    a = ap.parse_args()
    RES_DIR.mkdir(exist_ok=True); ART_DIR.mkdir(exist_ok=True)

    t0 = time.time()
    ratings, movies = load_dataset(a.dataset, a.max_users)
    D = make_split(ratings, movies)
    M = D.M
    print(f"{a.dataset}: {len(ratings):,} ratings, {D.n_users:,} users, {D.n_items:,} movies "
          f"| train {len(D.train):,} val {len(D.val):,} test {len(D.test):,}  ({time.time()-t0:.0f}s)")

    # users with something to predict in both val and test; sample a fixed set so runs are repeatable
    ok = np.flatnonzero((M["rel_val"].getnnz(1) > 0) & (M["rel_test"].getnnz(1) > 0))
    users = np.sort(np.random.default_rng(SEED).choice(ok, min(a.eval_users, len(ok)), replace=False))
    print(f"evaluating on {len(users):,} users")

    pop_counts = np.asarray(M["rel_train"].sum(0)).ravel()
    head = np.argsort(-pop_counts)[: int(HEAD_FRACTION * D.n_items)]
    tail_mask = np.ones(D.n_items, dtype=bool); tail_mask[head] = False

    rows = []
    def log(model_name, split, row, params=""):
        rows.append({"dataset": a.dataset, "model": model_name, "split": split, "k": K, "params": params, **row})
        print(f"  {model_name:<10} {split:<4} recall {row['recall']:.4f}  ndcg {row['ndcg']:.4f}  "
              f"coverage {row['coverage']:.3f}  tail {row['tail_share']:.3f}")

    # 1. popularity
    pop = Popularity(M["rel_train"])
    log("popularity", "val", evaluate(pop, users, M["seen_train"], M["rel_val"], tail_mask)[0])

    # 2. ALS: small grid on VAL only, test is touched once at the end
    grid = [(64, 0.1)] if a.no_grid else [(f, r) for f in (32, 64) for r in (0.01, 0.1)]
    best = None
    for f, reg in grid:
        t = time.time()
        als = ALS(f, reg, 15).fit(M["rel_train"])
        row = evaluate(als, users, M["seen_train"], M["rel_val"], tail_mask)[0]
        log("als", "val", row, f"factors={f} reg={reg} [{als.backend}] {time.time()-t:.0f}s")
        if best is None or row["ndcg"] > best[0]:
            best = (row["ndcg"], als, f, reg)
    _, als, f, reg = best

    # 3. content + hybrid (alpha tuned on val)
    genome_dir = None if (a.no_genome or a.dataset != "ml-25m") else DATA_DIR / "ml-25m"
    item_vecs = build_item_vecs(D.movies, genome_dir)
    content = Content(item_vecs, M["rel_train"])
    log("content", "val", evaluate(content, users, M["seen_train"], M["rel_val"], tail_mask)[0])
    best_a = (-1, 0.5)
    for alpha in (0.25, 0.5, 0.75):
        row = evaluate(Hybrid(als, content, alpha), users, M["seen_train"], M["rel_val"], tail_mask)[0]
        log("hybrid", "val", row, f"alpha={alpha}")
        if row["ndcg"] > best_a[0]:
            best_a = (row["ndcg"], alpha)
    alpha = best_a[1]
    hybrid = Hybrid(als, content, alpha)

    # 4. test, once. History = train + val so we never "recommend" something already seen.
    print("TEST")
    hist = M["seen_trainval"]
    res = {}
    for name, mod, params in [("popularity", pop, ""), ("als", als, f"factors={f} reg={reg}"),
                              ("content", content, ""), ("hybrid", hybrid, f"alpha={alpha}")]:
        row, topk, pm = evaluate(mod, users, hist, M["rel_test"], tail_mask)
        log(name, "test", row, params)
        res[name] = (topk, pm)
    pd.DataFrame(rows).to_csv(RES_DIR / "results.csv", index=False)

    # 5. slices: light vs heavy users (by number of train ratings)
    counts = np.asarray(M["seen_train"].sum(1)).ravel()[users]
    q25, q75 = np.percentile(counts, [25, 75])
    sl = []
    for name, (_, pm) in res.items():
        for label, sel in [("light (bottom 25%)", counts <= q25), ("heavy (top 25%)", counts >= q75), ("all", counts > -1)]:
            sl.append({"model": name, "slice": label, "n_users": int(sel.sum()),
                       "recall": pm["recall"][sel].mean(), "ndcg": pm["ndcg"][sel].mean()})
    sl = pd.DataFrame(sl)
    sl.to_csv(RES_DIR / "slices.csv", index=False)
    print(sl.round(4).to_string(index=False))

    # 6. save what the A/B simulation and the demo app need
    np.savez(RES_DIR / "test_topk.npz", users=users,
             pop_topk=res["popularity"][0], hyb_topk=res["hybrid"][0],
             pop_hit=res["popularity"][1]["hit"], hyb_hit=res["hybrid"][1]["hit"])
    # for serving, the content profile and "seen" list use ALL of a user's history
    save_bundle(ART_DIR / "bundle.pkl", movies=D.movies, item_ids=D.item_ids, user_ids=D.user_ids,
                item_vecs=item_vecs, als_user=als.user_f, als_item=als.item_f, alpha=alpha,
                seen_all=M["seen_all"], rel_all=M["rel_all"])
    print(f"done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
