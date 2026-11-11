"""Per-user chronological split + the sparse matrices every other module uses."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from src.config import SPLIT, REL_THRESHOLD


@dataclass
class Data:
    user_ids: np.ndarray      # index -> original userId
    item_ids: np.ndarray      # index -> original movieId
    movies: pd.DataFrame      # row i describes item index i (title, year, genres, avg_rating, n_ratings)
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    M: dict                   # sparse user x item matrices, see make_split

    @property
    def n_users(self): return len(self.user_ids)

    @property
    def n_items(self): return len(self.item_ids)


def _mat(df, shape):
    return coo_matrix((np.ones(len(df), dtype=np.float32), (df["u"].to_numpy(), df["i"].to_numpy())),
                      shape=shape).tocsr()


def make_split(ratings, movies, rel_threshold=REL_THRESHOLD):
    r = ratings.copy()
    codes_u, user_ids = pd.factorize(r["userId"])
    codes_i, item_ids = pd.factorize(r["movieId"])
    r["u"], r["i"] = codes_u.astype(np.int32), codes_i.astype(np.int32)
    user_ids, item_ids = np.asarray(user_ids), np.asarray(item_ids)

    # sort each user's ratings by time, then cut by position within the user's own history
    r = r.sort_values(["u", "timestamp", "i"], kind="stable").reset_index(drop=True)
    n = r.groupby("u")["i"].transform("size").to_numpy()
    pos = r.groupby("u").cumcount().to_numpy() / n
    r["split"] = np.where(pos < SPLIT[0], "train", np.where(pos < SPLIT[1], "val", "test"))
    r["relevant"] = r["rating"] >= rel_threshold

    movies = movies.set_index("movieId").loc[item_ids].reset_index()
    stats = r.groupby("i")["rating"].agg(["mean", "size"]).reindex(range(len(item_ids)))
    movies["avg_rating"] = stats["mean"].to_numpy()
    movies["n_ratings"] = stats["size"].to_numpy()

    tr, va, te = (r[r["split"] == s] for s in ("train", "val", "test"))
    shape = (len(user_ids), len(item_ids))
    M = {"seen_train": _mat(tr, shape), "seen_val": _mat(va, shape), "seen_test": _mat(te, shape),
         "rel_train": _mat(tr[tr["relevant"]], shape), "rel_val": _mat(va[va["relevant"]], shape),
         "rel_test": _mat(te[te["relevant"]], shape)}
    M["seen_trainval"] = M["seen_train"] + M["seen_val"]
    M["seen_all"] = M["seen_trainval"] + M["seen_test"]
    M["rel_all"] = M["rel_train"] + M["rel_val"] + M["rel_test"]
    return Data(user_ids, item_ids, movies, tr, va, te, M)
