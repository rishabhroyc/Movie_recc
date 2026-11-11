"""Popularity, ALS, content and hybrid models. Each has score_batch(user_indices) -> (n_users x n_items)."""
import pickle
import warnings
import numpy as np
from sklearn.decomposition import TruncatedSVD
from src.config import GENRES, ALS_CONFIDENCE, SEED


def minmax(x):
    """Per-user 0..1 scaling so ALS and content scores can be mixed."""
    lo, hi = x.min(1, keepdims=True), x.max(1, keepdims=True)
    return (x - lo) / np.maximum(hi - lo, 1e-9)


class Popularity:
    """The number to beat: how many people liked each movie in train."""
    def __init__(self, rel_train):
        self.pop = np.asarray(rel_train.sum(0)).ravel().astype(np.float32)

    def score_batch(self, users):
        return np.tile(self.pop[None, :], (len(users), 1))


class ALS:
    """Collaborative filtering. Uses the `implicit` library; if it isn't installed it falls back to
    TruncatedSVD (only so the code still runs for smoke tests, install implicit for real results)."""
    def __init__(self, factors=64, reg=0.1, iters=15):
        self.factors, self.reg, self.iters = factors, reg, iters
        self.backend = None

    def fit(self, rel_train):
        try:
            from implicit.cpu.als import AlternatingLeastSquares
            model = AlternatingLeastSquares(factors=self.factors, regularization=self.reg,
                                            iterations=self.iters, random_state=SEED)
            model.fit(rel_train * ALS_CONFIDENCE, show_progress=False)
            self.user_f, self.item_f = np.asarray(model.user_factors), np.asarray(model.item_factors)
            self.backend = "implicit-als"
        except ImportError:
            warnings.warn("implicit not installed: using TruncatedSVD fallback (smoke tests only)")
            svd = TruncatedSVD(n_components=self.factors, random_state=SEED)
            self.user_f = svd.fit_transform(rel_train)
            self.item_f = svd.components_.T
            self.backend = "svd-fallback"
        return self

    @classmethod
    def from_factors(cls, user_f, item_f):
        m = cls()
        m.user_f, m.item_f, m.backend = user_f, item_f, "loaded"
        return m

    def score_batch(self, users):
        return self.user_f[users] @ self.item_f.T


def build_item_vecs(movies, genome_dir=None):
    """Item features: genre multi-hot + decade one-hot (+ tag-genome SVD if the ml-25m genome files exist).
    Rows are L2 normalised so a dot product is a cosine similarity."""
    n = len(movies)
    g_index = {g: j for j, g in enumerate(GENRES)}
    genre = np.zeros((n, len(GENRES)), dtype=np.float32)
    for i, gs in enumerate(movies["genres"]):
        for g in gs:
            if g in g_index:
                genre[i, g_index[g]] = 1.0
    decades = np.arange(1910, 2030, 10)
    year = movies["year"].to_numpy(dtype=float)
    dec = np.zeros((n, len(decades)), dtype=np.float32)
    ok = ~np.isnan(year)
    idx = np.clip(((year[ok] // 10) * 10 - 1910) // 10, 0, len(decades) - 1).astype(int)
    dec[np.flatnonzero(ok), idx] = 1.0
    blocks = [genre, 0.5 * dec]
    if genome_dir is not None:
        g = _genome_block(movies, genome_dir)
        if g is not None:
            blocks.append(g)
    v = np.hstack(blocks)
    norms = np.linalg.norm(v, axis=1, keepdims=True)
    return v / np.maximum(norms, 1e-9)


def _genome_block(movies, genome_dir, dims=32):
    """Tag genome (ml-25m only): 1128 relevance scores per movie, squeezed to `dims` with SVD."""
    import pandas as pd
    path = genome_dir / "genome-scores.csv"
    if not path.exists():
        return None
    gs = pd.read_csv(path, dtype={"movieId": "int32", "tagId": "int16", "relevance": "float32"})
    pos = {mid: i for i, mid in enumerate(movies["movieId"].to_numpy())}
    gs = gs[gs["movieId"].isin(pos)]
    rows = gs["movieId"].map(pos).to_numpy()
    tag_idx = gs["tagId"].astype("category").cat.codes.to_numpy()
    full = np.zeros((len(movies), tag_idx.max() + 1), dtype=np.float32)
    full[rows, tag_idx] = gs["relevance"].to_numpy()
    has = full.any(1)
    out = np.zeros((len(movies), dims), dtype=np.float32)
    svd = TruncatedSVD(n_components=dims, random_state=SEED)
    out[has] = svd.fit_transform(full[has])
    out[has] /= np.maximum(np.linalg.norm(out[has], axis=1, keepdims=True), 1e-9)
    return out


class Content:
    """User profile = average vector of the movies the user liked. Score = cosine with every movie."""
    def __init__(self, item_vecs, rel):
        self.item_vecs, self.rel = item_vecs, rel

    def score_batch(self, users):
        prof = self.rel[users] @ self.item_vecs
        norm = np.linalg.norm(prof, axis=1, keepdims=True)
        prof = prof / np.maximum(norm, 1e-9)
        return (prof @ self.item_vecs.T).astype(np.float32)


class Hybrid:
    """alpha * ALS + (1 - alpha) * content, each min-max scaled per user."""
    def __init__(self, als, content, alpha=0.5):
        self.als, self.content, self.alpha = als, content, alpha

    def score_batch(self, users):
        a = self.alpha
        if a >= 1.0:
            return self.als.score_batch(users)
        if a <= 0.0:
            return self.content.score_batch(users)
        return a * minmax(self.als.score_batch(users)) + (1 - a) * minmax(self.content.score_batch(users))


def save_bundle(path, **kw):
    path.parent.mkdir(exist_ok=True, parents=True)
    with open(path, "wb") as f:
        pickle.dump(kw, f, protocol=4)


def load_bundle(path):
    with open(path, "rb") as f:
        return pickle.load(f)
