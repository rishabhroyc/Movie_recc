"""Metrics + top-k helper. Binary relevance. Everything works in batches so 25M ratings fit in RAM."""
import numpy as np


def topk_from_model(model, users, history, k=10, batch=500):
    """Top-k item indices per user, never recommending anything in `history` (a user x item csr)."""
    out = np.zeros((len(users), k), dtype=np.int32)
    for s in range(0, len(users), batch):
        chunk = users[s:s + batch]
        scores = model.score_batch(chunk).astype(np.float32, copy=True)
        h = history[chunk].tocoo()
        scores[h.row, h.col] = -np.inf
        part = np.argpartition(-scores, k, axis=1)[:, :k]
        order = np.argsort(-np.take_along_axis(scores, part, 1), axis=1)
        out[s:s + batch] = np.take_along_axis(part, order, 1)
    return out


def per_user_metrics(topk, relevant, users, batch=500):
    """Recall@k, NDCG@k and hit@k for each user. `relevant` is a user x item csr of held-out liked items.
    Users must have at least one relevant item (skip the others before calling)."""
    n, k = topk.shape
    hits = np.zeros((n, k), dtype=bool)
    for s in range(0, n, batch):
        dense = relevant[users[s:s + batch]].toarray() > 0
        hits[s:s + batch] = np.take_along_axis(dense, topk[s:s + batch], 1)
    n_rel = np.asarray(relevant[users].getnnz(axis=1)).ravel()
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits * discounts).sum(1)
    ideal = np.cumsum(discounts)[np.minimum(n_rel, k) - 1]
    return {"recall": hits.sum(1) / n_rel, "ndcg": dcg / ideal, "hit": hits.any(1)}


def coverage(topk, n_items):
    """Share of the catalogue that shows up in at least one user's top-k."""
    return len(np.unique(topk)) / n_items


def tail_share(topk, tail_mask):
    """Share of recommendation slots filled by tail (non-popular) movies. Higher = less popularity bias."""
    return float(tail_mask[topk].mean())
