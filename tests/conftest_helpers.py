"""Builds a small in-memory dataset and trained bundle for tests."""
import numpy as np
import pandas as pd
from types import SimpleNamespace
from src.config import GENRES
from src.data import clean_movies
from src.split import make_split
from src.models import ALS, build_item_vecs
from src.recommend import Recommender


def generate_test_data(n_users=600, n_items=400, seed=7):
    rng = np.random.default_rng(seed)
    n_clusters = 8
    item_cluster = rng.integers(0, n_clusters, n_items)
    cluster_genres = [rng.choice(len(GENRES), 2, replace=False) for _ in range(n_clusters)]
    pop = np.minimum(rng.zipf(1.5, n_items), 300).astype(float)
    pop /= pop.sum()
    years = rng.integers(1970, 2020, n_items)
    movie_ids = (np.arange(n_items) + 1) * 3
    genres = ["|".join(GENRES[g] for g in cluster_genres[c]) for c in item_cluster]
    movies = pd.DataFrame({"movieId": movie_ids,
                           "title": [f"Movie {i} ({y})" for i, y in zip(movie_ids, years)],
                           "genres": genres})
    rows = []
    for u in range(n_users):
        taste = rng.dirichlet(np.full(n_clusters, 0.3))
        n_u = int(min(n_items // 2, 20 + rng.lognormal(3.0, 0.8)))
        p = pop * (taste[item_cluster] + 0.02)
        p /= p.sum()
        items = rng.choice(n_items, n_u, replace=False, p=p)
        liking = taste[item_cluster[items]] / taste.max()
        rating = np.clip(np.round((2.5 + 2.5 * liking + rng.normal(0, 0.7, n_u)) * 2) / 2, 0.5, 5.0)
        ts = np.sort(rng.integers(1_000_000_000, 1_300_000_000, n_u))
        rows.append(pd.DataFrame({"userId": u + 1, "movieId": movie_ids[items], "rating": rating, "timestamp": ts}))
    ratings = pd.concat(rows, ignore_index=True)
    return ratings, movies


def small_recommender():
    r, m = generate_test_data(600, 400, seed=7)
    D = make_split(r, clean_movies(m))
    als = ALS(16, 0.1, 10).fit(D.M["rel_train"])
    bundle = dict(movies=D.movies, item_ids=D.item_ids, user_ids=D.user_ids,
                  item_vecs=build_item_vecs(D.movies), als_user=als.user_f, als_item=als.item_f,
                  alpha=0.5, seen_all=D.M["seen_all"], rel_all=D.M["rel_all"])
    return Recommender(bundle), D


def req(**kw):
    base = dict(include_genres=[], exclude_genres=[], year_min=None, year_max=None,
                similar_to_title=None, off_topic=False, parse_failed=False)
    base.update(kw)
    return SimpleNamespace(**base)
