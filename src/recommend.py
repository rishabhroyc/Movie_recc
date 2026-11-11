"""Filter + rank. No LLM in this file: the ranker decides which movies come back."""
import difflib
import numpy as np
from src.config import GENRES, MIN_RATINGS_SERVE
from src.guardrails import norm_title
from src.models import ALS, Content, Hybrid, minmax

SIMILAR_BOOST = 0.5


class Recommender:
    def __init__(self, bundle):
        self.movies = bundle["movies"]
        self.item_ids, self.user_ids = bundle["item_ids"], bundle["user_ids"]
        self.user_index = {int(u): i for i, u in enumerate(self.user_ids)}
        self.item_vecs = bundle["item_vecs"]
        self.seen = bundle["seen_all"]            # every movie the user has rated, never recommended again
        self.rel = bundle["rel_all"]              # movies the user liked
        self.pop = np.asarray(self.rel.sum(0)).ravel().astype(np.float32)
        als = ALS.from_factors(bundle["als_user"], bundle["als_item"])
        self.model = Hybrid(als, Content(self.item_vecs, self.rel), bundle["alpha"])
        self.years = self.movies["year"].to_numpy(dtype=float)
        self.n_ratings = self.movies["n_ratings"].to_numpy()
        gi = {g: j for j, g in enumerate(GENRES)}
        self.G = np.zeros((len(self.movies), len(GENRES)), dtype=bool)
        for i, gs in enumerate(self.movies["genres"]):
            for g in gs:
                if g in gi:
                    self.G[i, gi[g]] = True
        self.title_index = {}
        for i in np.argsort(self.n_ratings):      # most-rated movie wins when titles collide
            self.title_index[norm_title(self.movies["title"].iat[i])] = int(i)

    def resolve_title(self, text):
        """Fuzzy match a typed title to a real catalogue movie index, or None."""
        key = norm_title(text)
        if key in self.title_index:
            return self.title_index[key]
        near = difflib.get_close_matches(key, self.title_index.keys(), n=1, cutoff=0.8)
        return self.title_index[near[0]] if near else None

    def _similar_liked(self, u, i):
        """The movie this user liked that is most like movie i (a real fact we can show in the reason)."""
        liked = self.rel.indices[self.rel.indptr[u]:self.rel.indptr[u + 1]]
        if len(liked) == 0:
            return None
        sims = self.item_vecs[liked] @ self.item_vecs[i]
        return self.movies["title"].iat[int(liked[int(np.argmax(sims))])] if sims.max() > 0.5 else None

    def _facts(self, u, i):
        row = self.movies.iloc[i]
        f = {"movie_id": int(self.item_ids[i]), "title": row["title"],
             "year": None if np.isnan(row["year"]) else int(row["year"]),
             "genres": list(row["genres"]), "avg_rating": round(float(row["avg_rating"]), 1),
             "n_ratings": int(row["n_ratings"])}
        sim = self._similar_liked(u, i) if u is not None else None
        if sim:
            f["similar_title"] = sim
        return f

    def recommend(self, user_id, request, k=5):
        notes = []
        if request.off_topic:
            return {"status": "refused", "message": "I can only help with picking movies.", "items": [], "notes": notes}
        if request.parse_failed:
            notes.append("I couldn't understand the request, so these are your default picks.")

        u = self.user_index.get(int(user_id)) if user_id is not None else None
        if u is None:
            scores = self.pop.copy()
            message = "New user - showing popular picks."
        else:
            scores = self.model.score_batch(np.array([u]))[0].astype(np.float32)
            scores[self.seen.indices[self.seen.indptr[u]:self.seen.indptr[u + 1]]] = -np.inf
            message = "Personalised picks."
        s = scores.copy()
        fin = np.isfinite(s)
        s[fin] = (s[fin] - s[fin].min()) / max(float(s[fin].max() - s[fin].min()), 1e-9)

        if request.similar_to_title:
            t = self.resolve_title(request.similar_to_title)
            if t is None:
                notes.append(f"Couldn't find '{request.similar_to_title}' in the catalogue, ignoring it.")
            else:
                s = s + SIMILAR_BOOST * (self.item_vecs @ self.item_vecs[t])
                s[t] = -np.inf

        mask = np.isfinite(s) & (self.n_ratings >= MIN_RATINGS_SERVE)
        with np.errstate(invalid="ignore"):
            if request.year_min is not None:
                mask &= self.years >= request.year_min
            if request.year_max is not None:
                mask &= self.years <= request.year_max
        gi = {g: j for j, g in enumerate(GENRES)}
        for g in request.exclude_genres:
            mask &= ~self.G[:, gi[g]]
        inc = [gi[g] for g in request.include_genres]
        if inc:
            strict = mask & self.G[:, inc].all(1)
            if strict.sum() >= k:
                mask = strict
            else:
                mask &= self.G[:, inc].any(1)
                notes.append("Not enough movies match every genre, so I relaxed it to any of them.")

        idx = np.flatnonzero(mask)
        if len(idx) == 0:
            return {"status": "no_results", "message": "No movies match those filters.", "items": [], "notes": notes}
        top = idx[np.argsort(-s[idx])[:k]]
        status = "cold_start" if u is None else "ok"
        return {"status": status, "message": message, "items": [self._facts(u, int(i)) for i in top], "notes": notes}
