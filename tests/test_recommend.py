import numpy as np
from tests.conftest_helpers import small_recommender, req
from src.llm import explain

REC, D = small_recommender()
ID2IDX = {int(m): i for i, m in enumerate(D.item_ids)}


def _seen_ids(u):
    s = D.M["seen_all"]
    return set(s.indices[s.indptr[u]:s.indptr[u + 1]])


def test_seen_movies_never_recommended():
    for u in range(0, 600, 60):
        out = REC.recommend(int(D.user_ids[u]), req(), k=10)
        picked = {ID2IDX[it["movie_id"]] for it in out["items"]}
        assert picked and not (picked & _seen_ids(u))


def test_every_returned_movie_exists_in_catalogue():
    out = REC.recommend(int(D.user_ids[0]), req(), k=10)
    assert all(it["movie_id"] in ID2IDX for it in out["items"])


def test_hard_filters_are_respected():
    out = REC.recommend(int(D.user_ids[3]), req(year_min=1995, year_max=2005, exclude_genres=["Comedy"]), k=10)
    assert out["items"]
    for it in out["items"]:
        assert 1995 <= it["year"] <= 2005 and "Comedy" not in it["genres"]


def test_cold_start_user_gets_popular_picks_and_a_message():
    out = REC.recommend(99999999, req(), k=5)
    assert out["status"] == "cold_start" and "New user" in out["message"] and len(out["items"]) == 5


def test_off_topic_is_refused():
    out = REC.recommend(int(D.user_ids[0]), req(off_topic=True))
    assert out["status"] == "refused" and out["items"] == []


def test_impossible_filter_returns_no_results():
    out = REC.recommend(int(D.user_ids[0]), req(year_min=2100), k=5)
    assert out["status"] == "no_results"


def test_similar_title_is_resolved_and_excluded():
    title = REC.movies["title"].iat[5]
    assert REC.resolve_title(title.lower()) == 5
    out = REC.recommend(None, req(similar_to_title=title), k=5)
    assert all(it["title"] != title for it in out["items"])


def test_explainer_replaces_bad_llm_text_with_template():
    items = REC.recommend(int(D.user_ids[1]), req(), k=3)["items"]
    bad = '[{"title": "%s", "reason": "A 1850 Sci-Fi masterpiece."}, {"title": "Invented Movie", "reason": "x"}]' % items[0]["title"]
    out = explain(items, call=lambda s, u: bad)
    assert out[0]["fallback"] and "1850" not in out[0]["reason"]
    assert all(o["fallback"] for o in out)          # no LLM output usable -> all templates, nothing invented
    assert [o["title"] for o in out] == [it["title"] for it in items]


def test_explainer_keeps_a_clean_reason():
    it = REC.recommend(int(D.user_ids[1]), req(), k=1)["items"][0]
    good = '[{"title": "%s", "reason": "Rated %.1f on average."}]' % (it["title"], it["avg_rating"])
    assert not explain([it], call=lambda s, u: good)[0]["fallback"]


def test_recommendations_include_valid_imdb_url():
    out = REC.recommend(int(D.user_ids[0]), req(), k=3)
    assert out["items"]
    for it in out["items"]:
        assert it["imdb_url"].startswith("https://www.imdb.com/")
