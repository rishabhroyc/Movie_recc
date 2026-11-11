import numpy as np
from tests.conftest_helpers import generate_test_data
from src.data import clean_movies
from src.split import make_split


def test_no_future_leakage_and_all_rows_used():
    r, m = generate_test_data(300, 200, seed=1)
    D = make_split(r, clean_movies(m))
    assert len(D.train) + len(D.val) + len(D.test) == len(r)
    last_train = D.train.groupby("u")["timestamp"].max()
    first_val = D.val.groupby("u")["timestamp"].min()
    last_val = D.val.groupby("u")["timestamp"].max()
    first_test = D.test.groupby("u")["timestamp"].min()
    assert (last_train.reindex(first_val.index) <= first_val).all()
    assert (last_val.reindex(first_test.index).dropna() <= first_test.reindex(last_val.dropna().index)).all()


def test_every_user_has_test_items_and_split_ratio_is_close():
    r, m = generate_test_data(300, 200, seed=2)
    D = make_split(r, clean_movies(m))
    assert D.test["u"].nunique() == D.n_users
    assert 0.65 < len(D.train) / len(r) < 0.75
