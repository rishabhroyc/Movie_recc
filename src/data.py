"""Download, read and clean MovieLens. Works for ml-25m and ml-1m."""
import sys
import zipfile
import urllib.request
import numpy as np
import pandas as pd
from src.config import DATA_DIR, URLS, SEED, MIN_ITEM_RATINGS, MIN_USER_RATINGS


def download(name):
    DATA_DIR.mkdir(exist_ok=True)
    if (DATA_DIR / name).exists():
        print(f"{name} already downloaded")
        return
    zpath = DATA_DIR / f"{name}.zip"
    print(f"downloading {URLS[name]} ...")
    try:
        import certifi
        import ssl
        ctx = ssl.create_default_context(cafile=certifi.where())
        urllib.request.install_opener(urllib.request.build_opener(urllib.request.HTTPSHandler(context=ctx)))
    except Exception:
        pass

    def _reporthook(block_num, block_size, total_size):
        if total_size > 0 and block_num % 1000 == 0:
            downloaded = block_num * block_size
            pct = min(100.0, downloaded * 100 / total_size)
            print(f"\rDownloading: {pct:.1f}% ({downloaded / (1024*1024):.1f} MB / {total_size / (1024*1024):.1f} MB)", end="", flush=True)

    urllib.request.urlretrieve(URLS[name], zpath, reporthook=_reporthook)
    print("\nextracting ...")
    with zipfile.ZipFile(zpath) as z:
        z.extractall(DATA_DIR)
    print("done")


def clean_movies(m):
    """year from the title, genres as a list, 'Children's' -> 'Children'."""
    m = m.drop_duplicates("movieId").copy()
    m["year"] = m["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("float64")
    m["genres"] = m["genres"].apply(
        lambda g: [] if (not isinstance(g, str) or g == "(no genres listed)")
        else [x.replace("Children's", "Children") for x in g.split("|")])
    return m[["movieId", "title", "year", "genres"]]


def read_raw(name):
    d = DATA_DIR / name
    if name == "ml-1m":
        cols = ["userId", "movieId", "rating", "timestamp"]
        r = pd.read_csv(d / "ratings.dat", sep="::", engine="python", names=cols)
        m = pd.read_csv(d / "movies.dat", sep="::", engine="python",
                        names=["movieId", "title", "genres"], encoding="latin-1")
    else:  # ml-25m csv format
        r = pd.read_csv(d / "ratings.csv", dtype={"userId": "int32", "movieId": "int32",
                                                  "rating": "float32", "timestamp": "int64"})
        m = pd.read_csv(d / "movies.csv")
    return r, m


def load_dataset(name, max_users=None):
    """max_users: random sample of users, handy when RAM is tight (e.g. 50000 for ml-25m)."""
    ratings, movies = read_raw(name)
    if max_users and ratings["userId"].nunique() > max_users:
        rng = np.random.default_rng(SEED)
        keep = rng.choice(ratings["userId"].unique(), max_users, replace=False)
        ratings = ratings[ratings["userId"].isin(keep)]
    item_n = ratings["movieId"].value_counts()
    ratings = ratings[ratings["movieId"].isin(item_n[item_n >= MIN_ITEM_RATINGS].index)]
    user_n = ratings["userId"].value_counts()
    ratings = ratings[ratings["userId"].isin(user_n[user_n >= MIN_USER_RATINGS].index)]
    movies = clean_movies(movies)
    movies = movies[movies["movieId"].isin(ratings["movieId"].unique())]
    return ratings.reset_index(drop=True), movies


if __name__ == "__main__":
    download(sys.argv[1] if len(sys.argv) > 1 else "ml-25m")
