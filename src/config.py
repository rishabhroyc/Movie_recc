"""Fixed decisions for the whole project live here, so nothing is scattered."""
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
ART_DIR = ROOT / "artifacts"
RES_DIR = ROOT / "results"

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    load_dotenv()
except Exception:
    pass

SEED = 42
K = 10                    # metric cut-off
REL_THRESHOLD = 4.0       # rating >= 4 counts as "liked"
SPLIT = (0.70, 0.85)      # per user, by time: <70% train, 70-85% val, rest test
MIN_ITEM_RATINGS = 5      # drop movies with fewer ratings than this
MIN_USER_RATINGS = 10     # drop users with fewer ratings than this
MIN_RATINGS_SERVE = 20    # never recommend a movie with fewer ratings than this
HEAD_FRACTION = 0.20      # top 20% most popular movies = "head", the rest = "tail"
ALS_CONFIDENCE = 40.0     # implicit ALS needs confidence weights, plain 1s carry no signal

URLS = {
    "ml-1m": "https://files.grouplens.org/datasets/movielens/ml-1m.zip",
    "ml-25m": "https://files.grouplens.org/datasets/movielens/ml-25m.zip",
}

GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime", "Documentary",
          "Drama", "Fantasy", "Film-Noir", "Horror", "IMAX", "Musical", "Mystery", "Romance",
          "Sci-Fi", "Thriller", "War", "Western"]
