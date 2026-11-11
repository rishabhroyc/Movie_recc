import numpy as np
from scipy.sparse import csr_matrix
from src.metrics import per_user_metrics, coverage, tail_share


def test_hand_computed_example():
    # user 0 liked items {1, 3}; top-3 list is [3, 0, 1]  -> hits at ranks 1 and 3
    rel = csr_matrix(np.array([[0, 1, 0, 1, 0]], dtype=np.float32))
    topk = np.array([[3, 0, 1]])
    m = per_user_metrics(topk, rel, np.array([0]))
    assert m["recall"][0] == 1.0                                   # both liked items found
    dcg = 1 / np.log2(2) + 1 / np.log2(4)                          # ranks 1 and 3
    ideal = 1 / np.log2(2) + 1 / np.log2(3)                        # both liked items at ranks 1 and 2
    assert abs(m["ndcg"][0] - dcg / ideal) < 1e-9
    assert m["hit"][0]


def test_no_hit_gives_zero_and_partial_recall():
    rel = csr_matrix(np.array([[1, 1, 0, 0]], dtype=np.float32))
    assert per_user_metrics(np.array([[2, 3]]), rel, np.array([0]))["recall"][0] == 0.0
    assert per_user_metrics(np.array([[0, 3]]), rel, np.array([0]))["recall"][0] == 0.5


def test_coverage_and_tail_share():
    topk = np.array([[0, 1], [1, 2]])
    assert coverage(topk, 10) == 0.3
    tail = np.array([False, False, True, True])
    assert tail_share(topk, tail) == 0.25
