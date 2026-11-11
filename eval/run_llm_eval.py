"""Measures the two LLM parts. Needs an LLM (set LLM_PROVIDER / LLM_MODEL) and a trained bundle.
  python -m eval.run_llm_eval
Writes results/llm_eval.csv and prints every failure so you can read them (your best interview material)."""
import json
import pathlib
import numpy as np
import pandas as pd
from src.config import RES_DIR, ART_DIR, SEED
from src.models import load_bundle
from src.recommend import Recommender
from src import llm

YEAR_TOL = 1   # "before 1970" can reasonably mean 1969 or 1970


def _year_ok(exp, got):
    if exp is None:
        return got is None
    return got is not None and abs(exp - got) <= YEAR_TOL


def matches(expect, got):
    """Does the parsed request (object with attributes) match what we expected?"""
    for key in ("include_genres", "exclude_genres"):
        if set(expect.get(key, [])) != set(getattr(got, key)):
            return False
    if not _year_ok(expect.get("year_min"), got.year_min) or not _year_ok(expect.get("year_max"), got.year_max):
        return False
    sim = expect.get("similar_to_title")
    if sim is None:
        if got.similar_to_title is not None:
            return False
    elif not got.similar_to_title or sim.lower() not in got.similar_to_title.lower():
        return False
    return bool(expect.get("off_topic", False)) == bool(got.off_topic)


def run_parser_eval(cases, call=None):
    call = call or llm.call_llm
    rows, failures = [], []
    for c in cases:
        log = []
        def spy(system, user, _log=log):
            out = call(system, user)
            _log.append(out)
            return out
        got = llm.parse_request(c["q"], call=spy)
        if c["type"] == "adversarial":
            ok = not any(c["canary"].lower() in o.lower() for o in log)   # model must not obey or leak
        elif c["type"] == "offtopic":
            ok = bool(got.off_topic)
        else:
            ok = matches(c["expect"], got)
        rows.append({"type": c["type"], "ok": ok, "retried": len(log) > 1, "parse_failed": got.parse_failed})
        if not ok:
            failures.append((c["q"], got.model_dump() if hasattr(got, "model_dump") else got, log[-1][:200] if log else ""))
    return pd.DataFrame(rows), failures


def run_explainer_eval(n_sets=20, call=None):
    from src.schema import UserRequest
    rec = Recommender(load_bundle(ART_DIR / "bundle.pkl"))
    rng = np.random.default_rng(SEED)
    users = rng.choice(rec.user_ids, n_sets, replace=False)
    fb = []
    for u in users:
        items = rec.recommend(int(u), UserRequest(), k=5)["items"]
        fb += [e["fallback"] for e in llm.explain(items, call=call)]
    return float(np.mean(fb)), len(fb)


def main():
    cases = json.loads((pathlib.Path(__file__).parent / "llm_testset.json").read_text())
    df, failures = run_parser_eval(cases)
    out = []
    for t, label in [("normal", "parse_accuracy_normal"), ("offtopic", "offtopic_detection"),
                     ("vague", "vague_handled_ok"), ("adversarial", "injection_resilience")]:
        sub = df[df["type"] == t]
        out.append({"metric": label, "value": sub["ok"].mean(), "n": len(sub)})
    out.append({"metric": "json_retry_rate", "value": df["retried"].mean(), "n": len(df)})
    out.append({"metric": "parse_failed_rate", "value": df["parse_failed"].mean(), "n": len(df)})
    fb_rate, n_reasons = run_explainer_eval()
    out.append({"metric": "explainer_fallback_rate", "value": fb_rate, "n": n_reasons})
    out.append({"metric": "explainer_first_pass_rate", "value": 1 - fb_rate, "n": n_reasons})
    res = pd.DataFrame(out)
    RES_DIR.mkdir(exist_ok=True)
    res.to_csv(RES_DIR / "llm_eval.csv", index=False)
    print(res.round(3).to_string(index=False))
    print("\nFAILURES (read every one):")
    for q, got, raw in failures:
        print(f"- {q!r}\n    parsed: {got}\n    raw: {raw!r}")


if __name__ == "__main__":
    main()
