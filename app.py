import json
import pandas as pd
import streamlit as st
from src.config import ART_DIR, RES_DIR
from src.models import load_bundle
from src.recommend import Recommender
from src.llm import parse_request, explain
from src.guardrails import looks_like_injection

st.set_page_config(page_title="ReadNext", layout="wide")


@st.cache_resource
def get_rec():
    bundle_path = ART_DIR / "bundle.pkl"
    if not bundle_path.exists():
        return None
    return Recommender(load_bundle(bundle_path))


rec = get_rec()
if rec is None:
    st.error("Model bundle not found at artifacts/bundle.pkl. Run `python -m scripts.train` first.")
    st.stop()

tab_demo, tab_results = st.tabs(["Try it", "Results"])

with tab_demo:
    st.title("ReadNext")
    st.caption("Recommender system on MovieLens 25M with natural language request filtering.")

    with st.sidebar:
        st.subheader("User settings")
        new_user = st.checkbox("New user (no history)", value=False)
        user_id = None if new_user else st.number_input("User ID", value=int(rec.user_ids[0]), step=1)
        k_items = st.slider("Recommendations count", min_value=1, max_value=10, value=5)

    text = st.text_input("What do you feel like watching?", placeholder="e.g. funny to watch with friends, or 90s thriller no horror")

    if st.button("Recommend") and text:
        if looks_like_injection(text):
            st.warning("That looks like it's trying to change instructions. Treating input as search query only.")

        req = parse_request(text)

        with st.expander("Parsed filters"):
            st.json(json.loads(req.model_dump_json()))

        out = rec.recommend(user_id, req, k=k_items)
        st.write(f"**{out['message']}**")
        for n in out["notes"]:
            st.info(n)

        if out["items"]:
            for it, ex in zip(out["items"], explain(out["items"])):
                st.subheader(it["title"])
                st.caption(f"{', '.join(it['genres'])} | Avg rating: {it['avg_rating']}/5 ({it['n_ratings']:,} ratings)")
                st.write(ex["reason"])
                if ex["fallback"]:
                    st.caption("(Template reason)")

with tab_results:
    for name in ("results.csv", "slices.csv", "llm_eval.csv"):
        p = RES_DIR / name
        if p.exists():
            st.subheader(name)
            st.dataframe(pd.read_csv(p).round(4))

    p = RES_DIR / "ab_results.json"
    if p.exists():
        st.subheader("Simulated A/B test (offline replay)")
        st.json(json.loads(p.read_text()))
