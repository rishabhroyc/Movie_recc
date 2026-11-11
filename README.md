# ReadNext

A movie recommender on **MovieLens 25M** (25M ratings, ~162k users, ~62k movies) that also understands
plain-English requests like *"something like Heat but lighter, nothing before 1990"*.

**The rule that keeps it honest: the recommender picks the movies, the LLM never does.** The LLM only
(1) turns the request into a validated filter object and (2) writes one short reason per pick using facts
we pass in. Every reason is fact-checked against the data; failures are replaced with a template sentence.

## Architecture

```
BUILD   ratings -> per-user time split -> popularity / ALS / genre(+tag genome) content -> hybrid scores
SERVE   request -> LLM parser (Pydantic) -> filter + rank (no LLM) -> LLM explainer -> fact check -> answer
CHECK   offline metrics + user slices | 25-query LLM test set | simulated A/B | pytest | Streamlit demo
```

## Setup and run

```bash
pip install -r requirements.txt

# 1. Download dataset (if not already downloaded)
python -m src.data ml-25m

# 2. Train and evaluate recommender
python -m scripts.train --dataset ml-25m --eval-users 10000
python -m scripts.ab_sim
python -m pytest

# 3. LLM parts (configured via .env file or env vars)
# Copy .env.example to .env and set your API key:
#   LLM_PROVIDER=grok LLM_MODEL=grok-2-latest GROK_API_KEY=<your-key>
#   or LLM_PROVIDER=openai / anthropic / groq
python -m eval.run_llm_eval
streamlit run app.py
```

Tight on RAM? `--max-users 50000` samples users (about 8M ratings). `--no-genome` skips the tag-genome features.
`--no-grid` skips the ALS grid search.

## Fixed decisions

| Decision | Choice |
|---|---|
| Relevant item | rating >= 4 |
| Split | per user, by time: first 70% train, next 15% val, last 15% test |
| Cleaning | drop movies with < 5 ratings, users with < 10 ratings |
| Candidates | never recommend a movie the user already rated (train+val when scoring test) |
| Tuning | ALS grid and hybrid alpha chosen on val only; test is evaluated once |
| Metrics | Recall@10, NDCG@10, coverage, share of recommendations from the tail (least popular 80%) |
| Serving | movies with < 20 ratings are never recommended |

## Results (from results/*.csv and results/ab_results.json)

| Model (test) | Recall@10 | NDCG@10 | Coverage@10 | Tail share |
|---|---|---|---|---|
| Popularity | 0.0475 | 0.0562 | 0.0066 | 0.0000 |
| ALS | 0.0801 | 0.0877 | 0.0410 | 0.0000 |
| Content (genres + decade + genome) | 0.0086 | 0.0100 | 0.1343 | 0.6035 |
| Hybrid (alpha=0.75) | 0.0755 | 0.0848 | 0.0518 | 0.0022 |

Weakest slice (light vs heavy users): Content-based model on light users (NDCG 0.0037 vs 0.0169 for heavy users). Across ALS and Hybrid, heavy users achieve nearly 2x NDCG (0.1348 vs 0.0709) thanks to rich interaction history.

LLM layer (25-query test set, 20 explanation sets from eval/run_llm_eval.py):
- Parse accuracy (normal): 100% (15/15)
- Off-topic detection: 100% (3/3)
- Injection resilience: 100% (5/5)
- Vague request handling: 100% (2/2)
- Parse failed rate: 0.0%
- JSON retry rate: 0.0%
- Explainer first pass rate: 10.0% (strict fact-checker caught unverified claims, falling back safely to grounded templates)

Simulated A/B (popularity vs hybrid, offline replay, **not real traffic**):
- Arm A (Popularity): 26.4% hit rate
- Arm B (Hybrid): 37.2% hit rate
- Lift: +10.8 percentage points (95% CI: [5.1%, 16.5%], p = 0.00025)
- A/A false positive rate: 4.2% (close to 5% expectation)
- Diversity guardrail: Arm A 10.9 distinct genres, Arm B 9.2 distinct genres

## A failure I found

Content-based cold start on light users: Users in the bottom 25% of ratings saw content model NDCG drop to 0.0037 (vs 0.0169 for heavy users). The content model needs a sufficiently rich profile of liked movies to compute genre/genome affinities; with few ratings, recommendations degenerate toward obscure tail items (60% tail share) that miss user taste. The hybrid model mitigates this by blending collaborative filtering signals. Additionally, the strict explainer fact checker flags claims containing years or genres unverified in the input facts, falling back to deterministic template sentences.

## Guardrails and how they are tested

Seen movies hidden, catalogue/year/genre filters enforced, new users get popular picks and are told so,
off-topic requests refused, user text wrapped as data for the LLM and truncated to 500 characters,
parser output validated by Pydantic with one retry then a safe fallback, explanations fact-checked
(years, genres, ratings, quoted titles) with template fallback. Each one has a test in `tests/`
(the LLM is mocked there, so tests need no API key).

## Limitations

- MovieLens is ratings, not clicks, and the data is old; results may not transfer to a real product.
- The A/B test is an offline replay.
- The LLM test set has 25 queries I wrote myself.
- Per-user time split, not a single global cut-off.
- ALS is trained on the train split only; no retraining on train+val for the final model.
- No new-movie (item cold-start) handling; the fact check is rule based and can miss subtle wrong claims.

## What I'd do next

LightGBM re-ranker, sentence embeddings for movie text, real traffic / online test, CI that fails when a metric drops.
