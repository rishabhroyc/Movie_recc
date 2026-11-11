"""Rule-based checks. No LLM in here, so everything is testable."""
import re
from src.config import GENRES

GENRE_SYNONYMS = {"science fiction": "Sci-Fi", "sci fi": "Sci-Fi", "scifi": "Sci-Fi"}
YEAR_RE = re.compile(r"\b(1[89]\d{2}|20\d{2})\b")
FLOAT_RE = re.compile(r"\b\d\.\d\b")
QUOTE_RE = re.compile(r'["\u201c\u201d]([^"\u201c\u201d]{2,80})["\u201c\u201d]')
INJECTION_HINTS = ["ignore previous", "ignore all previous", "ignore your instructions", "system prompt",
                   "reveal your prompt", "disregard the above", "you are now"]


def norm_title(t):
    """'Godfather, The (1972)' and 'the godfather' both become 'godfather'."""
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", t or "")
    t = re.sub(r"\s*\(.*\)\s*$", "", t)                       # alternative titles in brackets
    m = re.match(r"^(.*), (The|A|An)$", t.strip())
    if m:
        t = m.group(1)
    t = re.sub(r"[^a-z0-9 ]", "", t.lower()).strip()
    return re.sub(r"^(the|a|an) ", "", t)


def looks_like_injection(text):
    low = (text or "").lower()
    return any(h in low for h in INJECTION_HINTS)


def fact_problems(reason, facts):
    """Everything in `reason` that is not backed by `facts`. Empty list = the reason is clean."""
    problems = []
    for y in YEAR_RE.findall(reason):
        if facts.get("year") is None or int(y) != int(facts["year"]):
            problems.append(f"year {y}")
    own = {g.lower() for g in facts.get("genres", [])}
    low = reason.lower()
    for g in GENRES:
        if re.search(rf"\b{re.escape(g.lower())}\b", low) and g.lower() not in own:
            problems.append(f"genre {g}")
    for phrase, g in GENRE_SYNONYMS.items():
        if phrase in low and g.lower() not in own:
            problems.append(f"genre {g}")
    for f in FLOAT_RE.findall(reason):
        if facts.get("avg_rating") is None or abs(float(f) - float(facts["avg_rating"])) > 0.051:
            problems.append(f"rating {f}")
    allowed = {norm_title(facts.get("title")), norm_title(facts.get("similar_title"))}
    for q in QUOTE_RE.findall(reason):
        if norm_title(q) not in allowed:
            problems.append(f"title {q}")
    return problems


def fact_check(reason, facts):
    return len(fact_problems(reason, facts)) == 0


def template_reason(facts):
    """Boring but always true. Used whenever the LLM text fails the fact check."""
    g = ", ".join(facts.get("genres") or []) or "no genre listed"
    year = f" ({facts['year']})" if facts.get("year") else ""
    out = f"{facts['title'].split(' (')[0]}{year} is tagged {g} and rated {facts['avg_rating']}/5 on average."
    if facts.get("similar_title"):
        out += f" Similar to {facts['similar_title']}, which you liked."
    return out


def catalog_check(titles, allowed_titles):
    """Keep only titles that are really in the candidate list (stops invented movies)."""
    allowed = set(allowed_titles)
    return [t for t in titles if t in allowed]
