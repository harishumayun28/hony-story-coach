"""
tag_recommender.py
------------------
Person B deliverable — CIS 434 Group 8, Milestone 2

Cosine-similarity tag recommender for HONY pre-publish tool.

Usage
-----
    from tag_recommender import TagRecommender

    rec = TagRecommender(
        benchmark_csv="benchmark_tagged_posts.csv",   # Person A's output
        hony_texts=hony_df["text"].tolist(),           # HONY archive texts (for fitting TF-IDF)
    )

    results = rec.recommend_tags(draft_text="He told me he hadn't spoken to his son in six years...")
    for r in results:
        print(r)

Output schema (list of dicts, sorted by expected_lift desc)
------------------------------------------------------------
    [
        {"tag": "personal narrative", "frequency": 14, "expected_lift": 2.31},
        {"tag": "storytelling",       "frequency": 12, "expected_lift": 1.87},
        ...
    ]
"""

import ast
import warnings
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# How many benchmark neighbors to pull for tag voting
N_NEIGHBORS = 20

# How many top tags to return
N_TOP_TAGS = 5

# Minimum cosine similarity to count a neighbor as relevant
# (guards against returning tags from totally unrelated posts)
MIN_SIMILARITY = 0.05


# ---------------------------------------------------------------------------
# Helper — parse tags column
# ---------------------------------------------------------------------------

def _parse_tags(raw) -> list[str]:
    """
    Robustly parse a tags value. Person A's CSV uses pipe-delimited strings
    (e.g. "storytelling|writer|personal narrative"), so we check for '|' first.
    Also handles list literals, comma-separated, actual lists, and NaN/None.
    Returns a list of lowercase stripped strings.
    """
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return []
    if isinstance(raw, list):
        return [t.lower().strip() for t in raw if t]
    raw = str(raw).strip()
    # Person A's format: pipe-delimited
    if "|" in raw:
        return [t.lower().strip() for t in raw.split("|") if t.strip()]
    # list-literal format
    if raw.startswith("["):
        try:
            parsed = ast.literal_eval(raw)
            return [t.lower().strip() for t in parsed if t]
        except (ValueError, SyntaxError):
            pass
    # fallback: comma-separated
    return [t.lower().strip() for t in raw.split(",") if t.strip()]


# ---------------------------------------------------------------------------
# Core class
# ---------------------------------------------------------------------------

class TagRecommender:
    """
    Fits a shared TF-IDF vectorizer on HONY archive texts (so the vocabulary
    matches the main model), then uses cosine similarity to find benchmark
    posts that look like a given draft and recommends their tags.

    Parameters
    ----------
    benchmark_csv : str
        Path to Person A's benchmark CSV.
        Required columns: post_id, text, note_count, tags, timestamp
    hony_texts : list[str]
        All HONY post texts used to fit the shared TF-IDF vocabulary.
        Pass your full hony_df["text"].tolist() here.
    tfidf_kwargs : dict, optional
        Extra keyword arguments forwarded to TfidfVectorizer.
        Defaults are set to match typical NLP text settings.
    """

    def __init__(
        self,
        benchmark_csv: str,
        hony_texts: list[str],
        tfidf_kwargs: dict | None = None,
    ):
        self._load_benchmark(benchmark_csv)
        self._fit_vectorizer(hony_texts, tfidf_kwargs or {})
        self._vectorize_benchmark()

    # ------------------------------------------------------------------
    # Setup methods
    # ------------------------------------------------------------------

    def _load_benchmark(self, path: str) -> None:
        """Load and validate Person A's benchmark CSV."""
        df = pd.read_csv(path)

        required = {"post_id", "text", "note_count", "tags", "timestamp"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"benchmark CSV is missing columns: {missing}\n"
                f"Found columns: {list(df.columns)}"
            )

        # Drop rows with missing text (can't vectorize them)
        before = len(df)
        df = df.dropna(subset=["text"]).reset_index(drop=True)
        if len(df) < before:
            warnings.warn(f"Dropped {before - len(df)} rows with missing text.")

        # Ensure note_count is numeric
        df["note_count"] = pd.to_numeric(df["note_count"], errors="coerce").fillna(0)

        # Parse tags into lists
        df["tags_parsed"] = df["tags"].apply(_parse_tags)

        # High-engagement filter: keep only top-quartile posts by note_count.
        # Person A's benchmark is heavily skewed (75th pct = 5 notes, max = 24580),
        # so we restrict to posts above the 75th percentile so cosine neighbors
        # are drawn from genuinely well-performing content, not zero-note posts.
        q75 = df["note_count"].quantile(0.75)
        n_before = len(df)
        if q75 > 0:
            df = df[df["note_count"] >= q75].reset_index(drop=True)
            print(f"[TagRecommender] High-engagement filter: kept {len(df)}/{n_before} posts "
                  f"(note_count >= {q75:.0f}, 75th pct).")
        else:
            # If 75th pct is 0 (lots of zero-note posts), use top-25% positive
            df_pos = df[df["note_count"] > 0]
            if len(df_pos) > 0:
                q75_pos = df_pos["note_count"].quantile(0.75)
                df = df[df["note_count"] >= q75_pos].reset_index(drop=True)
                print(f"[TagRecommender] High-engagement filter (positive only): kept "
                      f"{len(df)}/{n_before} posts (note_count >= {q75_pos:.0f}).")
            else:
                print("[TagRecommender] Warning: all posts have 0 notes. No filter applied.")

        self.benchmark = df
        print(f"[TagRecommender] Loaded {len(df)} benchmark posts from '{path}'.")

    def _fit_vectorizer(self, hony_texts: list[str], kwargs: dict) -> None:
        """Fit TF-IDF on HONY archive so vocabulary is consistent with the main model."""
        # min_df=2 is safe for large corpora (2000+ HONY posts);
        # for very small corpora it prunes everything, so we clamp it.
        n_docs = len(hony_texts)
        safe_min_df = min(2, max(1, n_docs // 10))

        defaults = dict(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=safe_min_df,
            strip_accents="unicode",
            stop_words="english",
        )
        defaults.update(kwargs)

        self.vectorizer = TfidfVectorizer(**defaults)
        self.vectorizer.fit(hony_texts)
        print(
            f"[TagRecommender] TF-IDF fitted on {len(hony_texts)} HONY texts. "
            f"Vocabulary size: {len(self.vectorizer.vocabulary_):,}."
        )

    def _vectorize_benchmark(self) -> None:
        """Transform all benchmark posts into TF-IDF space (done once at init)."""
        self.benchmark_matrix = self.vectorizer.transform(
            self.benchmark["text"].tolist()
        )
        print("[TagRecommender] Benchmark matrix ready.")

    # ------------------------------------------------------------------
    # Main public method
    # ------------------------------------------------------------------

    def recommend_tags(
        self,
        draft_text: str,
        n_neighbors: int = N_NEIGHBORS,
        n_tags: int = N_TOP_TAGS,
        min_similarity: float = MIN_SIMILARITY,
    ) -> list[dict]:
        """
        Given a draft post text, return the top N tags most associated with
        similar high-engagement benchmark posts.

        Parameters
        ----------
        draft_text : str
            The raw text of the draft post.
        n_neighbors : int
            How many most-similar benchmark posts to consider (default 20).
        n_tags : int
            How many tags to return in the final ranked list (default 5).
        min_similarity : float
            Minimum cosine similarity threshold; neighbors below this are ignored.

        Returns
        -------
        list[dict]  — sorted by expected_lift descending
            Each dict has:
                tag            : str   — the recommended tag
                frequency      : int   — how many of the top neighbors used this tag
                expected_lift  : float — mean note_count of neighbors using this tag
                                        divided by mean note_count of all neighbors
                                        (>1.0 means posts using this tag score higher)
        """
        if not draft_text or not draft_text.strip():
            raise ValueError("draft_text cannot be empty.")

        # Step 1 — vectorize the draft
        draft_vec = self.vectorizer.transform([draft_text])

        # Step 2 — cosine similarity against all benchmark posts
        sims = cosine_similarity(draft_vec, self.benchmark_matrix).flatten()

        # Step 3 — rank and filter by minimum threshold
        ranked_idx = np.argsort(sims)[::-1]
        neighbor_idx = [
            i for i in ranked_idx[:n_neighbors]
            if sims[i] >= min_similarity
        ]

        if not neighbor_idx:
            warnings.warn(
                "No benchmark posts met the minimum similarity threshold. "
                "Returning empty list. Try lowering min_similarity or "
                "check that your benchmark CSV covers similar topics."
            )
            return []

        # Step 4 — collect tags from neighbors
        neighbors = self.benchmark.iloc[neighbor_idx].copy()
        neighbors["similarity"] = [sims[i] for i in neighbor_idx]

        mean_note_all_neighbors = neighbors["note_count"].mean()

        # Step 5 — count tag frequency and compute expected lift
        tag_counter = Counter()
        tag_note_sums = {}   # tag -> list of note_counts of posts that used it

        for _, row in neighbors.iterrows():
            for tag in row["tags_parsed"]:
                tag_counter[tag] += 1
                tag_note_sums.setdefault(tag, []).append(row["note_count"])

        # Step 6 — build output list
        results = []
        for tag, freq in tag_counter.most_common():
            mean_notes_with_tag = np.mean(tag_note_sums[tag])
            lift = (
                mean_notes_with_tag / mean_note_all_neighbors
                if mean_note_all_neighbors > 0 else 1.0
            )
            results.append({
                "tag": tag,
                "frequency": freq,
                "expected_lift": round(float(lift), 3),
            })

        # Sort: primary = frequency (vote count), secondary = lift
        results.sort(key=lambda x: (x["frequency"], x["expected_lift"]), reverse=True)

        return results[:n_tags]

    # ------------------------------------------------------------------
    # Convenience: human-readable summary
    # ------------------------------------------------------------------

    def format_recommendations(self, draft_text: str) -> str:
        """
        Returns a plain-text summary suitable for the Streamlit dashboard
        recommendations panel.
        """
        recs = self.recommend_tags(draft_text)
        if not recs:
            return "No tag recommendations available for this draft."

        lines = ["**Recommended tags** (based on similar high-engagement posts):\n"]
        for i, r in enumerate(recs, 1):
            lift_label = (
                f"+{(r['expected_lift'] - 1) * 100:.0f}% engagement lift"
                if r["expected_lift"] >= 1.0
                else f"{(r['expected_lift'] - 1) * 100:.0f}% vs. baseline"
            )
            lines.append(
                f"  {i}. `#{r['tag']}` — "
                f"used by {r['frequency']}/{N_NEIGHBORS} similar posts — "
                f"est. {lift_label}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Quick self-test (run: python tag_recommender.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import textwrap

    # Simulate a tiny benchmark CSV in memory for the smoke test
    import io

    FAKE_CSV = """post_id,text,note_count,tags,timestamp
1,"He told me he had not spoken to his daughter in six years. He said he wrote letters but they went unanswered.",450,"['personal narrative','storytelling','family stories']",2019-03-01
2,"She was sitting on the stoop with a cigarette, watching the pigeons.",180,"['new york city','street photography','portrait photography']",2018-07-15
3,"I met him outside the shelter. He said today was his birthday and nobody knew.",920,"['human interest','personal narrative','nyc']",2020-01-10
4,"They had been married for fifty two years. She still made his coffee first.",310,"['family stories','love','personal narrative']",2017-11-20
5,"He came to the US from Lagos in 1992 with forty dollars and a phone number on a napkin.",670,"['human interest','immigrant stories','storytelling']",2021-05-05
"""

    HONY_TEXTS = [
        "He told me he had not spoken to his daughter in six years.",
        "She came to New York at seventeen with nothing but a suitcase.",
        "I asked him what he was most proud of. He looked at his hands.",
        "They met in a refugee camp in 1989. Today they are citizens.",
        "She said the city taught her how to be alone without being lonely.",
    ]

    # Write temp CSV
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
        f.write(FAKE_CSV)
        tmp_path = f.name

    rec = TagRecommender(benchmark_csv=tmp_path, hony_texts=HONY_TEXTS)

    draft = (
        "She told me she hadn't called her mother in three years. "
        "Not because she was angry, she said. Just because she didn't "
        "know how to begin again."
    )

    print("\n--- recommend_tags() raw output ---")
    for r in rec.recommend_tags(draft):
        print(r)

    print("\n--- format_recommendations() output ---")
    print(textwrap.dedent(rec.format_recommendations(draft)))

    os.unlink(tmp_path)
    print("\nSmoke test passed.")
