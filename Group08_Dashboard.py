"""
Group08_Dashboard.py — HONY Story Coach
CIS 434 Group 8, Simon Business School, Spring 2026
Run: streamlit run Group08_Dashboard.py
"""
import os
import re
import warnings
warnings.filterwarnings("ignore")
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

st.set_page_config(
    page_title="HONY Story Coach",
    page_icon="📷",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@400;600;700&family=Inter:wght@300;400;500&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', sans-serif; background: #FFF; color: #1A1A1A; }
  .main > div { padding-top: 2rem; padding-bottom: 3rem; }
  .hony-header { text-align: center; padding: 2.5rem 1rem 1.5rem; border-bottom: 1px solid #E8E8E8; margin-bottom: 2rem; }
  .hony-eyebrow { font-size: .7rem; font-weight: 500; letter-spacing: .25em; text-transform: uppercase; color: #999; margin-bottom: .5rem; }
  .hony-title { font-family: 'Playfair Display', Georgia, serif; font-size: 2.6rem; font-weight: 700; color: #111; line-height: 1.15; margin: 0; }
  .hony-subtitle { font-size: .92rem; font-weight: 300; color: #777; margin-top: .6rem; }
  .verdict-card { text-align: center; padding: 2rem 1.5rem; border: 1px solid #EFEFEF; border-radius: 4px; margin: 1.5rem 0; background: #FAFAFA; }
  .verdict-text { font-family: 'Playfair Display', Georgia, serif; font-size: 1.35rem; font-weight: 600; line-height: 1.3; margin: .4rem 0; }
  .verdict-meta { font-size: .78rem; color: #AAA; font-weight: 300; margin-top: .4rem; letter-spacing: .03em; }
  .rec-card { padding: 1.1rem 1.3rem; border-left: 3px solid #1A1A1A; margin-bottom: .9rem; background: #FAFAFA; border-radius: 0 4px 4px 0; }
  .rec-headline { font-family: 'Playfair Display', Georgia, serif; font-size: 1rem; font-weight: 600; color: #111; margin-bottom: .3rem; }
  .rec-detail { font-size: .87rem; color: #555; font-weight: 300; line-height: 1.6; }
  .topic-badge { display: inline-block; background: #111; color: #fff; font-size: .72rem; font-weight: 500; letter-spacing: .08em; text-transform: uppercase; padding: .3rem .8rem; border-radius: 2px; }
  .topic-perf { font-size: .78rem; color: #999; font-weight: 300; margin-top: .4rem; }
  .tag-chip { display: inline-block; border: 1px solid #DDD; color: #555; font-size: .78rem; padding: .22rem .65rem; border-radius: 20px; margin: .2rem .1rem; }
  .section-label { font-size: .65rem; font-weight: 600; letter-spacing: .2em; text-transform: uppercase; color: #AAA; margin-bottom: .7rem; margin-top: 1.6rem; }
  .hony-divider { border: none; border-top: 1px solid #EFEFEF; margin: 1.8rem 0; }
  .disclaimer { font-size: .75rem; color: #AAA; font-weight: 300; font-style: italic; line-height: 1.5; margin-top: 1rem; }
  .change-card { padding: .8rem 1.1rem; border-left: 3px solid #27AE60; margin-bottom: .6rem; background: #F8FFF8; border-radius: 0 4px 4px 0; font-size: .85rem; color: #333; font-weight: 300; line-height: 1.5; }
  .rewrite-label { font-size: .65rem; font-weight: 600; letter-spacing: .2em; text-transform: uppercase; color: #AAA; margin-bottom: .4rem; }
  .stTextArea textarea { font-family: 'Inter', sans-serif !important; font-size: .92rem !important; font-weight: 300 !important; line-height: 1.7 !important; border: 1px solid #E0E0E0 !important; border-radius: 4px !important; }
  .stButton > button { font-family: 'Inter', sans-serif !important; font-size: .8rem !important; font-weight: 500 !important; letter-spacing: .1em !important; text-transform: uppercase !important; border-radius: 3px !important; }
  .stButton > button[kind="primary"] { background: #1A1A1A !important; color: white !important; border: none !important; }
  .stButton > button[kind="secondary"] { background: white !important; color: #1A1A1A !important; border: 1px solid #CCC !important; }
  #MainMenu { visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Artifact loading ──────────────────────────────────────────────────────────
ARTIFACT_DIR = "."

@st.cache_resource
def load_artifacts():
    try:
        shap_model  = joblib.load(f"{ARTIFACT_DIR}/shap_model.pkl")
        scaler      = joblib.load(f"{ARTIFACT_DIR}/scaler.pkl")
        explainer   = joblib.load(f"{ARTIFACT_DIR}/shap_explainer.pkl")
        tv_rec      = joblib.load(f"{ARTIFACT_DIR}/tfidf_rec.pkl")
        lda         = joblib.load(f"{ARTIFACT_DIR}/lda_model.pkl")
        lda_vec     = joblib.load(f"{ARTIFACT_DIR}/lda_vectorizer.pkl")
        meta        = joblib.load(f"{ARTIFACT_DIR}/metadata.pkl")
        bench       = pd.read_csv(f"{ARTIFACT_DIR}/bench_filtered.csv")
        bench["tags_parsed"] = bench["tags"].apply(
            lambda r: [t.lower().strip() for t in str(r).split("|") if t.strip()]
            if pd.notna(r) else []
        )
        bench_matrix = tv_rec.transform(bench["text"].tolist())
        return dict(shap_model=shap_model, scaler=scaler, explainer=explainer,
                    tv_rec=tv_rec, lda=lda, lda_vec=lda_vec, meta=meta,
                    bench=bench, bench_matrix=bench_matrix, ok=True)
    except Exception as e:
        return {"ok": False, "error": str(e)}

arts = load_artifacts()
if not arts["ok"]:
    st.error(f"Could not load model artifacts.\n\n{arts['error']}")
    st.stop()

shap_model   = arts["shap_model"]
scaler       = arts["scaler"]
explainer    = arts["explainer"]
tv_rec       = arts["tv_rec"]
lda          = arts["lda"]
lda_vec      = arts["lda_vec"]
meta         = arts["meta"]
bench        = arts["bench"]
bench_matrix = arts["bench_matrix"]
FEATURES     = meta["features"]
TOPIC_LABELS = meta["topic_labels"]
analyzer     = SentimentIntensityAnalyzer()

# ── Helpers ───────────────────────────────────────────────────────────────────
def extract_features(text):
    f = {}
    f["word_count"]          = len(text.split())
    f["sentence_count"]      = max(len(re.findall(r"[.!?]+", text)), 1)
    f["avg_sentence_length"] = f["word_count"] / f["sentence_count"]
    s = analyzer.polarity_scores(text)
    f["vader_compound"] = s["compound"]
    f["vader_pos"]      = s["pos"]
    f["vader_neg"]      = s["neg"]
    f["vader_neu"]      = s["neu"]
    f["question_count"] = text.count("?")
    f["exclaim_count"]  = text.count("!")
    qc = sum(text.count(c) for c in ["\u201C", "\u201D", '"'])
    f["dialogue_ratio"]     = qc / max(f["word_count"], 1)
    fp = len(re.findall(r"\b(i|me|my|mine|we|us|our)\b", text.lower()))
    f["first_person_ratio"] = fp / max(f["word_count"], 1)
    f["uppercase_ratio"]    = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    return f


def get_recommendations(feats, shap_vals):
    """
    Generate up to 3 editorial suggestions.
    Uses SHAP to rank which issues matter most, but evaluates ALL possible
    suggestions so the user always gets meaningful feedback.
    """
    # Build all possible suggestions with their SHAP priority score
    candidates = []

    wc = feats["word_count"]
    dr = feats["dialogue_ratio"]
    fpr = feats["first_person_ratio"]
    vc = feats["vader_compound"]
    qm = feats["question_count"]
    asl = feats["avg_sentence_length"]

    # Map feature names to their SHAP values for prioritisation
    shap_map = dict(zip(FEATURES, shap_vals))

    if wc > 500:
        candidates.append((
            shap_map.get("word_count", 0),
            "📏 Consider shortening this story",
            f"This draft is {wc} words. Posts in the 250–400 word range earn the highest engagement in the HONY archive — shorter stories are more likely to be read in full and shared."
        ))
    elif wc > 400:
        candidates.append((
            shap_map.get("word_count", 0),
            "📏 Tighten the middle",
            f"At {wc} words, this story is slightly above the sweet spot. Try cutting one paragraph — the punchiest HONY posts land in 250–400 words."
        ))
    elif wc < 100:
        candidates.append((
            shap_map.get("word_count", 0),
            "📝 Expand the narrative",
            f"At {wc} words this is very short. HONY's highest-performing posts give the subject room to breathe — aim for at least 150 words."
        ))

    if dr < 0.02:
        candidates.append((
            shap_map.get("dialogue_ratio", 0),
            "💬 Add direct quotes from the subject",
            "This post has almost no direct quotation. HONY's signature is the subject's exact words — pull at least one or two of their actual lines into the story. It makes the person real to the reader."
        ))
    elif dr < 0.04:
        candidates.append((
            shap_map.get("dialogue_ratio", 0),
            "💬 Let them speak a little more",
            "There's some dialogue here, but posts with stronger first-person quotes from the subject consistently earn higher engagement. Consider adding one more direct line."
        ))

    if fpr < 0.03:
        candidates.append((
            shap_map.get("first_person_ratio", 0),
            "🎤 More first-person voice from the subject",
            "The post reads more as narration than as the subject's own voice. HONY's style puts 'I' at the centre — shift more of the story into the subject's perspective."
        ))

    if abs(vc) < 0.25:
        candidates.append((
            shap_map.get("vader_compound", 0),
            "❤️ Heighten the emotional stakes",
            "This draft reads as emotionally neutral. HONY's most-shared stories have a clear emotional core — vulnerability, grief, love, or unexpected joy. Find the moment of highest feeling and lean into it."
        ))

    if qm == 0:
        candidates.append((
            shap_map.get("question_count", 0),
            "🤔 End on an open moment",
            "The post closes with a statement. HONY stories that end on a question or an unresolved reflection tend to generate more replies, reblogs, and discussion. Consider a closing line that leaves the reader wondering."
        ))

    if asl > 22:
        candidates.append((
            shap_map.get("avg_sentence_length", 0),
            "✍️ Break up the long sentences",
            f"Average sentence length is {asl:.0f} words — on the longer side. HONY's most-read stories punch between long emotional sentences and short ones. A single short sentence at a key moment can be very powerful."
        ))

    # Always add a universal HONY craft note if we have fewer than 3
    if len(candidates) < 3:
        candidates.append((
            -0.01,
            "📸 Ground it in a specific detail",
            "The strongest HONY posts anchor an emotional truth in a single concrete detail — a job, a street corner, an object, a name. If this story feels slightly abstract, look for the one specific thing that made this person memorable."
        ))

    # Sort by SHAP magnitude (most negative = most damaging = show first)
    candidates.sort(key=lambda x: x[0])

    # Return top 3 as (headline, detail) tuples
    return [(h, d) for _, h, d in candidates[:3]]


def recommend_tags(draft, n_neighbors=20, n_tags=5):
    vec  = tv_rec.transform([draft])
    sims = cosine_similarity(vec, bench_matrix).flatten()
    idx  = [i for i in sims.argsort()[::-1][:n_neighbors] if sims[i] >= 0.05]
    if not idx:
        return []
    neighbors = bench.iloc[idx]
    mean_nc   = neighbors["note_count"].mean()
    tc, tnotes = Counter(), {}
    for _, row in neighbors.iterrows():
        for tag in row["tags_parsed"]:
            tc[tag] += 1
            tnotes.setdefault(tag, []).append(row["note_count"])
    out = [{"tag": t, "frequency": f,
            "lift": round(np.mean(tnotes[t]) / mean_nc if mean_nc > 0 else 1, 2)}
           for t, f in tc.most_common()]
    out.sort(key=lambda x: (x["frequency"], x["lift"]), reverse=True)
    return out[:n_tags]


TOPIC_ENG = {
    "Personal Struggle & Night Stories": "0.94 notes/day  ·  highest engagement",
    "Family & Parenthood":               "0.85 notes/day",
    "School & Childhood Memories":       "0.80 notes/day",
    "Philosophical Reflection":          "0.76 notes/day",
    "Fundraising & Advocacy":            "0.74 notes/day",
    "International Field Stories":       "0.66 notes/day",
    "NYC Life & Work":                   "0.46 notes/day",
    "Iran / Shahnameh Series":           "0.23 notes/day  ·  lowest engagement",
}

def infer_topic(text):
    clean = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    dist  = lda.transform(lda_vec.transform([clean]))[0]
    top   = int(dist.argmax())
    return TOPIC_LABELS.get(top, f"Topic {top}"), float(dist[top])

def get_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            key = ""
    return key

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hony-header">
  <div class="hony-eyebrow">Humans of New York</div>
  <div class="hony-title">Story Coach</div>
  <div class="hony-subtitle">Paste a draft caption to see how it might perform on Tumblr — and how to make it stronger.</div>
</div>
""", unsafe_allow_html=True)

draft = st.text_area(
    "",
    height=260,
    placeholder='"I hadn\'t talked to my mother in three years. Not because I was angry. I just didn\'t know how to begin again..."',
    label_visibility="collapsed",
)

col_btn, _ = st.columns([2, 1])
with col_btn:
    analyse = st.button("Analyse this story", type="primary", use_container_width=True)

# ── Save to session state ─────────────────────────────────────────────────────
if analyse:
    if len(draft.strip()) < 30:
        st.warning("Please paste a full story draft — at least a few sentences.")
    else:
        feats = extract_features(draft)
        x_raw = np.array([[feats[f] for f in FEATURES]])
        x_sc  = scaler.transform(x_raw)
        proba = float(shap_model.predict_proba(x_sc)[0, 1])
        sv    = explainer.shap_values(x_sc)[0]
        topic_label, _ = infer_topic(draft)
        tags  = recommend_tags(draft)
        recs  = get_recommendations(feats, sv)
        st.session_state["results"] = dict(
            proba=proba, sv=sv, x_sc=x_sc, feats=feats,
            topic_label=topic_label, tags=tags, recs=recs, draft=draft
        )
        if "rewrite" in st.session_state:
            del st.session_state["rewrite"]
        if "changes" in st.session_state:
            del st.session_state["changes"]

# ── Render from session state ─────────────────────────────────────────────────
if "results" in st.session_state:
    r           = st.session_state["results"]
    proba       = r["proba"]
    sv          = r["sv"]
    x_sc        = r["x_sc"]
    feats       = r["feats"]
    topic_label = r["topic_label"]
    tags        = r["tags"]
    recs        = r["recs"]
    draft       = r["draft"]
    conf        = max(proba, 1 - proba) * 100

    if proba >= 0.65:
        verdict, color, dot = "This story is likely to be a hit.", "#1A7340", "🟢"
    elif proba >= 0.42:
        verdict, color, dot = "This story has potential — a few small changes could lift it.", "#A05A00", "🟡"
    else:
        verdict, color, dot = "This story needs some work before posting.", "#B0002A", "🔴"

    st.markdown(f"""
    <div class="verdict-card">
      <div style="font-size:2rem; margin-bottom:.4rem;">{dot}</div>
      <div class="verdict-text" style="color:{color};">{verdict}</div>
      <div class="verdict-meta">Confidence {conf:.0f}%  ·  Based on 2,000 HONY posts  ·  Content features only</div>
    </div>
    """, unsafe_allow_html=True)

    # Always show 3 suggestions
    st.markdown('<div class="section-label">Editorial suggestions</div>', unsafe_allow_html=True)
    for headline, detail in recs:
        st.markdown(f"""
        <div class="rec-card">
          <div class="rec-headline">{headline}</div>
          <div class="rec-detail">{detail}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<hr class="hony-divider">', unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown('<div class="section-label">Narrative theme</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="topic-badge">{topic_label}</div>', unsafe_allow_html=True)
        eng = TOPIC_ENG.get(topic_label, "")
        if eng:
            st.markdown(f'<div class="topic-perf">{eng}  ·  historical average</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-label">Suggested Tumblr tags</div>', unsafe_allow_html=True)
        if tags:
            chips = "".join(f'<span class="tag-chip">#{t["tag"]}</span>' for t in tags)
            st.markdown(chips, unsafe_allow_html=True)
            st.markdown('<div class="topic-perf" style="margin-top:.5rem;">HONY used zero tags across 2,000 posts — adding tags could significantly boost discovery.</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="topic-perf">No tag suggestions for this draft.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="hony-divider">', unsafe_allow_html=True)

    # AI Rewrite section
    st.markdown('<div class="section-label">AI-assisted rewrite</div>', unsafe_allow_html=True)
    st.markdown('<div class="rec-detail" style="margin-bottom:.9rem;">Apply these suggestions automatically. The rewrite will explain exactly what was changed and why — review and edit in your own voice before publishing.</div>', unsafe_allow_html=True)

    if st.button("Rewrite for better engagement", type="secondary"):
        rec_bullets = "\n".join([f"- {h}: {d}" for h, d in recs])
        wc_target = min(feats["word_count"], 400) if feats["word_count"] > 400 else feats["word_count"]

        prompt = f"""You are an editorial assistant for Humans of New York (HONY), the narrative photography blog by Brandon Stanton.

ORIGINAL STORY:
{draft}

ISSUES IDENTIFIED BY OUR ENGAGEMENT MODEL:
{rec_bullets}

INSTRUCTIONS:
Rewrite this story to address ALL of the issues above. Be specific and structural — not cosmetic. Specifically:
- If the story is over 400 words, cut it to under {wc_target} words by removing repetition and keeping only the most essential details
- If dialogue is missing or weak, add 2-3 direct quotes from the subject in their own voice, consistent with the story
- If it lacks emotional intensity, find the single most vulnerable moment and make it the heart of the piece
- If it ends with a statement, rewrite the final sentence as a question or open reflection
- Keep Brandon Stanton's voice: first person, conversational, grounded in specific detail, intimate

YOUR RESPONSE MUST HAVE EXACTLY TWO SECTIONS:

REWRITTEN STORY:
[the full rewritten story here]

WHAT I CHANGED:
[3-5 specific bullet points explaining exactly what you changed and why, referencing the original]"""

        with st.spinner("Rewriting..."):
            try:
                import anthropic as _anthropic
                api_key = get_api_key()
                if not api_key:
                    st.error("No Anthropic API key found. Add ANTHROPIC_API_KEY to Streamlit secrets.")
                else:
                    client  = _anthropic.Anthropic(api_key=api_key)
                    message = client.messages.create(
                        model="claude-opus-4-5",
                        max_tokens=2000,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    response = message.content[0].text

                    # Parse the two sections
                    if "WHAT I CHANGED:" in response:
                        parts   = response.split("WHAT I CHANGED:")
                        rewrite = parts[0].replace("REWRITTEN STORY:", "").strip()
                        changes = parts[1].strip()
                    else:
                        rewrite = response.strip()
                        changes = ""

                    st.session_state["rewrite"] = rewrite
                    st.session_state["changes"] = changes

            except ImportError:
                st.error("anthropic package not installed. Add `anthropic` to requirements.txt")
            except Exception as e:
                st.error(f"Rewrite unavailable: {e}")

    # Show rewrite results
    if "rewrite" in st.session_state:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="rewrite-label">Original</div>', unsafe_allow_html=True)
            st.text_area("", value=st.session_state["results"]["draft"],
                         height=340, disabled=True, key="orig_txt")
            orig_wc = len(st.session_state["results"]["draft"].split())
            st.caption(f"{orig_wc} words")
        with c2:
            st.markdown('<div class="rewrite-label">Suggested rewrite</div>', unsafe_allow_html=True)
            st.text_area("", value=st.session_state["rewrite"],
                         height=340, key="rewrite_txt")
            new_wc = len(st.session_state["rewrite"].split())
            st.caption(f"{new_wc} words")

        if st.session_state.get("changes"):
            st.markdown('<div class="section-label" style="margin-top:1.4rem;">What was changed</div>', unsafe_allow_html=True)
            for line in st.session_state["changes"].split("\n"):
                line = line.strip().lstrip("-•").strip()
                if line:
                    st.markdown(f'<div class="change-card">✓ {line}</div>', unsafe_allow_html=True)

        st.markdown('<div class="disclaimer">Generated by Claude. Brandon should review and edit in his own voice before publishing.</div>', unsafe_allow_html=True)

    # SHAP expander
    with st.expander("See the data behind this prediction"):
        shap_exp = shap.Explanation(
            values=sv, base_values=float(explainer.expected_value),
            data=x_sc[0], feature_names=FEATURES)
        fig, _ = plt.subplots(figsize=(8, 4))
        shap.waterfall_plot(shap_exp, show=False)
        st.pyplot(fig, use_container_width=True)
        plt.close()
        st.markdown('<div class="disclaimer">XGBoost trained on 12 content features. Temporal features excluded to avoid era bias. Cross-validated ROC AUC: 0.85. Does not model the photograph.</div>', unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="hony-divider">', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;font-size:.68rem;color:#CCC;font-weight:300;letter-spacing:.1em;">'
    'HONY STORY COACH  ·  GROUP 8  ·  CIS 434 SOCIAL MEDIA AND TEXT ANALYTICS  ·  SIMON BUSINESS SCHOOL  ·  SPRING 2026'
    '</div>', unsafe_allow_html=True)
