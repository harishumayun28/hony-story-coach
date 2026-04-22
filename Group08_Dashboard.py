"""
Group08_Dashboard.py
HONY Story Coach — Pre-Publish Engagement Advisor
CIS 434 Group 8, Simon Business School, Spring 2026

Run: streamlit run Group08_Dashboard.py
Requires: artifacts/ folder from Group08_Analysis.ipynb
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

  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #FFFFFF;
    color: #1A1A1A;
  }
  .main > div { padding-top: 2rem; padding-bottom: 3rem; }

  .hony-header {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
    border-bottom: 1px solid #E8E8E8;
    margin-bottom: 2rem;
  }
  .hony-eyebrow {
    font-size: 0.7rem;
    font-weight: 500;
    letter-spacing: 0.25em;
    text-transform: uppercase;
    color: #999;
    margin-bottom: 0.5rem;
  }
  .hony-title {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 2.6rem;
    font-weight: 700;
    color: #111;
    line-height: 1.15;
    margin: 0;
  }
  .hony-subtitle {
    font-size: 0.92rem;
    font-weight: 300;
    color: #777;
    margin-top: 0.6rem;
  }

  .verdict-card {
    text-align: center;
    padding: 2rem 1.5rem;
    border: 1px solid #EFEFEF;
    border-radius: 4px;
    margin: 1.5rem 0;
    background: #FAFAFA;
  }
  .verdict-text {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1.35rem;
    font-weight: 600;
    line-height: 1.3;
    margin: 0.4rem 0;
  }
  .verdict-meta {
    font-size: 0.78rem;
    color: #AAA;
    font-weight: 300;
    margin-top: 0.4rem;
    letter-spacing: 0.03em;
  }

  .rec-card {
    padding: 1.1rem 1.3rem;
    border-left: 3px solid #1A1A1A;
    margin-bottom: 0.9rem;
    background: #FAFAFA;
    border-radius: 0 4px 4px 0;
  }
  .rec-headline {
    font-family: 'Playfair Display', Georgia, serif;
    font-size: 1rem;
    font-weight: 600;
    color: #111;
    margin-bottom: 0.3rem;
  }
  .rec-detail {
    font-size: 0.87rem;
    color: #555;
    font-weight: 300;
    line-height: 1.6;
  }

  .topic-badge {
    display: inline-block;
    background: #111;
    color: #fff;
    font-size: 0.72rem;
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    padding: 0.3rem 0.8rem;
    border-radius: 2px;
  }
  .topic-perf {
    font-size: 0.78rem;
    color: #999;
    font-weight: 300;
    margin-top: 0.4rem;
  }

  .tag-chip {
    display: inline-block;
    border: 1px solid #DDD;
    color: #555;
    font-size: 0.78rem;
    font-weight: 400;
    padding: 0.22rem 0.65rem;
    border-radius: 20px;
    margin: 0.2rem 0.1rem;
  }

  .section-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #AAA;
    margin-bottom: 0.7rem;
    margin-top: 1.6rem;
  }

  .hony-divider {
    border: none;
    border-top: 1px solid #EFEFEF;
    margin: 1.8rem 0;
  }

  .disclaimer {
    font-size: 0.75rem;
    color: #AAA;
    font-weight: 300;
    font-style: italic;
    line-height: 1.5;
    margin-top: 1rem;
  }

  .rewrite-label {
    font-size: 0.65rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #AAA;
    margin-bottom: 0.4rem;
  }

  .stTextArea textarea {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.92rem !important;
    font-weight: 300 !important;
    line-height: 1.7 !important;
    color: #1A1A1A !important;
    border: 1px solid #E0E0E0 !important;
    border-radius: 4px !important;
    background: #FEFEFE !important;
  }
  .stButton > button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 3px !important;
  }
  .stButton > button[kind="primary"] {
    background: #1A1A1A !important;
    color: white !important;
    border: none !important;
  }
  .stButton > button[kind="secondary"] {
    background: white !important;
    color: #1A1A1A !important;
    border: 1px solid #CCC !important;
  }

  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Load artifacts ────────────────────────────────────────────────────────────
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
                    bench=bench, bench_matrix=bench_matrix)
    except FileNotFoundError as e:
        return {"error": str(e)}


arts = load_artifacts()
if "error" in arts:
    st.error(f"Could not load model artifacts. Run Group08_Analysis.ipynb first.\n\n{arts['error']}")
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
    f["dialogue_ratio"]      = qc / max(f["word_count"], 1)
    fp = len(re.findall(r"\b(i|me|my|mine|we|us|our)\b", text.lower()))
    f["first_person_ratio"]  = fp / max(f["word_count"], 1)
    f["uppercase_ratio"]     = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    return f


def get_recommendations(feats, shap_vals):
    contrib = sorted(zip(FEATURES, shap_vals, [feats[f] for f in FEATURES]), key=lambda x: x[1])
    recs = []
    for fname, sval, fval in contrib:
        if sval >= 0:
            break
        if fname == "word_count":
            if fval > 450:
                recs.append(("📏 Consider shortening this story",
                    f"Stories under 400 words earn significantly more engagement in the HONY archive. "
                    f"Yours is {fval:.0f} words — try trimming the middle."))
            elif fval < 100:
                recs.append(("📝 This story could use more depth",
                    f"Very short posts miss the narrative richness HONY readers come for. "
                    f"Yours is {fval:.0f} words — give the subject more room to breathe."))
        elif fname == "dialogue_ratio" and fval < 0.03:
            recs.append(("💬 Add more of their exact words",
                "Posts with strong first-person dialogue from the subject earn higher engagement. "
                "Let them speak more directly — pull a key line from the conversation."))
        elif fname == "first_person_ratio" and fval < 0.04:
            recs.append(("🎤 Let the subject's voice lead",
                "HONY's signature is raw, intimate first-person voice. More 'I' and 'my' "
                "language draws the reader in."))
        elif fname == "vader_compound" and abs(fval) < 0.3:
            recs.append(("❤️ Find the emotional core",
                "This draft reads as fairly neutral. Posts with a clear emotional "
                "undercurrent — tenderness, grief, or joy — consistently outperform."))
        elif fname == "question_count" and fval == 0:
            recs.append(("🤔 End on an open moment",
                "A reflective question or unresolved thought at the end invites "
                "the reader to stay with the story and drives more reblogs."))
        elif fname == "avg_sentence_length" and fval > 25:
            recs.append(("✍️ Vary the sentence rhythm",
                f"Average sentence length is {fval:.0f} words. Short punchy sentences "
                "mixed with longer reflective ones give HONY stories their distinctive pace."))
        if len(recs) >= 3:
            break
    return recs


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


# ── UI ────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hony-header">
  <div class="hony-eyebrow">Humans of New York</div>
  <div class="hony-title">Story Coach</div>
  <div class="hony-subtitle">Paste a draft caption below to see how it might perform on Tumblr — and how to make it stronger.</div>
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

if analyse:
    if len(draft.strip()) < 30:
        st.warning("Please paste a full story draft — at least a few sentences.")
    else:
        feats  = extract_features(draft)
        x_raw  = np.array([[feats[f] for f in FEATURES]])
        x_sc   = scaler.transform(x_raw)
        proba  = float(shap_model.predict_proba(x_sc)[0, 1])
        sv     = explainer.shap_values(x_sc)[0]
        topic_label, _ = infer_topic(draft)
        tags   = recommend_tags(draft)
        recs   = get_recommendations(feats, sv)
        st.session_state['results'] = dict(
            proba=proba, sv=sv, x_sc=x_sc, feats=feats,
            topic_label=topic_label, tags=tags, recs=recs, draft=draft
        )
        if 'rewrite' in st.session_state:
            del st.session_state['rewrite']

if 'results' in st.session_state:
    r      = st.session_state['results']
    proba  = r['proba']
    sv     = r['sv']
    x_sc   = r['x_sc']
    feats  = r['feats']
    topic_label = r['topic_label']
    tags   = r['tags']
    recs   = r['recs']
    draft  = r['draft']
    conf   = max(proba, 1 - proba) * 100

if proba >= 0.65:
    verdict, color, dot = "This story is likely to be a hit.", "#1A7340", "🟢"
elif proba >= 0.42:
    verdict, color, dot = "This story has potential — a few small changes could lift it.", "#A05A00", "🟡"
else:
    verdict, color, dot = "This story needs some work before posting.", "#B0002A", "🔴"

st.markdown(f"""
<div class="verdict-card">
  <div style="font-size:2rem; margin-bottom:0.4rem;">{dot}</div>
  <div class="verdict-text" style="color:{color};">{verdict}</div>
  <div class="verdict-meta">Confidence {conf:.0f}%  ·  Based on 2,000 HONY posts  ·  Content features only</div>
</div>
""", unsafe_allow_html=True)

if recs:
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
        chips = "".join(f'<span class="tag-chip">#{r["tag"]}</span>' for r in tags)
        st.markdown(chips, unsafe_allow_html=True)
        st.markdown(
            '<div class="topic-perf" style="margin-top:0.5rem;">'
            'HONY used zero tags across 2,000 posts — adding tags could significantly boost discovery.</div>',
            unsafe_allow_html=True)
    else:
        st.markdown('<div class="topic-perf">No tag suggestions for this draft.</div>', unsafe_allow_html=True)

st.markdown('<hr class="hony-divider">', unsafe_allow_html=True)

st.markdown('<div class="section-label">AI-assisted rewrite</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="rec-detail" style="margin-bottom:0.9rem;">'
    'Apply these suggestions automatically. Review and edit before publishing — '
    'this is a starting point, not a final draft.</div>',
    unsafe_allow_html=True)

if st.button("Rewrite for better engagement", type="secondary"):
    rec_text = "\n".join([f"- {h}: {d}" for h, d in recs]) if recs else "No major issues found."
    prompt = f"""You are an editorial assistant for Humans of New York (HONY), the narrative photography blog by Brandon Stanton.

Here is a draft story caption:

{draft}

Our engagement model identified these specific issues:
{rec_text}

Rewrite this story applying the suggestions above. Keep Brandon Stanton's voice — first person, conversational, intimate, grounded in specific detail. Do not invent facts. Keep the same subject and story. Aim for under 400 words if the original is longer. If there are no direct quotes, add 1-2 that feel consistent with the story.

Return only the rewritten story. No preamble."""

    with st.spinner("Rewriting..."):
        try:
            import anthropic
            api_key = os.environ.get("ANTHROPIC_API_KEY") or st.secrets.get("ANTHROPIC_API_KEY", None)
            client = anthropic.Anthropic(api_key=api_key)
            message = client.messages.create(
                model="claude-opus-4-5",
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}]
            )
            rewrite = message.content[0].text
            c1, c2 = st.columns(2, gap="large")
            with c1:
                st.markdown('<div class="rewrite-label">Original</div>', unsafe_allow_html=True)
                st.text_area("", value=draft, height=300, disabled=True, key="orig_txt")
            with c2:
                st.markdown('<div class="rewrite-label">Suggested rewrite</div>', unsafe_allow_html=True)
                st.text_area("", value=rewrite, height=300, key="rewrite_txt")
            st.markdown(
                '<div class="disclaimer">Generated by Claude. Brandon should review and edit in his own voice before publishing.</div>',
                unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Rewrite unavailable: {e}")

with st.expander("See the data behind this prediction"):
    shap_exp = shap.Explanation(
        values=sv, base_values=float(explainer.expected_value),
        data=x_sc[0], feature_names=FEATURES)
    fig, _ = plt.subplots(figsize=(8, 4))
    shap.waterfall_plot(shap_exp, show=False)
    st.pyplot(fig, use_container_width=True)
    plt.close()
    st.markdown(
        '<div class="disclaimer">XGBoost trained on 12 content features. '
        'Temporal features excluded to avoid era bias. Cross-validated ROC AUC: 0.85. '
        'Does not model the photograph.</div>',
        unsafe_allow_html=True)

st.markdown('<hr class="hony-divider">', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center; font-size:0.68rem; color:#CCC; font-weight:300; letter-spacing:0.1em;">'
    'HONY STORY COACH  ·  GROUP 8  ·  CIS 434 SOCIAL MEDIA AND TEXT ANALYTICS  ·  SIMON BUSINESS SCHOOL  ·  SPRING 2026'
    '</div>', unsafe_allow_html=True)
