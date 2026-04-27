"""
Group08_Dashboard.py — HONY Story Coach
CIS 434 Group 8, Simon Business School, Spring 2026
Run: streamlit run Group08_Dashboard.py
"""
import os
import re
import gc
import warnings
warnings.filterwarnings("ignore")
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

gc.enable()

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
  .change-card { padding: .8rem 1.1rem; border-left: 3px solid #27AE60; margin-bottom: .6rem; background: #F6FFF6; border-radius: 0 4px 4px 0; font-size: .87rem; color: #222; font-weight: 300; line-height: 1.55; }
  .rewrite-label { font-size: .65rem; font-weight: 600; letter-spacing: .2em; text-transform: uppercase; color: #AAA; margin-bottom: .4rem; }
  .stTextArea textarea { font-family: 'Inter', sans-serif !important; font-size: .92rem !important; font-weight: 300 !important; line-height: 1.7 !important; border: 1px solid #E0E0E0 !important; border-radius: 4px !important; }
  .stButton > button { font-family: 'Inter', sans-serif !important; font-size: .8rem !important; font-weight: 500 !important; letter-spacing: .1em !important; text-transform: uppercase !important; border-radius: 3px !important; }
  .stButton > button[kind="primary"] { background: #1A1A1A !important; color: white !important; border: none !important; }
  .stButton > button[kind="secondary"] { background: white !important; color: #1A1A1A !important; border: 1px solid #CCC !important; }
  #MainMenu { visibility: hidden; } footer { visibility: hidden; } header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Plain-English SHAP feature labels ────────────────────────────────────────
SHAP_LABELS = {
    "word_count":          "Story length (words)",
    "sentence_count":      "Number of sentences",
    "avg_sentence_length": "Average sentence length",
    "vader_compound":      "Overall emotional tone",
    "vader_pos":           "Positive emotion",
    "vader_neg":           "Negative emotion",
    "vader_neu":           "Neutral / flat tone",
    "question_count":      "Questions in the story",
    "exclaim_count":       "Exclamations",
    "dialogue_ratio":      "Direct quotes from subject",
    "first_person_ratio":  "First-person voice (I / my)",
    "uppercase_ratio":     "Capitalisation",
}

# ── Intuitive topic engagement descriptions ───────────────────────────────────
TOPIC_ENG = {
    "Personal Struggle & Night Stories": "around 94 likes and reblogs per day on average  ·  highest engagement",
    "Family & Parenthood":               "around 85 likes and reblogs per day on average",
    "School & Childhood Memories":       "around 80 likes and reblogs per day on average",
    "Philosophical Reflection":          "around 76 likes and reblogs per day on average",
    "Fundraising & Advocacy":            "around 74 likes and reblogs per day on average",
    "International Field Stories":       "around 66 likes and reblogs per day on average",
    "NYC Life & Work":                   "around 46 likes and reblogs per day on average",
    "Iran / Shahnameh Series":           "around 23 likes and reblogs per day on average  ·  lowest engagement",
}

# ── Artifact loading ──────────────────────────────────────────────────────────
ARTIFACT_DIR = "."

@st.cache_resource
def load_artifacts():
    try:
        # Single file does everything — RF trained on 12 engineered features
        # prediction + SHAP explanations both come from this one model
        rf_explainer = joblib.load(f"{ARTIFACT_DIR}/rf_explainer.pkl")
        scaler       = joblib.load(f"{ARTIFACT_DIR}/scaler.pkl")
        tv_rec       = joblib.load(f"{ARTIFACT_DIR}/tfidf_rec.pkl")
        lda          = joblib.load(f"{ARTIFACT_DIR}/lda_model.pkl")
        lda_vec      = joblib.load(f"{ARTIFACT_DIR}/lda_vectorizer.pkl")
        meta         = joblib.load(f"{ARTIFACT_DIR}/metadata.pkl")
        bench        = pd.read_csv(f"{ARTIFACT_DIR}/bench_filtered.csv")
        bench["tags_parsed"] = bench["tags"].apply(
            lambda r: [t.lower().strip() for t in str(r).split("|") if t.strip()]
            if pd.notna(r) else [])
        bench_matrix = tv_rec.transform(bench["text"].tolist())
        gc.collect()
        return dict(rf_explainer=rf_explainer, scaler=scaler,
                    tv_rec=tv_rec, lda=lda, lda_vec=lda_vec,
                    meta=meta, bench=bench, bench_matrix=bench_matrix, ok=True)
    except Exception as e:
        return {"ok": False, "error": str(e)}

arts = load_artifacts()
if not arts["ok"]:
    st.error(f"Could not load model artifacts.\n\n{arts['error']}")
    st.stop()

rf_explainer = arts["rf_explainer"]
rf_model     = rf_explainer.model          # the RF itself lives inside the explainer
scaler       = arts["scaler"]
tv_rec       = arts["tv_rec"]
lda          = arts["lda"]
lda_vec      = arts["lda_vec"]
meta         = arts["meta"]
bench        = arts["bench"]
bench_matrix = arts["bench_matrix"]
FEATURES     = meta["features"]
TOPIC_LABELS = meta["topic_labels"]
FEATURE_LABELS = [SHAP_LABELS.get(f, f) for f in FEATURES]
analyzer     = SentimentIntensityAnalyzer()

# ── Feature extraction ────────────────────────────────────────────────────────
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

# ── Always-3 suggestions ──────────────────────────────────────────────────────
def get_recommendations(feats, shap_vals):
    shap_map = dict(zip(FEATURES, shap_vals))
    wc  = feats["word_count"]
    dr  = feats["dialogue_ratio"]
    fpr = feats["first_person_ratio"]
    vc  = feats["vader_compound"]
    qm  = feats["question_count"]
    asl = feats["avg_sentence_length"]

    candidates = []

    if dr < 0.01:
        candidates.append((shap_map.get("dialogue_ratio", -0.5),
            "💬 Add direct quotes from the subject",
            "This post has almost no direct quotation. HONY's most-engaged stories let the subject speak in their own words — pull at least 2 of their actual lines into the story."))
    elif dr < 0.04:
        candidates.append((shap_map.get("dialogue_ratio", -0.3),
            "💬 Let them speak a little more",
            "There's some dialogue here but more would help. Posts with strong first-person quotes from the subject earn significantly higher engagement. Add one more direct line."))

    if qm == 0:
        candidates.append((shap_map.get("question_count", -0.4),
            "🤔 End on an open question",
            "This post closes with a statement. HONY stories that end on a reflective question drive more reblogs — they leave the reader still thinking. Rewrite the last sentence as a question."))

    if wc > 500:
        candidates.append((shap_map.get("word_count", -0.3),
            "📏 Tighten to under 400 words",
            f"At {wc} words this is longer than the HONY sweet spot. Posts in the 200–400 word range earn the highest engagement. Cut any sentences that repeat an idea already made."))
    elif wc > 400:
        candidates.append((shap_map.get("word_count", -0.2),
            "📏 Trim one paragraph",
            f"At {wc} words you're slightly above the sweet spot of 200–400 words. One cut paragraph could meaningfully lift this story's performance."))
    elif wc < 100:
        candidates.append((shap_map.get("word_count", -0.2),
            "📝 Expand the narrative",
            f"At {wc} words this is very short. HONY's highest-performing posts give the subject room to breathe — aim for at least 150–200 words."))

    if asl > 22:
        candidates.append((shap_map.get("avg_sentence_length", -0.25),
            "✍️ Break up the long sentences",
            f"Average sentence length is {asl:.0f} words. HONY's most-read stories mix long emotional sentences with very short ones. A single short sentence at a peak moment can be extremely powerful."))

    if abs(vc) < 0.3:
        candidates.append((shap_map.get("vader_compound", -0.2),
            "❤️ Find the emotional core",
            "This draft reads as emotionally measured. HONY's most-shared stories have a clear emotional undercurrent — vulnerability, grief, unexpected joy. Find the moment of highest feeling and lean into it."))

    if fpr < 0.03:
        candidates.append((shap_map.get("first_person_ratio", -0.2),
            "🎤 More of their voice, less narration",
            "The post reads more as narration about the subject than as their own voice. Shift more of the story into their perspective — HONY's style puts 'I' at the centre."))

    fallbacks = [
        (-0.05, "📸 Ground it in one specific detail",
         "The strongest HONY posts anchor an emotional truth in a single concrete detail — a job, a street corner, an object, a name. If any moment feels abstract, find the one specific thing that made this person unforgettable."),
        (-0.04, "🔤 Start with the most surprising line",
         "HONY's best posts open on the thing that made Brandon stop walking. If the most arresting moment is buried in the middle, move it to the first sentence."),
        (-0.03, "🧍 Name the person and the place",
         "Giving the subject a first name and a specific location — even just 'outside Penn Station' — makes the story feel grounded and real rather than universal."),
    ]

    candidates.sort(key=lambda x: x[0])
    result = [(h, d) for _, h, d in candidates]
    fb_idx = 0
    while len(result) < 3 and fb_idx < len(fallbacks):
        _, h, d = fallbacks[fb_idx]
        result.append((h, d))
        fb_idx += 1
    return result[:3]

# ── Tag recommender ───────────────────────────────────────────────────────────
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

# ── Topic inference ───────────────────────────────────────────────────────────
def infer_topic(text):
    clean = re.sub(r"[^a-zA-Z\s]", " ", text.lower())
    dist  = lda.transform(lda_vec.transform([clean]))[0]
    top   = int(dist.argmax())
    return TOPIC_LABELS.get(top, f"Topic {top}"), float(dist[top])

# ── API key ───────────────────────────────────────────────────────────────────
def get_api_key():
    key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not key:
        try:
            key = st.secrets.get("ANTHROPIC_API_KEY", "")
        except Exception:
            key = ""
    return key

# ── Rewrite prompt ────────────────────────────────────────────────────────────
def build_rewrite_prompt(draft, feats, recs):
    wc        = feats["word_count"]
    target_wc = min(wc, 380) if wc > 380 else wc
    rec_lines = "\n".join([f"  - {h}: {d}" for h, d in recs])
    return f"""You are an editorial assistant for Humans of New York (HONY), the narrative photography blog by Brandon Stanton. HONY posts are first-person, intimate, grounded in specific detail, and always centre the subject's own voice.

ORIGINAL STORY ({wc} words):
{draft}

ISSUES IDENTIFIED (address ALL of these):
{rec_lines}

YOUR TASK:
Rewrite the story so it addresses every issue. Make real structural changes — not cosmetic rearrangement.
- If dialogue is missing, add 2–3 direct quotes from the subject in quotation marks, written in their authentic voice
- If the ending is a statement, rewrite the final sentence as a reflective question
- If word count is over 380, cut to under {target_wc} words by removing repetition
- Keep Brandon Stanton's voice: short paragraphs, intimate, grounded in specific people and places

YOUR RESPONSE MUST CONTAIN EXACTLY THESE TWO SECTIONS:

REWRITTEN STORY:
[the full rewritten story here]

WHAT I CHANGED:
[exactly 3 bullet points starting with a dash explaining each structural change and why it helps]"""

# ── Parse rewrite response ────────────────────────────────────────────────────
def parse_response(text):
    if "WHAT I CHANGED:" in text:
        parts   = text.split("WHAT I CHANGED:", 1)
        rewrite = parts[0].replace("REWRITTEN STORY:", "").strip()
        raw     = parts[1].strip()
    elif "What I Changed:" in text:
        parts   = text.split("What I Changed:", 1)
        rewrite = parts[0].replace("REWRITTEN STORY:", "").replace("Rewritten Story:", "").strip()
        raw     = parts[1].strip()
    else:
        rewrite = text.strip()
        raw     = "- Story rewritten to address editorial suggestions above"
    changes = [l.strip().lstrip("-•*123456789.)").strip()
               for l in raw.split("\n") if len(l.strip()) > 10]
    return rewrite, changes[:5]

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hony-header">
  <div class="hony-eyebrow">Humans of New York</div>
  <div class="hony-title">Story Coach</div>
  <div class="hony-subtitle">Paste a draft caption to see how it might perform on Tumblr — and how to make it stronger.</div>
</div>
""", unsafe_allow_html=True)

draft_input = st.text_area(
    "",
    height=260,
    placeholder='"I hadn\'t talked to my mother in three years. Not because I was angry. I just didn\'t know how to begin again..."',
    label_visibility="collapsed",
)

col_btn, _ = st.columns([2, 1])
with col_btn:
    analyse = st.button("Analyse this story", type="primary", use_container_width=True)

# ── Run analysis ──────────────────────────────────────────────────────────────
if analyse:
    if len(draft_input.strip()) < 30:
        st.warning("Please paste a full story draft — at least a few sentences.")
    else:
        feats  = extract_features(draft_input)
        x_raw  = np.array([[feats[f] for f in FEATURES]])
        x_sc   = scaler.transform(x_raw)

        # Prediction — RF on 12 engineered features (same model as SHAP)
        proba  = float(rf_model.predict_proba(x_sc)[0, 1])

        # SHAP values — RF returns list [class0, class1], take class 1
        sv_raw = rf_explainer.shap_values(x_sc)
        if isinstance(sv_raw, list):
            sv = sv_raw[1][0]
        elif hasattr(sv_raw, 'ndim') and sv_raw.ndim == 3:
            sv = sv_raw[0, :, 1]
        else:
            sv = sv_raw[0]

        topic_label, _ = infer_topic(draft_input)
        tags  = recommend_tags(draft_input)
        recs  = get_recommendations(feats, sv)

        st.session_state["results"] = dict(
            proba=proba, sv=sv, x_sc=x_sc, feats=feats,
            topic_label=topic_label, tags=tags, recs=recs, draft=draft_input
        )
        for k in ["rewrite", "changes"]:
            if k in st.session_state:
                del st.session_state[k]

# ── Render results ────────────────────────────────────────────────────────────
if "results" in st.session_state:
    R           = st.session_state["results"]
    proba       = R["proba"]
    sv          = R["sv"]
    x_sc        = R["x_sc"]
    feats       = R["feats"]
    topic_label = R["topic_label"]
    tags        = R["tags"]
    recs        = R["recs"]
    draft       = R["draft"]
    conf        = max(proba, 1 - proba) * 100

    if proba >= 0.65:
        verdict, color, dot = "This story is likely to be a hit.", "#1A7340", "🟢"
    elif proba >= 0.42:
        verdict, color, dot = "This story has potential — a few changes could lift it.", "#A05A00", "🟡"
    else:
        verdict, color, dot = "This story needs some work before posting.", "#B0002A", "🔴"

    st.markdown(f"""
    <div class="verdict-card">
      <div style="font-size:2rem; margin-bottom:.4rem;">{dot}</div>
      <div class="verdict-text" style="color:{color};">{verdict}</div>
      <div class="verdict-meta">Confidence {conf:.0f}%  ·  Based on 2,000 HONY posts  ·  Content features only</div>
    </div>
    """, unsafe_allow_html=True)

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
            st.markdown(f'<div class="topic-perf">{eng}</div>', unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-label">Suggested Tumblr tags</div>', unsafe_allow_html=True)
        if tags:
            chips = "".join(f'<span class="tag-chip">#{t["tag"]}</span>' for t in tags)
            st.markdown(chips, unsafe_allow_html=True)
            st.markdown(
                '<div class="topic-perf" style="margin-top:.5rem;">HONY used zero tags across 2,000 posts — '
                'adding tags could significantly boost discovery.</div>',
                unsafe_allow_html=True)
        else:
            st.markdown('<div class="topic-perf">No tag suggestions for this draft.</div>', unsafe_allow_html=True)

    st.markdown('<hr class="hony-divider">', unsafe_allow_html=True)

    st.markdown('<div class="section-label">AI-assisted rewrite</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="rec-detail" style="margin-bottom:.9rem;">'
        'Apply all three suggestions automatically. The rewrite will explain exactly what was '
        'changed and why — review and edit in your own voice before publishing.</div>',
        unsafe_allow_html=True)

    if st.button("✦ Rewrite for better engagement", type="secondary"):
        prompt = build_rewrite_prompt(draft, feats, recs)
        with st.spinner("Rewriting..."):
            try:
                import anthropic as _ant
                api_key = get_api_key()
                if not api_key:
                    st.error("No Anthropic API key found. Add ANTHROPIC_API_KEY to Streamlit secrets.")
                else:
                    client  = _ant.Anthropic(api_key=api_key)
                    message = client.messages.create(
                        model="claude-opus-4-5",
                        max_tokens=2048,
                        messages=[{"role": "user", "content": prompt}]
                    )
                    rewrite, changes = parse_response(message.content[0].text)
                    st.session_state["rewrite"] = rewrite
                    st.session_state["changes"] = changes
            except ImportError:
                st.error("anthropic package not installed. Add `anthropic` to requirements.txt.")
            except Exception as e:
                st.error(f"Rewrite unavailable: {e}")

    if "rewrite" in st.session_state:
        c1, c2 = st.columns(2, gap="large")
        with c1:
            st.markdown('<div class="rewrite-label">Original</div>', unsafe_allow_html=True)
            st.text_area("", value=draft, height=340, disabled=True, key="orig_txt")
            st.caption(f"{feats['word_count']} words")
        with c2:
            st.markdown('<div class="rewrite-label">Suggested rewrite</div>', unsafe_allow_html=True)
            st.text_area("", value=st.session_state["rewrite"], height=340, key="rewrite_txt")
            st.caption(f"{len(st.session_state['rewrite'].split())} words")

        changes = st.session_state.get("changes", [])
        st.markdown('<div class="section-label" style="margin-top:1.4rem;">What was changed</div>', unsafe_allow_html=True)
        if changes:
            for item in changes:
                st.markdown(f'<div class="change-card">✓ {item}</div>', unsafe_allow_html=True)
        else:
            for headline, _ in recs:
                clean_h = headline.split(" ", 1)[1] if len(headline) > 2 else headline
                st.markdown(f'<div class="change-card">✓ {clean_h} — applied above</div>', unsafe_allow_html=True)

        st.markdown(
            '<div class="disclaimer">Generated by Claude. '
            'Brandon should review and edit in his own voice before publishing.</div>',
            unsafe_allow_html=True)

    # SHAP expander — plain-English labels
    with st.expander("See the data behind this prediction"):
        try:
            base_val = (float(rf_explainer.expected_value[1])
                        if hasattr(rf_explainer.expected_value, '__len__')
                        else float(rf_explainer.expected_value))

            shap_exp = shap.Explanation(
                values=sv,
                base_values=base_val,
                data=x_sc[0],
                feature_names=FEATURE_LABELS
            )
            fig, _ = plt.subplots(figsize=(8, 4))
            shap.waterfall_plot(shap_exp, show=False)
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            plt.close("all")
            gc.collect()
        except Exception as e:
            st.warning(f"Chart unavailable: {e}")

        st.markdown(
            '<div class="disclaimer">'
            'Prediction powered by a Random Forest model trained on 12 writing style features. '
            'Bars pointing right helped your score — bars pointing left hurt it. '
            'Temporal features (year, month) were excluded to avoid era bias. '
            'Cross-validated accuracy: 0.85 AUC. Does not analyse the photograph.</div>',
            unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown('<hr class="hony-divider">', unsafe_allow_html=True)
st.markdown(
    '<div style="text-align:center;font-size:.68rem;color:#CCC;font-weight:300;letter-spacing:.1em;">'
    'HONY STORY COACH  ·  GROUP 8  ·  CIS 434 SOCIAL MEDIA AND TEXT ANALYTICS  ·  SIMON BUSINESS SCHOOL  ·  SPRING 2026'
    '</div>', unsafe_allow_html=True)
