# streamlit_app.py (UPGRADED)
import os, sys, csv, pickle, math, json
import streamlit as st
from datetime import datetime
import numpy as np
import pandas as pd
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import URLError, HTTPError

# ML & text
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from dotenv import load_dotenv

from utils import remove_special_chars, tokenize_training_style, type_token_ratio, avg_word_length, avg_sentence_length, punctuation_density, sensational_word_count, extract_dates

# ================= PATHS & CONFIG =================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "final_dataset.csv")
MODEL_PATH = os.path.join(BASE_DIR, "model", "my_model.h5")
TOKENIZER_PATH = os.path.join(BASE_DIR, "model", "tokenizer.pkl")
LOG_DIR = os.path.join(BASE_DIR, "logs")
FEEDBACK_FILE = os.path.join(LOG_DIR, "feedback.csv")
KEYWORD_STATS_PATH = os.path.join(LOG_DIR, "keyword_stats.pkl")
TFIDF_STORE_PATH = os.path.join(LOG_DIR, "tfidf_store.npz")
TFIDF_META_PATH = os.path.join(LOG_DIR, "tfidf_meta.pkl")

MAX_LEN = 1000
SENSATIONAL_WORDS = set([
    # a small seed list - expanded automatically from dataset
    "shocking","breaking","exclusive","horrifying","unbelievable","urgent",
    "massive","miracle","secret","exposed","shocker","viral","alert"
])

os.makedirs(LOG_DIR, exist_ok=True)

# ========= Load model + tokenizer =========
@st.cache_resource
def load_model_tokenizer():
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(TOKENIZER_PATH, "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_model_tokenizer()

# ========= Load dataset =========
@st.cache_data
def load_dataset(path=DATA_PATH, sample_frac=None):
    df = pd.read_csv(path)
    if sample_frac:
        df = df.sample(frac=sample_frac, random_state=42).reset_index(drop=True)
    return df

data_df = load_dataset()

# ========= Build suspicious keyword scores =========
def build_keyword_stats(df):
    from collections import Counter, defaultdict
    fake_mask = df['label'] == 0 or df.get('target') is not None and df['target'].eq('FAKE') if False else (df['label']==0 if 'label' in df.columns else (df['target']=='FAKE'))
    if 'label' in df.columns:
        fake_mask = df['label'] == 0  # in your dataset 0 or 1? check — you can flip if needed
    elif 'target' in df.columns:
        fake_mask = df['target'].str.upper() == "FAKE"
    else:
        fake_mask = pd.Series([False]*len(df))

    fake_cnt = Counter()
    real_cnt = Counter()
    total_cnt = Counter()
    for _, row in df.iterrows():
        text = row.get('text','')
        tokens = tokenize_training_style(text)
        total_cnt.update(tokens)
        if fake_mask.iloc[_] if isinstance(fake_mask, pd.Series) else False:
            fake_cnt.update(tokens)
        else:
            real_cnt.update(tokens)

    words = list(total_cnt.keys())
    stats = {}
    for w in words:
        f = fake_cnt[w]
        r = real_cnt[w]
        total = f + r
        if total == 0: continue
        fake_score = f / total  # 0..1 (1 => only fake)
        stats[w] = {"fake_count": f, "real_count": r, "total": total, "fake_score": fake_score}
    return stats

@st.cache_data
def get_keyword_stats():
    # load cached stats if exist
    if os.path.exists(KEYWORD_STATS_PATH):
        try:
            with open(KEYWORD_STATS_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    # else compute
    stats = build_keyword_stats(data_df)
    with open(KEYWORD_STATS_PATH, "wb") as f:
        pickle.dump(stats, f)
    return stats

keyword_stats = get_keyword_stats()

# build sensational set from top fake-associated words
def build_sensational_set(keyword_stats, threshold=0.8, topk=200):
    items = sorted(keyword_stats.items(), key=lambda kv: (-kv[1]['fake_score'], -kv[1]['total']))
    selected = [w for w, _ in items[:topk] if _['fake_score'] >= threshold]
    # fallback: add seed words:
    sel = set([w for w,_ in items[:topk] if _['fake_score']>=threshold])
    return sel.union(SENSATIONAL_WORDS)

SENSATIONAL_SET = build_sensational_set(keyword_stats)

# ========= Build TF-IDF index for semantic search =========
@st.cache_data
def build_tfidf_index(df):
    texts = df['text'].astype(str).tolist()
    # Use cleaned strings for TF-IDF
    cleaned_texts = [" ".join(tokenize_training_style(t)) for t in texts]
    vec = TfidfVectorizer(max_features=50000, ngram_range=(1,2))
    tfidf = vec.fit_transform(cleaned_texts)
    meta = {"docs": texts, "vectorizer": vec}
    # cache to disk
    with open(TFIDF_META_PATH, "wb") as f:
        pickle.dump(meta, f)
    # store matrix using scipy sparse save
    from scipy import sparse
    sparse.save_npz(TFIDF_STORE_PATH, tfidf)
    return tfidf, vec, texts

def load_tfidf_index():
    try:
        from scipy import sparse
        if os.path.exists(TFIDF_STORE_PATH) and os.path.exists(TFIDF_META_PATH):
            tfidf = sparse.load_npz(TFIDF_STORE_PATH)
            with open(TFIDF_META_PATH, "rb") as f:
                meta = pickle.load(f)
            return tfidf, meta['vectorizer'], meta['docs']
    except Exception:
        pass
    return build_tfidf_index(data_df)

tfidf_matrix, tfidf_vectorizer, docs = load_tfidf_index()

# ========= Prediction & helpers =========
def preprocess_for_model(text):
    tokens = tokenize_training_style(text)
    seq = tokenizer.texts_to_sequences([tokens])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')
    return tokens, seq, padded

def predict_label(text):
    tokens, seq, padded = preprocess_for_model(text)
    pred = float(model.predict(padded, verbose=0)[0][0])
    label = "Fake" if pred < 0.5 else "Real"
    return label, pred, tokens, seq, int((padded != 0).sum())

# semantic search
def get_similar_articles(text, topn=5):
    cleaned = " ".join(tokenize_training_style(text))
    qv = tfidf_vectorizer.transform([cleaned])
    sims = cosine_similarity(qv, tfidf_matrix).flatten()
    idx = np.argsort(-sims)[:topn]
    return [(int(i), float(sims[i]), docs[i]) for i in idx]

# suspicious highlighting score (word-level)
def word_suspiciousness(tokens, stats=keyword_stats):
    out = []
    for t in tokens:
        info = stats.get(t, None)
        score = info['fake_score'] if info else 0.0
        out.append((t, score))
    return out

# readability and linguistic features
def compute_readability_features(text, tokens):
    return {
        "avg_word_length": round(avg_word_length(tokens), 3),
        "avg_sentence_length": round(avg_sentence_length(text), 3),
        "type_token_ratio": round(type_token_ratio(tokens), 3),
        "punctuation_density": round(punctuation_density(text), 4),
        "sensational_word_count": sensational_word_count(tokens, SENSATIONAL_SET)
    }

# timeline check
def timeline_checks(text):
    extracted = extract_dates(text)
    now = datetime.now()
    issues = []
    for d in extracted:
        # very simple checks: future-date words or relative terms
        if "next" in d or "tomorrow" in d:
            issues.append(f"Date/word '{d}' mentions future — check plausibility")
    return extracted, issues

# ========== Feedback logging ==========
def log_feedback(news, predicted, user_feedback, correct_label):
    header = ["timestamp", "news_text", "predicted_label", "user_feedback", "correct_label"]
    if not os.path.exists(FEEDBACK_FILE):
        with open(FEEDBACK_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(header)
    with open(FEEDBACK_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([datetime.now(), news, predicted, user_feedback, correct_label])

# ========== Gemini (optional for expanded explanations) ==========
APP_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(APP_DIR, "api_key.env"), override=True)
load_dotenv(os.path.join(BASE_DIR, "api_key.env"), override=True)
YOU_API_KEY = (
    os.getenv("YOU_API_KEY")
    or os.getenv("YOUCOM_API_KEY")
    or os.getenv("YDC_INDEX_API_KEY")
)
YOU_SEARCH_AVAILABLE = bool(YOU_API_KEY)


@st.cache_data(ttl=300)
def search_with_you(query, count=5):
    if not YOU_SEARCH_AVAILABLE or not query.strip():
        return []

    params = urlencode({"query": query, "count": count})
    request = Request(
        f"https://ydc-index.io/v1/search?{params}",
        headers={"X-API-Key": YOU_API_KEY},
        method="GET",
    )

    try:
        with urlopen(request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError):
        return []

    results = []
    for source_name in ("web", "news"):
        for item in payload.get("results", {}).get(source_name, []) or []:
            results.append(
                {
                    "source": source_name,
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "description": item.get("description", ""),
                    "snippets": item.get("snippets", []) or [],
                    "page_age": item.get("page_age", ""),
                }
            )

    return results


def format_search_evidence(search_results, max_items=5):
    lines = []
    for item in search_results[:max_items]:
        snippet = item["description"]
        if item["snippets"]:
            snippet = item["snippets"][0]
        if item["title"] or snippet:
            lines.append(f"- {item['title']}: {snippet}")
    return "\n".join(lines)


def render_you_results(search_results, title="You.com results"):
    if not search_results:
        st.info("You.com returned no results.")
        return

    st.markdown(f"### {title}")
    rows = []
    for item in search_results[:8]:
        rows.append(
            {
                "source": item["source"],
                "title": item["title"],
                "snippet": (item["snippets"][0] if item["snippets"] else item["description"]),
                "url": item["url"],
                "page_age": item["page_age"],
            }
        )
    df_rows = pd.DataFrame(rows)
    # tidy up long snippets and URLs for a cleaner layout
    if "snippet" in df_rows.columns:
        df_rows["snippet"] = df_rows["snippet"].astype(str).str.replace("\n", " ").str.slice(0, 400)
    if "url" in df_rows.columns:
        df_rows["url"] = df_rows["url"].astype(str)

    st.dataframe(df_rows, use_container_width=True, height=560)

# ========== STREAMLIT UI ==========
st.set_page_config(page_title="VeriFy", layout="wide")
st.title("📰VeriFy - Smart Fake News Detection")

tabs = st.tabs(["Analyze", "Similar Articles", "Explainability", "Text Analysis", "Feedback", "About"])

with tabs[0]:
    st.header("Analyze an article")
    user_text = st.text_area("Paste article or headline here:", height=220)

    if st.button("Analyze"):
        if not user_text.strip():
            st.warning("Please enter some text.")
        else:
            label, pred, tokens, seq, nonzero = predict_label(user_text)
            # top suspicious words
            susp = word_suspiciousness(tokens)
            top_susp = sorted(susp, key=lambda x: -x[1])[:10]

            # show prediction
            col1, col2 = st.columns([2,1])
            with col1:
                if label == "Fake":
                    st.error(f"Prediction: **{label}**")
                else:
                    st.success(f"Prediction: **{label}**")
                st.write(f"Model raw score (sigmoid output): {pred:.6f}")
                st.write(f"Non-zero tokens in padded input: {nonzero}")

                # highlight text with colors depending on score
                def colored_text(tokens_scores):
                    parts = []
                    for tok, score in tokens_scores:
                        if score >= 0.8:
                            parts.append(f"<span style='background:#ff9999;padding:2px;border-radius:3px'>{tok}</span>")
                        elif score >= 0.5:
                            parts.append(f"<span style='background:#ffd699;padding:2px;border-radius:3px'>{tok}</span>")
                        else:
                            parts.append(tok)
                    return " ".join(parts)

                st.markdown("### Suspicious keyword heatmap")
                st.markdown(colored_text(susp[:200]), unsafe_allow_html=True)

                # show top suspicious list
                if top_susp:
                    st.write("Top suspicious tokens (fake-score, counts):")
                    top_table = [(t, round(keyword_stats.get(t,{}).get("fake_score",0),3), keyword_stats.get(t,{}).get("total",0)) for t,_ in top_susp]
                    st.table(pd.DataFrame(top_table, columns=["token","fake_score","total_count"]).head(10))

                search_results = search_with_you(user_text, count=5)
                news_results = [r for r in (search_results or []) if r.get("source") == "news"]
                if label == "Fake":
                    st.info("Model predicted Fake")
                else:
                    if news_results:
                        render_you_results(news_results, title="News evidence")
                    else:
                        st.info("You.com returned no news results; skipping external evidence.")

            with col2:
                st.markdown("### Quick stats")
                features = compute_readability_features(user_text, tokens)
                st.metric("Avg sentence length", features['avg_sentence_length'])
                st.metric("Type-token ratio", features['type_token_ratio'])
                st.metric("Sensational words", features['sensational_word_count'])
                st.metric("Avg word length", features['avg_word_length'])
                st.metric("Punctuation density", features['punctuation_density'])

            # timeline checks
            extracted_dates, issues = timeline_checks(user_text)
            if extracted_dates:
                st.markdown("### Extracted dates/days")
                st.write(extracted_dates)
                if issues:
                    st.warning("Timeline issues: " + "; ".join(issues))

with tabs[1]:
    st.header("Similar articles (semantic search)")
    q = st.text_area("Enter text to find similar articles (or use above)", height=120)
    if st.button("Find similar"):
        if not q.strip():
            st.warning("Please enter query text.")
        else:
            you_results = search_with_you(q, count=8)
            if you_results:
                rows = []
                for item in you_results:
                    rows.append(
                        {
                            "source": item["source"],
                            "title": item["title"],
                            "snippet": (item["snippets"][0] if item["snippets"] else item["description"]),
                            "url": item["url"],
                            "page_age": item["page_age"],
                        }
                    )
                df_rows = pd.DataFrame(rows)
                if "snippet" in df_rows.columns:
                    df_rows["snippet"] = df_rows["snippet"].astype(str).str.replace("\n", " ").str.slice(0, 400)
                st.dataframe(df_rows, use_container_width=True, height=560)
            else:
                st.info("You.com search is unavailable or returned no results, so showing local semantic matches instead.")
                sims = get_similar_articles(q, topn=8)
                rows = []
                for i, s, _ in sims:
                    rows.append({"idx": i, "score": round(s, 4), "text": docs[i][:400], "label": data_df.iloc[i].get("target", data_df.iloc[i].get("label", ""))})
                df_local = pd.DataFrame(rows)
                df_local["text"] = df_local["text"].astype(str).str.replace("\n", " ")
                st.dataframe(df_local, use_container_width=True, height=520)

with tabs[2]:
    st.header("Explainability & Counterfactuals")
    expl_query = st.text_area("Enter article for explainability:", height=160)
    if st.button("Explain now"):
        if not expl_query.strip():
            st.warning("Enter text first")
        else:
            label, pred, tokens, seq, nonzero = predict_label(expl_query)
            st.write("Prediction:", label, f"({pred:.6f})")
            st.write("Tokens (first 80):", tokens[:80])
            st.write("Sequence indices (first 80):", seq[0][:80])
            st.write("Non-zero token count:", nonzero)

            # local counterfactual hint: top suspicious tokens
            susp = word_suspiciousness(tokens)
            top = sorted(susp, key=lambda x: -x[1])[:6]
            if top:
                st.markdown("### Top suspicious tokens (fake score)")
                st.table(pd.DataFrame(top, columns=["token","fake_score"]))
                st.markdown("### Counterfactual hint")
                st.write("Try removing or neutralizing suspicious tokens above — if many high-score tokens are removed the model may flip label.")
            search_results = search_with_you(expl_query, count=5)
            news_results = [r for r in (search_results or []) if r.get("source") == "news"]
            if label == "Fake":
                st.info("Model predicted Fake — external news evidence suppressed.")
            else:
                if news_results:
                    render_you_results(news_results, title="You.com news evidence")
                else:
                    st.info("You.com returned no news results; skipping external evidence.")

with tabs[3]:
    st.header("Text Analysis (Readability & Linguistics)")
    analy_text = st.text_area("Paste text to analyze:", height=200)
    if st.button("Analyze text"):
        if not analy_text.strip():
            st.warning("Enter text")
        else:
            tokens = tokenize_training_style(analy_text)
            feats = compute_readability_features(analy_text, tokens)
            st.subheader("Readability / Linguistic Features")
            st.json(feats)
            st.markdown("### Frequency Wordcloud (top tokens)")
            from collections import Counter
            c = Counter(tokens)
            top = c.most_common(30)
            st.table(pd.DataFrame(top, columns=["token","count"]))

with tabs[4]:
    st.header("Feedback collected")
    st.write("You can see all feedback saved in logs/feedback.csv")

    # === New feedback input ===
    st.subheader("Submit Feedback")
    fb_text = st.text_area("Paste the article text you analyzed:")
    fb_pred = st.selectbox("Model's predicted label", ["Fake", "Real"])
    fb_user_feedback = st.selectbox("Was the model correct?", ["Yes", "No"])
    fb_correct = st.selectbox("What was the correct label?", ["Fake", "Real"])
    if st.button("Submit Feedback"):
        if not fb_text.strip():
            st.warning("Please paste the text.")
        else:
            log_feedback(fb_text, fb_pred, fb_user_feedback, fb_correct)
            st.success("✅ Feedback logged successfully!")

    # === Show existing feedback ===
    if os.path.exists(FEEDBACK_FILE):
        fb = pd.read_csv(FEEDBACK_FILE)
        st.dataframe(fb.tail(200))
    else:
        st.info("No feedback yet.")

with tabs[5]:
    st.header("About / Notes")
    st.write("""
    Features:
    - Suspicious keyword highlighting computed from your training dataset.
    - You.com web search for evidence shown directly in the UI.
    - Semantic search (TF-IDF) over entire dataset as fallback.
    - Explainability panel: tokens, suspicious scores, local counterfactual hint, and You.com evidence.
    - Readability & linguistics.
    - Timeline checks for extracted dates.
    """)
    #st.write(" Developed by VeriFy Team. \n 1. Pratyush Anand(23103073)\n 2. Avinash Joshi(23103074)\n 3. Divij Verma(23103076)")
