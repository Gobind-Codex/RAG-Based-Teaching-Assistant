import streamlit as st
import numpy as np
import requests
import joblib
import json
import random
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="RAG Assistant", page_icon="🎬", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
    <style>
        .main { background-color: #0e1117; }
        .stTextInput > div > div > input {
            background-color: #1e2130;
            color: white;
            border: 1px solid #4a4e69;
            border-radius: 10px;
            padding: 12px;
            font-size: 16px;
        }
        .response-box {
            background-color: #1e2130;
            border-left: 4px solid #7c83fd;
            border-radius: 10px;
            padding: 20px;
            margin-top: 20px;
            color: #e0e0e0;
            font-size: 15px;
            line-height: 1.7;
        }
        .chunk-card {
            background-color: #16213e;
            border: 1px solid #4a4e69;
            border-radius: 8px;
            padding: 12px 16px;
            margin-bottom: 10px;
            color: #c9d1d9;
            font-size: 13px;
        }
        .chunk-title { color: #7c83fd; font-weight: bold; font-size: 14px; }
        .timestamp { color: #f0a500; font-size: 12px; }
        h1 { color: #7c83fd !important; }
        .stButton > button {
            background: linear-gradient(135deg, #7c83fd, #4a4e69);
            color: white;
            border: none;
            border-radius: 10px;
            padding: 10px 30px;
            font-size: 16px;
            width: 100%;
            transition: 0.3s;
        }
        .stButton > button:hover { opacity: 0.85; }
        [data-testid="stSidebar"] {
            background-color: #0e1117;
        }
        .history-item {
            background-color: #1e2130;
            border-radius: 8px;
            padding: 12px;
            margin-bottom: 8px;
            cursor: pointer;
            transition: 0.2s;
            color: #e0e0e0;
            font-size: 14px;
        }
        .history-item:hover { 
            background-color: #262b3d;
        }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_embeddings():
    df = joblib.load("embeddings.joblib")
    if df.empty:
        raise ValueError("Loaded embeddings data is empty")
    if "embedding" not in df.columns:
        raise ValueError("Loaded embeddings data is missing the 'embedding' column")
    return df


def create_embedding(text):
    r = requests.post("http://localhost:11434/api/embed", json={"model": "bge-m3", "input": [text]}, timeout=30)
    r.raise_for_status()
    return r.json()["embeddings"][0]


def inference(prompt):
    r = requests.post("http://localhost:11434/api/generate", json={"model": "llama3.2", "prompt": prompt, "stream": False}, timeout=120)
    r.raise_for_status()
    return r.json()["response"]


def format_time(seconds):
    return f"{int(seconds // 60):02d}:{int(seconds % 60):02d}"


# ── UI ──────────────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "queries" not in st.session_state:
    st.session_state.queries = []

# Random search examples
examples = [
    "How does backpropagation work?",
    "What is gradient descent?",
    "Explain neural network layers",
    "What are activation functions?",
    "How do convolutional layers work?",
    "What is overfitting?",
    "Explain dropout regularization",
    "What is batch normalization?",
    "How does transfer learning work?",
    "What are recurrent neural networks?",
    "Explain attention mechanism",
    "What is the vanishing gradient problem?",
    "How do optimizers like Adam work?",
    "What is the difference between CNN and RNN?",
    "Explain loss functions"
]
if "current_example" not in st.session_state:
    st.session_state.current_example = random.choice(examples)

# Sidebar for history
with st.sidebar:
    st.markdown("### 💬 Chat History")
    
    if st.button("➕ New Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.queries = []
        st.rerun()
    
    st.markdown("---")
    
    if st.session_state.queries:
        for i, q in enumerate(reversed(st.session_state.queries)):
            truncated = q[:40] + "..." if len(q) > 40 else q
            st.markdown(f'<div class="history-item">💭 {truncated}</div>', unsafe_allow_html=True)
    else:
        st.info("No chat history yet")

st.markdown("<h1 style='text-align: center;'>🎬 RAG Teaching Assistant</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; color: #888;'>Ask anything about the video playlist — get timestamped answers.</p>", unsafe_allow_html=True)

try:
    df = load_embeddings()
except Exception as e:
    st.error(f"Failed to load embeddings: {e}")
    st.stop()

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "sources" in msg:
            with st.expander("📂 Sources"):
                for src in msg["sources"]:
                    st.markdown(f"""
                    <div class="chunk-card">
                        <div class="chunk-title">📹 {src['title']} — Episode {src['number']}</div>
                        <div class="timestamp">⏱ {src['start']} → {src['end']}</div>
                        <div style="margin-top:6px">{src['text']}</div>
                    </div>""", unsafe_allow_html=True)

query = st.chat_input(f"e.g. {st.session_state.current_example}")

if query:
    st.session_state.current_example = random.choice(examples)  # Change example for next time

if query:
    st.session_state.queries.append(query)
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching..."):
            try:
                q_emb = create_embedding(query)
                embedding_matrix = np.vstack(df["embedding"].to_list())
                question_vector = np.asarray(q_emb, dtype=float).reshape(1, -1)

                if embedding_matrix.size == 0:
                    raise ValueError("Embedding matrix is empty")

                sims = cosine_similarity(embedding_matrix, question_vector).flatten()
                top_idx = sims.argsort()[::-1][:5]
                top_df = df.iloc[top_idx].copy()
                top_df["start"] = top_df["start"].apply(format_time)
                top_df["end"] = top_df["end"].apply(format_time)
            except Exception as e:
                st.error(f"Embedding/search error: {e}")
                st.stop()

        with st.spinner("✨ Generating answer..."):
            prompt = f"""You are an assistant that answers using only the provided video chunks.
{top_df[["title", "number", "start", "end", "text"]].to_json(orient="records")}
"{query}"

Task:
- Identify relevant videos and timestamps.
- Explain whether the topic is briefly mentioned or well explained.
- Guide the user where to watch.

Rules:
- Do NOT make up information.
- Use only given timestamps.
- Be clear and direct.
- Keep answer concise.

If no relevant information is found:
→ "The playlist does not contain enough information about this topic."""

            try:
                answer = inference(prompt)
            except Exception as e:
                st.error(f"Inference error: {e}")
                st.stop()

        st.markdown(answer)
        
        sources = [{"title": row["title"], "number": row["number"], "start": row["start"], "end": row["end"], "text": row["text"]} for _, row in top_df.iterrows()]
        
        with st.expander("📂 Sources"):
            for src in sources:
                st.markdown(f"""
                <div class="chunk-card">
                    <div class="chunk-title">📹 {src['title']} — Episode {src['number']}</div>
                    <div class="timestamp">⏱ {src['start']} → {src['end']}</div>
                    <div style="margin-top:6px">{src['text']}</div>
                </div>""", unsafe_allow_html=True)

        st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()
