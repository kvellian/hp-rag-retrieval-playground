"""
Streamlit app entrypoint for:
⚡ Harry Potter RAG — Retrieval Playground

Created by: Ken Vellian
Data Science Capstone Project
Professor: David Hubbard
DePaul University | Fall 2026

NOTE: This app expects:
- OPENAI_API_KEY to be set in environment / Streamlit secrets
- A helper function run_search(...) in src/hp_search.py
  that performs retrieval and returns a list of hits.

See `run_retrieval()` below for the expected signature / return value.
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st
from openai import OpenAI

# -------------------------------------------------------------------
# Path setup so we can import from src/
# -------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    # 👉 You may need to adjust this import depending on your hp_search.py
    # I’m assuming there is (or will be) a function:
    #   run_search(query: str, k: int, alpha: float, profile_name: str) -> List[Dict]
    from hp_search import run_search
except ImportError:
    run_search = None

# -------------------------------------------------------------------
# OpenAI client
# -------------------------------------------------------------------
def get_openai_client() -> OpenAI:
    """
    Create an OpenAI client. Requires OPENAI_API_KEY to be set
    via environment variable or Streamlit secrets.
    """
    # Streamlit Cloud: you can set OPENAI_API_KEY in Secrets.
    if "OPENAI_API_KEY" not in os.environ:
        # If using st.secrets, sync it into os.environ once
        if "OPENAI_API_KEY" in st.secrets:
            os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

    return OpenAI()


# -------------------------------------------------------------------
# Retrieval + LLM answer
# -------------------------------------------------------------------
def run_retrieval(
    query: str,
    k: int,
    alpha: float,
    profile_name: str,
) -> List[Dict[str, Any]]:
    """
    Wrapper around your retrieval backend.

    Expected behavior of hp_search.run_search:
        hits = run_search(
            query=query,
            k=k,
            alpha=alpha,
            profile_name=profile_name,
        )

    where `hits` is a list of dicts like:
        {
            "rank": int,          # optional
            "score": float,       # retrieval score
            "chunk_id": Any,      # optional ID
            "doc_id": Any,        # optional
            "source": str,        # e.g. "HP1 - Chapter 3"
            "text": str,          # retrieved chunk text  (REQUIRED)
        }

    If your actual function uses a slightly different signature or keys,
    just adapt this wrapper accordingly.
    """
    if run_search is None:
        raise RuntimeError(
            "Could not import run_search from hp_search. "
            "Please open src/hp_search.py and implement:\n\n"
            "    def run_search(query: str, k: int, alpha: float, profile_name: str) -> List[dict]:\n"
            "        ...\n"
        )

    hits = run_search(query=query, k=k, alpha=alpha, profile_name=profile_name)
    return hits


def generate_answer_from_hits(
    client: OpenAI,
    query: str,
    hits: List[Dict[str, Any]],
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.2,
) -> str:
    """
    Call the OpenAI model with the retrieved context to generate
    a grounded answer.
    """
    if not hits:
        return "I couldn't find any relevant passages in the knowledge base."

    # Build context string
    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        source = hit.get("source") or hit.get("doc_id") or f"Chunk {i}"
        text = hit.get("text", "")
        context_blocks.append(f"[{i}] Source: {source}\n{text}")

    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You are a helpful assistant answering questions about the Harry Potter books. "
        "Use ONLY the provided context excerpts from the books. "
        "If the answer cannot be found in the context, say that you don't know and avoid guessing. "
        "Cite the excerpt numbers you used like [1], [2] when appropriate."
    )

    user_prompt = (
        f"User question:\n{query}\n\n"
        f"Here are the retrieved context excerpts from the books:\n\n{context}\n\n"
        "Now answer the question using ONLY this information."
    )

    response = client.chat.completions.create(
        model=model_name,
        temperature=temperature,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return response.choices[0].message.content.strip()


# -------------------------------------------------------------------
# Simple CSV logging
# -------------------------------------------------------------------
def log_interaction(
    query: str,
    answer: str,
    k: int,
    alpha: float,
    profile_name: str,
    model_name: str,
    latency_retrieval: float,
    latency_llm: float,
    n_hits: int,
) -> None:
    """
    Append a row to logs/usage_log.csv (if possible).
    This is best-effort only; failures are swallowed.
    """
    try:
        import csv
        from datetime import datetime

        logs_dir = ROOT / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = logs_dir / "usage_log.csv"

        file_exists = log_path.exists()
        with log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(
                    [
                        "timestamp",
                        "query",
                        "answer_preview",
                        "k",
                        "alpha",
                        "profile_name",
                        "model_name",
                        "latency_retrieval_s",
                        "latency_llm_s",
                        "n_hits",
                    ]
                )
            writer.writerow(
                [
                    datetime.utcnow().isoformat(),
                    query,
                    (answer[:120] + "…") if len(answer) > 120 else answer,
                    k,
                    alpha,
                    profile_name,
                    model_name,
                    round(latency_retrieval, 3),
                    round(latency_llm, 3),
                    n_hits,
                ]
            )
    except Exception:
        # Logging should never break the app
        pass


# -------------------------------------------------------------------
# Streamlit UI
# -------------------------------------------------------------------
def main() -> None:
    st.set_page_config(
        page_title="HP RAG — Retrieval Playground",
        page_icon="⚡",
        layout="wide",
    )

    # ---- Header ----
    st.markdown(
        """
# ⚡ Harry Potter RAG — Retrieval Playground

**Created by:** Ken Vellian  
**Data Science Capstone Project**  
**Professor:** David Hubbard — DePaul University, Fall 2026
        """
    )

    with st.expander("What is this?", expanded=False):
        st.write(
            "This app lets you ask questions about the Harry Potter books. "
            "Under the hood it uses a Retrieval-Augmented Generation (RAG) pipeline: "
            "a hybrid retriever finds relevant passages from the books, which are "
            "then passed to an OpenAI model to generate a grounded answer."
        )

    # ---- Sidebar controls ----
    st.sidebar.header("Settings")

    # You can adjust this list to match your retrieval_profiles
    profile_name = st.sidebar.selectbox(
        "Retrieval profile",
        options=[
            "hybrid_minilm_bm25",
            "hybrid_e5_bm25",
            "bm25_only",
        ],
        index=0,
        help="Choose which retrieval configuration to use.",
    )

    k = st.sidebar.slider(
        "Top-k passages",
        min_value=3,
        max_value=10,
        value=5,
        step=1,
        help="Number of passages to retrieve from the index.",
    )

    alpha = st.sidebar.slider(
        "Hybrid weight α",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Blend between sparse and dense retrievers (interpretation depends on your backend).",
    )

    model_name = st.sidebar.selectbox(
        "OpenAI model",
        options=["gpt-4o-mini", "gpt-4o"],
        index=0,
    )

    temperature = st.sidebar.slider(
        "Temperature",
        min_value=0.0,
        max_value=1.0,
        value=0.2,
        step=0.05,
        help="Higher values = more creative, lower values = more deterministic.",
    )

    show_debug = st.sidebar.checkbox("Show retrieved passages", value=True)

    # ---- Main input ----
    st.subheader("Ask a question about the Harry Potter books")
    query = st.text_input(
        "Type your question",
        placeholder="e.g., Who is Harry's best friend?",
    )

    ask_button = st.button("Ask", type="primary")

    if ask_button:
        if not query.strip():
            st.warning("Please enter a question first.")
            st.stop()

        if "OPENAI_API_KEY" not in os.environ and "OPENAI_API_KEY" not in st.secrets:
            st.error(
                "OPENAI_API_KEY is not set. "
                "Add it as an environment variable or in Streamlit secrets."
            )
            st.stop()

        # Retrieval
        st.write("🔎 Retrieving relevant passages...")
        t0 = time.time()
        try:
            hits = run_retrieval(
                query=query,
                k=k,
                alpha=alpha,
                profile_name=profile_name,
            )
        except Exception as e:
            st.error(f"Error during retrieval: {e}")
            st.stop()
        t1 = time.time()
        latency_retrieval = t1 - t0

        # LLM answer
        st.write("🧠 Asking the OpenAI model...")
        client = get_openai_client()
        t2 = time.time()
        try:
            answer = generate_answer_from_hits(
                client=client,
                query=query,
                hits=hits,
                model_name=model_name,
                temperature=temperature,
            )
        except Exception as e:
            st.error(f"Error during LLM call: {e}")
            st.stop()
        t3 = time.time()
        latency_llm = t3 - t2

        # Display answer
        st.markdown("### ✅ Answer")
        st.write(answer)

        # Metrics
        st.markdown("### ⏱️ Latency")
        st.write(
            f"- Retrieval: **{latency_retrieval:.2f} s**  \n"
            f"- LLM generation: **{latency_llm:.2f} s**"
        )

        # Debug: show top passages
        if show_debug:
            st.markdown("### 📚 Retrieved passages")
            if not hits:
                st.info("No passages were retrieved.")
            else:
                for i, hit in enumerate(hits, start=1):
                    score = hit.get("score")
                    source = hit.get("source") or hit.get("doc_id") or f"Chunk {i}"
                    text = hit.get("text", "")

                    with st.expander(f"[{i}] {source} — score={score:.4f}" if score is not None else f"[{i}] {source}", expanded=False):
                        st.write(text)

        # Log interaction (best effort)
        log_interaction(
            query=query,
            answer=answer,
            k=k,
            alpha=alpha,
            profile_name=profile_name,
            model_name=model_name,
            latency_retrieval=latency_retrieval,
            latency_llm=latency_llm,
            n_hits=len(hits),
        )


if __name__ == "__main__":
    main()
