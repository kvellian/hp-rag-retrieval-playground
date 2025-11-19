from __future__ import annotations

import os
import sys
import re
import time
from pathlib import Path
from typing import List, Dict, Any

import streamlit as st

# =============================================================================
# Paths & imports
# =============================================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

try:
    from hp_search import run_search
except Exception as e:
    st.error(f"Failed to import run_search from hp_search: {e}")
    st.stop()

try:
    from retrieval_profiles import RETRIEVAL_PROFILES, DEFAULT_PROFILE_NAME
except Exception as e:
    st.error(f"Failed to import retrieval_profiles: {e}")
    st.stop()

PROFILE_NAMES = list(RETRIEVAL_PROFILES.keys())

# =============================================================================
# Query normalization helpers
# =============================================================================

# Simple heuristic synonym normalization for kinship terms
KINSHIP_REPLACEMENTS = {
    r"\bmom\b": "mother",
    r"\bmum\b": "mother",
    r"\bdad\b": "father",
    r"\bdaddy\b": "father",
}


def apply_kinship_synonyms(raw_query: str) -> tuple[str, str]:
    """
    Replace informal kinship terms (mom, dad, etc.) with more formal versions
    that actually appear in the HP corpus (mother, father).
    """
    q = raw_query or ""
    original = q
    for pattern, repl in KINSHIP_REPLACEMENTS.items():
        q = re.sub(pattern, repl, q, flags=re.IGNORECASE)

    if q == original:
        info = "No kinship terms needed normalization."
    else:
        info = "Normalized kinship terms (e.g., mom→mother, dad→father) before retrieval."
    return q, info


def normalize_query_with_llm(raw_query: str) -> tuple[str, str]:
    """
    Use a small LLM (gpt-4o-mini) to rewrite the question into clean,
    canonical Harry Potter phrasing.

    If no API key is set or anything fails, returns the original query.
    """
    raw_query = (raw_query or "").strip()
    if not raw_query:
        return raw_query, "Empty query — nothing to normalize."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return raw_query, "No OPENAI_API_KEY set — using original query."

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        system_msg = (
            "You rewrite user questions.\n"
            "Fix grammar, expand shorthand, and choose clearer wording.\n"
            "Prefer formal terms (mother/father instead of mom/dad) and full names "
            "when obvious (e.g., 'Harry' → 'Harry Potter').\n"
            "Preserve the original meaning.\n"
            "Return ONLY the rewritten question text, no explanations."
        )

        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": raw_query},
            ],
            temperature=0.2,
        )
        normalized = (resp.choices[0].message.content or "").strip()

        if not normalized:
            return raw_query, "LLM returned empty text — using original query."
        if normalized == raw_query:
            return normalized, "LLM normalization: no change (query already clean)."
        return normalized, "LLM normalization: query rewritten for clarity/HP phrasing."
    except Exception as e:
        return raw_query, f"LLM normalization failed: {e} — using original query."


# =============================================================================
# LLM helpers for answering
# =============================================================================

def build_hp_prompt(question: str, context: str) -> str:
    return (
        "You answer questions about the Harry Potter books.\n"
        "Use ONLY the context below. If the answer is not in the context, say so.\n\n"
        f"Question: {question}\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Answer:"
    )


def call_llm(question: str, context: str, model_name: str, temperature: float) -> str:
    context = (context or "").strip()
    if not context:
        return "No context retrieved — cannot answer."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        snippet = context[:700]
        return "LLM disabled (no OPENAI_API_KEY). Showing retrieved context:\n\n" + snippet

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = build_hp_prompt(question, context)
        resp = client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You answer Harry Potter questions using only the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=float(temperature),
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        snippet = context[:700]
        return f"LLM call failed ({e}). Showing context instead:\n\n" + snippet


# =============================================================================
# Streamlit UI
# =============================================================================

st.set_page_config(page_title="HP RAG — Retrieval Playground", layout="wide")

# ---------- Sidebar: settings ----------
st.sidebar.title("Settings")

# Retrieval profile
try:
    default_profile_index = PROFILE_NAMES.index(DEFAULT_PROFILE_NAME)
except ValueError:
    default_profile_index = 0

profile_name = st.sidebar.selectbox(
    "Retrieval profile",
    PROFILE_NAMES,
    index=default_profile_index,
)

profile_cfg = RETRIEVAL_PROFILES.get(profile_name, RETRIEVAL_PROFILES[DEFAULT_PROFILE_NAME])
default_alpha = float(profile_cfg.get("alpha", 0.6))
default_k = int(profile_cfg.get("k", 5))

# Top-k slider (how many passages we actually use)
top_k = st.sidebar.slider("Top-k passages", min_value=3, max_value=20, value=min(5, default_k), step=1)

# Alpha slider
alpha = st.sidebar.slider("Hybrid weight α", min_value=0.0, max_value=1.0, value=default_alpha, step=0.05)

# OpenAI model selection (DEFAULT = gpt-4o-mini)
OPENAI_MODELS = ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"]
openai_model = st.sidebar.selectbox("OpenAI model", OPENAI_MODELS, index=0)

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.2, step=0.05)

show_passages = st.sidebar.checkbox("Show retrieved passages", value=True)

# ---------- Main layout ----------
st.title("⚡ Harry Potter RAG — Retrieval Playground")

st.markdown(
    """
**Created by:** Ken Vellian  
**Data Science Capstone Project**  
**Professor:** David Hubbard — DePaul University, Fall 2026
"""
)

with st.expander("What is this?", expanded=True):
    st.markdown(
        """
This app lets you ask questions about the Harry Potter books. Under the hood it
uses a Retrieval-Augmented Generation (RAG) pipeline:

1. Your question is lightly **normalized** (e.g., *mom → mother*).
2. A tuned **retriever** finds the most relevant passages from the books.
3. An **OpenAI model** uses only those passages to generate a grounded answer.
        """
    )

st.markdown("## Ask a question about the Harry Potter books")

query = st.text_input("Type your question", placeholder="Who is Harry's mom?")
ask_clicked = st.button("Ask")

if ask_clicked and query.strip():
    original_query = query.strip()

    # ==============================
    # Normalization + retrieval
    # ==============================
    overall_t0 = time.time()

    kin_q, kin_info = apply_kinship_synonyms(original_query)
    norm_q, norm_info = normalize_query_with_llm(kin_q)

    retrieval_t0 = time.time()
    with st.status("🔍 Retrieving relevant passages...", expanded=True) as status_box:
        try:
            # Use our hp_search.run_search wrapper
            results: List[Dict[str, Any]] = run_search(
                query=norm_q,
                k=int(top_k),
                alpha=float(alpha),
                profile_name=profile_name,
            )
            retrieval_t1 = time.time()
            status_box.update(label="✅ Retrieval complete", state="complete")
        except Exception as e:
            retrieval_t1 = time.time()
            status_box.update(label="❌ Retrieval failed", state="error")
            st.error(f"Error during retrieval: {e}")
            results = []

    retrieval_latency = retrieval_t1 - retrieval_t0

    if not results:
        st.warning("No passages retrieved.")
        st.stop()

    # ==============================
    # Build context + call LLM
    # ==============================
    # Build context string from the retrieved passages
    context_chunks = [r.get("text", "") for r in results if r.get("text")]
    context = "\n\n".join(context_chunks)

    llm_t0 = time.time()
    answer = call_llm(
        question=original_query,
        context=context,
        model_name=openai_model,
        temperature=temperature,
    )
    llm_t1 = time.time()
    llm_latency = llm_t1 - llm_t0

    # ==============================
    # Display results
    # ==============================

    st.markdown("## ✅ Answer")
    st.write(answer)

    st.markdown("## ⏱ Latency")
    st.write(f"Retrieval: **{retrieval_latency:.2f} s**")
    st.write(f"LLM generation: **{llm_latency:.2f} s**")

    if show_passages:
        st.markdown("## 📖 Retrieved passages")
        for i, r in enumerate(results, start=1):
            title = r.get("title") or f"Chunk {i}"
            score = r.get("score", 0.0)
            header = f"[{i}] {title} — score={score:.4f}" if isinstance(score, (int, float)) else f"[{i}] {title}"
            with st.expander(header):
                meta_bits = []
                if r.get("book"):
                    meta_bits.append(f"**Book:** {r['book']}")
                if r.get("chapter"):
                    meta_bits.append(f"**Chapter:** {r['chapter']}")
                if meta_bits:
                    st.markdown(" | ".join(meta_bits))
                st.write(r.get("text", ""))

    # Optional debug info
    with st.expander("🔧 Debug info"):
        st.json(
            {
                "original_query": original_query,
                "kinship_normalized_query": kin_q,
                "llm_normalized_query": norm_q,
                "kinship_info": kin_info,
                "llm_norm_info": norm_info,
                "profile": profile_name,
                "alpha": float(alpha),
                "top_k": int(top_k),
                "openai_model": openai_model,
            }
        )

elif not query.strip():
    st.info("Type a Harry Potter question above and click **Ask**.")
