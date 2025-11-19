import os
import sys
import re
from pathlib import Path
from typing import Tuple

import pandas as pd
import streamlit as st

# ------------------------------------------------------------------
# Paths & imports
# ------------------------------------------------------------------

# Repo root = folder where this file lives
BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR / "src"

# Make src importable
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

# Import our retrieval engine + profiles
try:
    from hp_search import run_search as hp_search_fn
except Exception as e:
    st.error(f"Failed to import hp_search.run_search: {e}")
    st.stop()

try:
    from retrieval_profiles import RETRIEVAL_PROFILES, DEFAULT_PROFILE_NAME
except Exception as e:
    st.error(f"Failed to import retrieval_profiles: {e}")
    st.stop()

PROFILE_NAMES = list(RETRIEVAL_PROFILES.keys())

# ------------------------------------------------------------------
# Query normalization helpers
# ------------------------------------------------------------------

# Simple heuristic synonym normalization for kinship terms
KINSHIP_REPLACEMENTS = {
    r"\bmom\b": "mother",
    r"\bmum\b": "mother",
    r"\bdad\b": "father",
    r"\bdaddy\b": "father",
}


def apply_kinship_synonyms(raw_query: str) -> Tuple[str, str]:
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


def normalize_query_with_llm(raw_query: str) -> Tuple[str, str]:
    """
    Use a small LLM to rewrite the question into clean, canonical Harry Potter phrasing.
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


# ------------------------------------------------------------------
# LLM helpers for answering (always gpt-4o-mini under the hood)
# ------------------------------------------------------------------

def build_hp_prompt(question: str, context: str) -> str:
    return (
        "You are an assistant that answers questions.\n"
        "Use ONLY the context below. If the answer is not in the context, say so.\n\n"
        f"Question: {question}\n\n"
        "Context:\n"
        f"{context}\n\n"
        "Answer:"
    )


def call_llm(question: str, context: str) -> str:
    context = (context or "").strip()
    if not context:
        return "No context retrieved — cannot answer."

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        snippet = context[:600]
        return "LLM disabled (no OPENAI_API_KEY). Showing retrieved context:\n\n" + snippet

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = build_hp_prompt(question, context)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You answer Harry Potter questions using only the provided context.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        snippet = context[:600]
        return f"LLM call failed ({e}). Showing context instead:\n\n" + snippet


# ------------------------------------------------------------------
# Streamlit UI (no sidebar – all controls in the main page)
# ------------------------------------------------------------------

st.set_page_config(page_title="HP RAG — Retrieval Playground", layout="wide")
st.title("⚡ Harry Potter RAG — Retrieval Playground")

st.markdown(
    """
**Created by:** Ken Vellian  
**Data Science Capstone Project**  
**Professor:** David Hubbard — DePaul University, Fall 2026
"""
)

st.markdown(
    """
This app lets you **interactively explore** the retrieval behavior of your RAG system
using the three tuned embedding models:

- `prod_e5_balanced` → **e5_small** — balanced accuracy + latency  
- `fast_minilm` → **minilm_l6** — fastest model with strong hit@k  
- `max_precision_bge` → **bge_base** — highest early precision (heavier model)  

**How to use this playground:**

1. Pick a **retrieval profile** above. Notice how it sets the embedding model, α, and k defaults.  
2. Type a Harry Potter question in the box and click **Search**.  
3. Play with **α (dense vs lexical)** and **Base k** to see how the retrieved passages and scores change.  
4. Adjust **Expanded k**, **Show top N**, and **Use LLM to answer** if you want more passages or a generated answer.  

Behind the scenes, your question is lightly normalized (e.g., “mom” → “mother”) and then sent to the retriever, but only your original question is shown here.
"""
)

# ---- Top control row: profile + model + sliders ----
col_profile, col_model, col_alpha, col_k, col_topn = st.columns([1.3, 1.3, 1, 1, 1])

with col_profile:
    profile_name = st.selectbox(
        "Retrieval profile",
        PROFILE_NAMES,
        index=PROFILE_NAMES.index(DEFAULT_PROFILE_NAME),
    )

profile_cfg = RETRIEVAL_PROFILES[profile_name]
default_model = profile_cfg["model"]
default_alpha = float(profile_cfg["alpha"])
default_k = int(profile_cfg["k"])

with col_model:
    st.markdown("**Embedding model**")
    # Read-only: driven by the selected profile
    st.write(f"`{default_model}`")

# The model used for retrieval is fixed by the profile
model_key = default_model

with col_alpha:
    alpha = st.slider("α (dense vs lexical)", 0.0, 1.0, float(default_alpha), 0.05)

with col_k:
    base_k = st.slider("Base k", 5, 40, int(default_k), 1)

with col_topn:
    top_n = st.slider("Show top N", 3, 15, 5, 1)

# second row: expanded k + options
col_exp, col_llm, col_debug = st.columns([1, 1, 1])
with col_exp:
    exp_k = st.slider("Expanded k", 10, 80, min(3 * base_k, 80), 1)
with col_llm:
    use_llm = st.checkbox("Use LLM to answer", value=False)
with col_debug:
    show_debug = st.checkbox("Show debug info", value=True)

# ---- Query box ----
query = st.text_input("🧙‍♂️ Ask a Harry Potter question")

st.markdown("---")

if st.button("Search") and query.strip():
    raw_q = query.strip()

    # 1) heuristic kinship normalization (mom/dad → mother/father)
    kin_q, kin_info = apply_kinship_synonyms(raw_q)

    # 2) LLM normalization (if API key is available)
    norm_q, norm_info = normalize_query_with_llm(kin_q)

    # ---- Run retrieval on the normalized query ----
    # IMPORTANT: use positional args to match run_search signature exactly
    out = hp_search_fn(
        norm_q,
        int(base_k),
        int(exp_k),
        model_key,
        float(alpha),
        int(top_n),
        True,  # apply_boosts
    )

    st.subheader("🔍 Query info")
    st.write("**Original query:**", raw_q)

    results = out.get("results")
    if not isinstance(results, pd.DataFrame):
        try:
            results_df = pd.DataFrame(results)
        except Exception:
            st.error("Could not convert results to DataFrame.")
            st.write("Raw results:", results)
            st.stop()
    else:
        results_df = results.copy()

    if results_df.empty:
        st.warning("No passages retrieved.")
        st.stop()

    st.subheader("📚 Top passages")
    display_cols = [
        c
        for c in [
            "title",
            "final_score",
            "sem",
            "lex",
            "doc_type",
            "book_title",
            "chapter_title",
            "chunk_id",
        ]
        if c in results_df.columns
    ]
    if display_cols:
        st.dataframe(results_df[display_cols], use_container_width=True)
    else:
        st.dataframe(results_df, use_container_width=True)

    # Build context: use all returned passages' text
    texts = results_df.get("text", pd.Series([], dtype=str)).fillna("")
    context = "\n\n".join(texts.tolist())

    st.subheader("✅ Answer")
    if use_llm:
        # Important: we ask on the ORIGINAL question, but with context from normalized query.
        answer = call_llm(raw_q, context)
        st.write(answer)
    else:
        snippet = (context or "").strip()[:900]
        if snippet:
            st.write("LLM disabled. Showing retrieved context snippet:")
            st.code(snippet)
        else:
            st.write("No context retrieved.")

    st.subheader("📖 Passages")
    for i, row in results_df.iterrows():
        header = f"{i+1}. {row.get('title', '(no title)')}"
        with st.expander(header):
            meta_bits = []
            if row.get("doc_type"):
                meta_bits.append(f"**Type:** {row['doc_type']}")
            if row.get("book_title"):
                meta_bits.append(f"**Book:** {row['book_title']}")
            if row.get("chapter_title"):
                meta_bits.append(f"**Chapter:** {row['chapter_title']}")
            if meta_bits:
                st.markdown(" | ".join(meta_bits))
            st.write(row.get("text", ""))

    if show_debug:
        st.subheader("🔧 Debug info")
        st.json(
            {
                "profile": profile_name,
                "model": model_key,
                "alpha": float(alpha),
                "base_k": int(base_k),
                "exp_k": int(exp_k),
                "top_n": int(top_n),
                "kinship_normalized_query": kin_q,
                "llm_normalized_query": norm_q,
                "kinship_info": kin_info,
                "llm_norm_info": norm_info,
                "expanded_query": out.get("expanded_query"),
            }
        )
else:
    st.info("Enter a question above and click **Search** to run retrieval.")
