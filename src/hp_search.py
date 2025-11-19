from __future__ import annotations

from typing import List, Dict, Any

import pandas as pd

from retrieval_api import search_hybrid
from retrieval_profiles import RETRIEVAL_PROFILES, DEFAULT_PROFILE_NAME

# Map old profile names (from earlier notebooks/UI) to the new ones.
PROFILE_ALIASES = {
    "hybrid_minilm_bm25": "fast_minilm",
    "prod_default": "prod_e5_balanced",
}


def _resolve_profile(
    profile_name: str | None,
    k: int | None,
    alpha: float | None,
) -> tuple[str, str, int, float]:
    """
    Decide which retrieval profile + parameters to actually use.
    Returns (resolved_profile_name, model_key, k_eff, alpha_eff).
    """
    if not profile_name:
        profile_name = DEFAULT_PROFILE_NAME

    # If the UI sends an old name, map it to a new profile
    profile_name = PROFILE_ALIASES.get(profile_name, profile_name)

    profile = RETRIEVAL_PROFILES.get(profile_name, RETRIEVAL_PROFILES[DEFAULT_PROFILE_NAME])

    model_key = profile.get("model", "minilm_l6")
    # If the UI slider gives k, let it override; otherwise use profile default
    k_eff = int(k) if k is not None and k > 0 else int(profile.get("k", 5))

    # If the UI slider gives alpha, let it override; otherwise use profile default
    if alpha is None:
        alpha_eff = float(profile.get("alpha", 0.6))
    else:
        alpha_eff = float(alpha)

    # Sanity checks
    k_eff = max(k_eff, 1)

    return profile_name, model_key, k_eff, alpha_eff


def run_search(query: str, k: int, alpha: float, profile_name: str) -> List[Dict[str, Any]]:
    """
    Main entry point used by streamlit_app.py.

    Args:
        query: user question
        k:     requested top-k (may be overridden by profile)
        alpha: requested alpha (may be overridden by profile)
        profile_name: retrieval profile key from the UI

    Returns:
        List of dicts with at least:
          - title
          - text
          - book
          - chapter
          - score
    """
    profile_name, model_key, k_eff, alpha_eff = _resolve_profile(
        profile_name=profile_name,
        k=k,
        alpha=alpha,
    )

    # Call the underlying retriever (semantic-only hybrid wrapper)
    hits_df: pd.DataFrame = search_hybrid(
        query=query,
        k=k_eff,
        alpha=alpha_eff,
        key=model_key,
        m=None,                 # let retrieval_api choose suitable M
        enable_house_bias=False # keep things simple for the public demo
    )

    results: List[Dict[str, Any]] = []

    for _, row in hits_df.iterrows():
        # Build a nice human-readable title from whatever metadata we have
        title_parts = []

        if "book" in row and pd.notna(row["book"]):
            title_parts.append(str(row["book"]))

        if "chapter" in row and pd.notna(row["chapter"]):
            title_parts.append(str(row["chapter"]))

        if "title" in row and pd.notna(row["title"]):
            title_parts.append(str(row["title"]))

        title = " — ".join(title_parts) if title_parts else "Passage"

        results.append(
            {
                "title": title,
                "text": row.get("text", ""),
                "book": row.get("book"),
                "chapter": row.get("chapter"),
                "score": float(row.get("final_score", 0.0)),
                "profile": profile_name,
                "model_key": model_key,
            }
        )

    return results
