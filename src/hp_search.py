"""
hp_search.py

Lightweight search wrapper for the Streamlit app.

- Uses repo-relative paths via retrieval_api (no /content, no Colab deps)
- Exposes a single function: run_search(query, k, model_key)
"""

from typing import Dict
import pandas as pd

from retrieval_api import (
    search_semantic,
    EMBED_MODELS,
)

# Default embedding model key (must exist in EMBED_MODELS)
DEFAULT_MODEL_KEY = "minilm_l6"


def get_available_models() -> Dict[str, str]:
    """
    Return a mapping of model_key -> human-readable name.
    Used by the Streamlit UI to populate the dropdown.
    """
    nice_names = {
        "minilm_l6": "MiniLM-L6 (small, fast, great baseline)",
        "e5_small": "E5-small-v2 (instruction-tuned)",
        "bge_base": "BGE-base-en-v1.5 (stronger, heavier)",
    }
    return {
        key: nice_names.get(key, key)
        for key in EMBED_MODELS.keys()
    }


def run_search(
    query: str,
    k: int = 5,
    model_key: str = DEFAULT_MODEL_KEY,
) -> pd.DataFrame:
    """
    Run a semantic search over precomputed HP chunks.

    Args:
        query: User question / search text.
        k:     Number of results to return.
        model_key: One of EMBED_MODELS keys.

    Returns:
        pandas.DataFrame with at least:
        ['title', 'book', 'chapter', 'chunk_id', 'text', 'score']
    """
    query = (query or "").strip()
    if not query:
        return pd.DataFrame()

    if model_key not in EMBED_MODELS:
        model_key = DEFAULT_MODEL_KEY

    # Delegate to retrieval_api; this will:
    # - Load hp_chunks.parquet from data/processed
    # - Load the FAISS index for the chosen model
    # - Encode the query and search
    hits = search_semantic(query=query, k=k, key=model_key)

    # For safety, make sure we always have a DataFrame
    if not isinstance(hits, pd.DataFrame):
        hits = pd.DataFrame(hits)

    # Rename score column if needed
    if "score" not in hits.columns:
        # Our retrieval_api uses 'sem_score' – expose a generic 'score' too
        if "sem_score" in hits.columns:
            hits["score"] = hits["sem_score"]

    return hits
