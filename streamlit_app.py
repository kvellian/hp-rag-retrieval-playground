 """
hp_search.py
Clean, simple interface for calling the hybrid retriever.

This module exposes one function:

    run_search(query, base_k, exp_k, model_key, alpha, top_return, apply_boosts=True)

which returns:
    {
        "results": DataFrame,
        "expanded_query": None
    }

It wraps the `search_hybrid` function from retrieval_api.py.
"""

from typing import Optional, Dict, Any
import pandas as pd

# Import the hybrid retriever
from retrieval_api import search_hybrid


# ----------------------------------------------------------------------
# MAIN ENTRY POINT FOR THE STREAMLIT APP
# ----------------------------------------------------------------------
def run_search(
    query: str,
    base_k: int,
    exp_k: int,
    model_key: str,
    alpha: float,
    top_return: int,
    apply_boosts: bool = True,
) -> Dict[str, Any]:
    """
    Execute a hybrid semantic search over the HP corpus.

    Parameters
    ----------
    query : str
        The normalized question (LLM-normalized + synonym rewrite).
    base_k : int
        k to return after re-ranking.
    exp_k : int
        How many FAISS neighbors to retrieve before re-ranking.
    model_key : str
        Which embedding model to use (minilm_l6, e5_small, bge_base).
    alpha : float
        Weight for semantic vs lexical score (semantic_weight = alpha).
    top_return : int
        Number of top passages to return to the UI.
    apply_boosts : bool
        Whether to apply title/entity boosts & penalties. (Kept for future flexibility.)

    Returns
    -------
    dict with:
        "results": DataFrame of retrieved passages (text + metadata + scores)
        "expanded_query": None  (placeholder for future expansion)
    """

    # NOTE: The deployed retrieval currently uses semantic-only scoring
    # due to safety constraints (no TF-IDF matrix shipped).
    # alpha is still passed for forward compatibility.

    # Call underlying retrieval engine
    hits_df = search_hybrid(
        query,
        k=top_return,   # how many results returned AFTER reranking
        alpha=alpha,
        key=model_key,
        m=exp_k,        # FAISS search depth
    )

    # Ensure DataFrame
    if not isinstance(hits_df, pd.DataFrame):
        try:
            hits_df = pd.DataFrame(hits_df)
        except Exception:
            raise ValueError("search_hybrid did not return a DataFrame-like result.")

    # Trim to requested top N
    results = hits_df.head(top_return).reset_index(drop=True)

    return {
        "results": results,
        "expanded_query": None,     # placeholder for optional LLM expansions
    }
