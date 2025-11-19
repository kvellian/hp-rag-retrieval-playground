from pathlib import Path
import re
import numpy as np
import pandas as pd

from retrieval_api import search_hybrid, chunks

# --- paths (mainly here in case we want logging later) ---
BASE_DIR = Path("/content/drive/MyDrive/hp_rag")
EVAL_DIR = BASE_DIR / "eval"
EVAL_DIR.mkdir(parents=True, exist_ok=True)

# --- 1) HP-aware expansion rules -----------------------------------------

def hp_expand(q: str) -> str:
    ql = q.lower()

    # spells
    if "expelliarmus" in ql:
        return q + " Disarming Charm makes opponent drop their wand"

    # houses
    if "which house is hermione" in ql or ("hermione" in ql and "house" in ql):
        return q + " Gryffindor House sorting hat"
    if "which house is draco" in ql or ("draco malfoy" in ql and "house" in ql):
        return q + " Slytherin House Malfoy family"

    # familiars / pets
    if "ron's rat" in ql or "rons rat" in ql:
        return q + " Scabbers Peter Pettigrew"
    if "harry's owl" in ql or "harrys owl" in ql:
        return q + " Hedwig snowy owl"

    # sirius death
    if "who killed sirius" in ql:
        return q + " Bellatrix Lestrange killed Sirius Black Department of Mysteries"

    # half-blood prince
    if "half-blood prince" in ql:
        return q + " Severus Snape identity"

    return q

# --- 2) improved_rerank: merge base + expanded, normalize, overlap bonus ---

def improved_rerank(base_df: pd.DataFrame,
                    exp_df: pd.DataFrame,
                    orig_q: str,
                    exp_q: str,
                    top_n: int = 50) -> pd.DataFrame:
    """
    Merge base + expanded hits and re-score them using:
      - normalized base scores (from final_score)
      - overlap between expanded-query tokens and title
    This matches the “old NB6” behavior.
    """
    base_df = base_df.copy()
    exp_df = exp_df.copy()
    base_df["source"] = "base"
    exp_df["source"] = "exp"

    base_df["title_lc"] = base_df["title"].fillna("").str.lower()
    exp_df["title_lc"]  = exp_df["title"].fillna("").str.lower()

    # tokens from expanded query (minus stopwords)
    exp_tokens = re.findall(r"\w+", (exp_q or "").lower())
    stop = {"what","who","is","the","in","of","and","a","an"}
    exp_tokens = [t for t in exp_tokens if t not in stop]

    def overlap_count(title_lc: str) -> int:
        return sum(1 for t in exp_tokens if t in title_lc)

    base_df["exp_overlap"] = base_df["title_lc"].apply(overlap_count)
    exp_df["exp_overlap"]  = exp_df["title_lc"].apply(overlap_count)

    # normalize original scores into [0,1]
    def _norm(col: pd.Series) -> pd.Series:
        arr = col.to_numpy(dtype=np.float32)
        if arr.size == 0:
            return pd.Series([], index=col.index, dtype=np.float32)
        lo, hi = float(arr.min()), float(arr.max())
        if hi - lo < 1e-6:
            return pd.Series(np.ones_like(arr) * 0.5, index=col.index)
        return pd.Series((arr - lo) / (hi - lo), index=col.index)

    if "final_score" in base_df.columns:
        base_df["base_score"] = _norm(base_df["final_score"])
    else:
        base_df["base_score"] = 0.5

    if "final_score" in exp_df.columns:
        exp_df["base_score"] = _norm(exp_df["final_score"])
    else:
        exp_df["base_score"] = 0.5

    # expansion bonus: more overlap → slightly higher score
    base_df["exp_bonus"] = base_df["exp_overlap"] * 0.12
    exp_df["exp_bonus"]  = exp_df["exp_overlap"] * 0.12

    merged = pd.concat([base_df, exp_df], ignore_index=True)

    # dedupe by chunk_id when possible, else by title
    if "chunk_id" in merged.columns:
        merged = merged.drop_duplicates(subset=["chunk_id"], keep="first")
    else:
        merged = merged.drop_duplicates(subset=["title"], keep="first")

    merged["rerank_score"] = merged["base_score"] + merged["exp_bonus"]
    merged = merged.sort_values("rerank_score", ascending=False).reset_index(drop=True)
    return merged.head(top_n)

# --- 3) light HP-specific title-based boosts ------------------------------

HP_SYNONYMS = {
    r"\bexpelliarmus\b": [
        "Disarming Charm",
        "Disarms",
    ],
    r"best friend": [
        "Ron Weasley",
        "Ronald Weasley",
    ],
    r"hermione": [
        "Gryffindor",
        "Sorting Hat",
    ],
}

def apply_hp_boosts(df: pd.DataFrame,
                    query: str,
                    bonus: float = 0.35) -> pd.DataFrame:
    """
    Light, general HP-aware nudges applied on top of rerank_score.
    No hard overrides; just small boosts to obviously-correct pages.
    """
    df = df.copy()
    q = (query or "").lower()

    if "rerank_score" not in df.columns:
        if "final_score" in df.columns:
            df["rerank_score"] = df["final_score"].astype(float)
        else:
            df["rerank_score"] = 1.0

    title_lc = df["title"].fillna("").str.lower()

    for pattern, good_titles in HP_SYNONYMS.items():
        if re.search(pattern, q):
            for good in good_titles:
                mask = title_lc.str.contains(good.lower())
                df.loc[mask, "rerank_score"] = df.loc[mask, "rerank_score"] + bonus

    df = df.sort_values("rerank_score", ascending=False).reset_index(drop=True)
    return df

# --- 4) main entry point for other notebooks ------------------------------

def hp_search(query: str,
              base_k: int = 15,
              exp_k: int = 40,
              model_key: str = "bge_base",
              alpha: float | None = None,
              top_return: int = 5,
              apply_boosts: bool = True) -> dict:
    """
    High-level HP search used by Notebooks 7+:

      1) Expand query with hp_expand(...)
      2) Run base + expanded retrieval
      3) Improved rerank (overlap-based normalization)
      4) Optional HP-aware boosts
      5) Return top N hits + expanded query
    """
    exp_query = hp_expand(query)

    base_hits = search_hybrid(query,    k=base_k, key=model_key, alpha=alpha)
    exp_hits  = search_hybrid(exp_query, k=exp_k, key=model_key, alpha=alpha)

    merged = improved_rerank(base_hits, exp_hits, query, exp_query, top_n=max(top_return, 50))

    if apply_boosts:
        merged = apply_hp_boosts(merged, query)

    merged = merged.head(top_return).reset_index(drop=True)

    return {
        "query": query,
        "expanded_query": exp_query,
        "results": merged,
    }
