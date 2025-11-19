# Retrieval API for HP RAG — repo-relative, Streamlit-friendly version

import re
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
import yaml
from sentence_transformers import SentenceTransformer

# --------------------------------------------------------------------------------------
# Paths: resolve relative to the repo root (…/hp-rag-retrieval-playground)
# --------------------------------------------------------------------------------------

# /repo/src/retrieval_api.py  -> parent is /repo/src -> parent.parent is /repo
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"
IDX_DIR = DATA_DIR / "indexes"
CFG = BASE_DIR / "config" / "retrieval.yml"

# --------------------------------------------------------------------------------------
# Embedding models / FAISS indexes
# --------------------------------------------------------------------------------------

EMBED_MODELS = {
    "minilm_l6": "sentence-transformers/all-MiniLM-L6-v2",
    "e5_small": "intfloat/e5-small-v2",
    "bge_base": "BAAI/bge-base-en-v1.5",
}


def _prep_query_text(key: str, q: str) -> str:
    """For some models (e5, bge), we need the 'query: ' prefix."""
    if key.startswith("e5_") or key.startswith("bge_"):
        return f"query: {q}"
    return q


def _minmax(x: np.ndarray) -> np.ndarray:
    """Safe min–max normalization to [0, 1]."""
    x = np.asarray(x, dtype=np.float32)
    rng = np.ptp(x)
    if rng == 0.0:
        return np.zeros_like(x)
    return (x - x.min()) / (rng + 1e-9)


# --------------------------------------------------------------------------------------
# Load corpus chunks
# --------------------------------------------------------------------------------------

if (PROCESSED_DIR / "hp_chunks.parquet").exists():
    chunks = pd.read_parquet(PROCESSED_DIR / "hp_chunks.parquet")
elif (PROCESSED_DIR / "hp_chunks.csv").exists():
    chunks = pd.read_csv(PROCESSED_DIR / "hp_chunks.csv")
else:
    raise FileNotFoundError(
        f"No processed chunks found.\n"
        f"Expected one of:\n"
        f"  - {PROCESSED_DIR / 'hp_chunks.parquet'}\n"
        f"  - {PROCESSED_DIR / 'hp_chunks.csv'}"
    )

# --------------------------------------------------------------------------------------
# Load config
# --------------------------------------------------------------------------------------

if CFG.exists():
    with open(CFG, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f) or {}
else:
    _cfg = {}

DEFAULTS = _cfg.get("defaults", {})
ALPHAS = _cfg.get("alphas", {k: 0.6 for k in EMBED_MODELS})

BONUS1 = float(DEFAULTS.get("title_bonus_1tok", 0.05))
BONUS2 = float(DEFAULTS.get("title_bonus_2tok", 0.05))
ENTITY_BOOST = float(DEFAULTS.get("entity_boost", 1.08))
SHORT_PENALTY_LEN = int(DEFAULTS.get("short_penalty_len", 80))
SHORT_PENALTY = float(DEFAULTS.get("short_penalty", 0.95))
TOP_M = int(DEFAULTS.get("top_m", 200))

# --------------------------------------------------------------------------------------
# Lazy loading of FAISS indexes + embedding models
# --------------------------------------------------------------------------------------

_loaded = {}


def _load_faiss_and_model(key: str = "minilm_l6"):
    """Load FAISS index + sentence transformer for the requested model key."""
    idx_path = IDX_DIR / f"faiss_{key}.index"
    if not idx_path.exists():
        raise FileNotFoundError(f"FAISS index not found: {idx_path}")

    index = faiss.read_index(str(idx_path))
    model = SentenceTransformer(EMBED_MODELS[key])
    return index, model


# --------------------------------------------------------------------------------------
# Main search function
# --------------------------------------------------------------------------------------

def search_hybrid(
    query: str,
    k: int = 5,
    alpha: float | None = None,
    key: str = "minilm_l6",
    m: int | None = None,
    enable_house_bias: bool = False,  # kept for future extension
):
    """
    Hybrid-ish search: FAISS semantic search + title/entity tweaks.
    (Lexical TF-IDF/BM25 part is disabled in the Streamlit deployment because
    precomputed artifacts are not shipped in the repo.)

    Args:
        query: User question.
        k: How many results to return.
        alpha: Kept for config compatibility; not used in the current scoring.
        key: Embedding model key ('minilm_l6', 'e5_small', 'bge_base').
        m: How many FAISS candidates to pull before re-ranking.
           If None, uses max(k, TOP_M).
        enable_house_bias: Placeholder flag (currently unused).

    Returns:
        pandas.DataFrame with top-k hits and scoring metadata.
    """
    if alpha is None:
        alpha = float(ALPHAS.get(key, 0.6))

    if key not in _loaded:
        _loaded[key] = _load_faiss_and_model(key)
    index, embed_model = _loaded[key]

    # Decide how many neighbors to retrieve from FAISS
    if m is None:
        m = TOP_M
    m = int(max(m, k))

    # --- Semantic similarity via FAISS ---
    q_sem = _prep_query_text(key, query)
    qv = embed_model.encode([q_sem], normalize_embeddings=True).astype("float32")
    D, I = index.search(qv, m)
    I = I[0]
    sem = D[0].astype(np.float32)

    # Placeholder lexical score (since TF-IDF artifacts are not included)
    lex = np.zeros_like(sem, dtype=np.float32)

    # --- Title / entity bonus ---
    q_tokens = set(re.findall(r"\w+", query.lower()))
    titles = (
        chunks.iloc[I]["title"]
        .fillna("")
        .str.lower()
        .str.findall(r"\w+")
        .apply(set)
    )
    overlaps = titles.apply(lambda s: len(q_tokens & s)).to_numpy()
    title_bonus = (
        (overlaps > 0).astype(float) * BONUS1
        + (overlaps > 1).astype(float) * BONUS2
    )
    entity_boost = np.where(overlaps > 0, ENTITY_BOOST, 1.0).astype(np.float32)

    # --- Short-length penalty ---
    short_mask = chunks.iloc[I]["n_words"].to_numpy() < SHORT_PENALTY_LEN
    penalties = np.where(short_mask, SHORT_PENALTY, 1.0).astype(np.float32)

    # --- Final score: semantic + tweaks (no lexical part in cloud) ---
    s_sem = _minmax(sem)
    final = (s_sem + title_bonus) * penalties * entity_boost

    hits = chunks.iloc[I].copy()
    hits["sem"] = sem
    hits["lex"] = lex  # kept for compatibility
    hits["title_bonus"] = title_bonus
    hits["penalty"] = penalties
    hits["entity_boost"] = entity_boost
    hits["final_score"] = final

    return (
        hits.sort_values("final_score", ascending=False)
        .head(k)
        .reset_index(drop=True)
    )
