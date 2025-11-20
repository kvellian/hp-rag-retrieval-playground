# Final retrieval configurations for the Harry Potter RAG project.

RETRIEVAL_PROFILES = {
    "fast_minilm": {
        "alpha": 0.75,
        "description": (
            "Fast profile: MiniLM-L6 with very low latency and strong hit@k. "
            "Tuned on NB8 qa_60_mixed (minilm_l6, α=0.75, k=16)."
        ),
        "k": 16,
        "model": "minilm_l6",
    },
    "max_precision_bge": {
        "alpha": 0.50,
        "description": (
            "Max-precision profile: bge-base tuned on NB8 qa_60_mixed for highest early precision; "
            "heavier model (bge_base, α=0.50, k=15)."
        ),
        "k": 15,
        "model": "bge_base",
    },
    "prod_e5_balanced": {
        "alpha": 0.55,
        "description": (
            "Production / balanced profile: good trade-off between accuracy and latency. "
            "Tuned on NB8 qa_60_mixed (e5_small, α=0.55, k=17)."
        ),
        "k": 17,
        "model": "e5_small",
    },
}

DEFAULT_PROFILE_NAME = "prod_e5_balanced"
