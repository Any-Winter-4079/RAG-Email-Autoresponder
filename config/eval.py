RESULTS_DIR_NAME = "results"
TRANSLATION_CACHE_DIR = "eval/cache/query_translations"
TOP_K = 5
DATA_VARIANT_TEST_SPLIT_NAME = "dev"
MAX_TRANSLATION_FALLBACK_RATE = 0.05

DATA_VARIANT_TEST_EVAL_VARIANTS = {
    "raw_chunks": {
        "encoders": {
            "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            "colbert": {"batch_size": 64},
            "bge_small": {"batch_size": 64},
        },
    },
    "manually_cleaned_chunks": {
        "encoders": {
            "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            "colbert": {"batch_size": 64},
            "bge_small": {"batch_size": 64},
        },
    },
    "lm_cleaned_text_chunks": {
        "encoders": {
            "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            "colbert": {"batch_size": 64},
            "bge_small": {"batch_size": 64},
        },
    },
    "lm_summary_chunks": {
        "encoders": {
            "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            "colbert": {"batch_size": 64},
            "bge_small": {"batch_size": 64},
        },
    },
    "lm_q_and_a_chunks": {
        "encoders": {
            "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            "colbert": {"batch_size": 64},
            "bge_small": {"batch_size": 64},
        },
    },
    "lm_q_and_a_for_q_only_chunks": {
        "encoders": {
            "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            "colbert": {"batch_size": 64},
            "bge_small": {"batch_size": 64},
        },
    },
}
