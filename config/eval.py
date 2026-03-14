RESULTS_DIR_NAME = "results"
QUERY_REWRITE_CACHE_DIR = "eval/cache/query_rewrites"
TOP_K = 5
RUN_RRF = True
RRF_ENCODERS = ["bge_small", "splade"]
DATA_VARIANT_TEST_SPLIT_NAME = "dev"
MAX_QUERY_REWRITE_FALLBACK_RATE = 0.05
DATA_VARIANT_N_EVAL_SAMPLES_PER_FOLDER_URI = 10
DATA_VARIANT_SKIP_CONTEXT_EMAILS = True

DATA_VARIANT_TEST_EVAL_VARIANTS = {
    "raw_chunks": {
        "encoders": {
            "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            # "colbert": {"batch_size": 64},
            "bge_small": {"batch_size": 64},
        },
    },
    "manually_cleaned_chunks": {
        "encoders": {
            "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            # "colbert": {"batch_size": 64},
            "bge_small": {"batch_size": 64},
        },
    },
    "lm_cleaned_text_chunks": {
        "encoders": {
            "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            # "colbert": {"batch_size": 64},
            "bge_small": {"batch_size": 64},
        },
    },
    "lm_summary_chunks": {
        "encoders": {
            "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            # "colbert": {"batch_size": 64},
            "bge_small": {"batch_size": 64},
        },
    },
    "lm_q_and_a_chunks": {
        "encoders": {
            "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            # "colbert": {"batch_size": 64},
            "bge_small": {"batch_size": 64},
        },
    },
    "lm_q_and_a_for_q_only_chunks": {
        "encoders": {
            "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            # "colbert": {"batch_size": 64},
            "bge_small": {"batch_size": 64},
        },
    },
}
