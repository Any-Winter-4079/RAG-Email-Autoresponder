from config.m3 import IS_FINETUNED

RESULTS_DIR_NAME = "results"
QUERY_REWRITE_CACHE_DIR = "eval/cache/query_rewrites"
TOP_K_PER_QUERY = 15
TOP_K_AFTER_QUERY_FUSION = 10
TOP_K_AFTER_RRF = 5
CATEGORY_TO_MIN_FINAL_COUNT_AFTER_RRF = {"master": 2}
RERANKER_NAME = "bge_reranker_v2_m3"
DATA_VARIANTS_TO_FORCE_EMAIL_AS_QUERY = ["raw_chunks", "manually_cleaned_chunks"]
DATA_VARIANTS_TO_FORCE_EMAIL_AS_RERANKER_QUERY = []
RUN_RRF = True
RRF_ENCODERS = [
    "splade",
    "jina_base_es",
    "bge_m3_muia" if IS_FINETUNED else "bge_m3",
]
DATA_VARIANT_TEST_SPLIT_NAME = "dev" if IS_FINETUNED else "train"
MAX_QUERY_REWRITE_ERROR_RATE = 0.05
DATA_VARIANT_N_EVAL_SAMPLES_PER_FOLDER_URI = 10
VALID_CONTEXT_EMAILS_MODES = {"without_context", "with_context", "all"}
DATA_VARIANT_CONTEXT_EMAILS_MODE = "all"

ORACLE_DISCRIMINATOR_VARIANT = "lm_summary_chunks"
ORACLE_DISCRIMINATOR_COLLECTION_DUMP_TIMESTAMP = None
ORACLE_DISCRIMINATOR_QUERY_REWRITE_CACHE_FILENAME = None
ORACLE_DISCRIMINATOR_MAX_SAMPLES_PER_RUN = 50
ORACLE_DISCRIMINATOR_DUMP_COLLECTION_PAYLOADS_SCRIPT_NAME = "run_dump_collection_payloads"

M3_EVAL_ENCODERS = {"bge_m3": {"batch_size": 64}}
if IS_FINETUNED:
    M3_EVAL_ENCODERS["bge_m3_muia"] = {"batch_size": 64}

DATA_VARIANT_TEST_EVAL_VARIANTS = {
    "raw_chunks": {
        "encoders": {
            # "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            # "colbert": {"batch_size": 64},
            # "bge_small": {"batch_size": 64},
            # "bge_large": {"batch_size": 64},
            **M3_EVAL_ENCODERS,
            "jina_base_es": {"batch_size": 64},
        },
    },
    "manually_cleaned_chunks": {
        "encoders": {
            # "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            # "colbert": {"batch_size": 64},
            # "bge_small": {"batch_size": 64},
            # "bge_large": {"batch_size": 64},
            **M3_EVAL_ENCODERS,
            "jina_base_es": {"batch_size": 64},
        },
    },
    "lm_cleaned_text_chunks": {
        "encoders": {
            # "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            # "colbert": {"batch_size": 64},
            # "bge_small": {"batch_size": 64},
            # "bge_large": {"batch_size": 64},
            **M3_EVAL_ENCODERS,
            "jina_base_es": {"batch_size": 64},
        },
    },
    "lm_summary_chunks": {
        "encoders": {
            # "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            # "colbert": {"batch_size": 64},
            # "bge_small": {"batch_size": 64},
            # "bge_large": {"batch_size": 64},
            **M3_EVAL_ENCODERS,
            "jina_base_es": {"batch_size": 64},
        },
    },
    "lm_q_and_a_chunks": {
        "encoders": {
            # "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            # "colbert": {"batch_size": 64},
            # "bge_small": {"batch_size": 64},
            # "bge_large": {"batch_size": 64},
            **M3_EVAL_ENCODERS,
            "jina_base_es": {"batch_size": 64},
        },
    },
    "lm_q_and_a_for_q_only_chunks": {
        "encoders": {
            # "bm25": {"batch_size": 64},
            "splade": {"batch_size": 64},
            # "colbert": {"batch_size": 64},
            # "bge_small": {"batch_size": 64},
            # "bge_large": {"batch_size": 64},
            **M3_EVAL_ENCODERS,
            "jina_base_es": {"batch_size": 64},
        },
    },
}
