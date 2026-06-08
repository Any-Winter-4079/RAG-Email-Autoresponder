from config.m3 import USE_FINETUNED_M3

RESULTS_DIR_NAME = "results"
QUERY_REWRITE_CACHE_DIR = "eval/cache/query_rewrites"
TO_FINE_TUNE = False

TOP_K_PER_QUERY = 15
TOP_K_AFTER_QUERY_FUSION = None
USE_MAX_SIMILARITY_QUERY_FUSION_BEFORE_RRF = True
TOP_K_AFTER_SOURCE_RRF = 5
TOP_K_AFTER_RRF = TOP_K_AFTER_SOURCE_RRF
CATEGORY_TO_MIN_FINAL_COUNT_AFTER_RRF = {"master": 2}

RERANKER_NAME = "bge_reranker_v2_m3"
TOP_K_AFTER_RERANK = 5

DATA_VARIANT_TEST_SPLIT_NAME = "train" if TO_FINE_TUNE else "dev"
MAX_QUERY_REWRITE_ERROR_RATE_THRES = 0.05
DATA_VARIANT_N_EVAL_SAMPLES_PER_FOLDER_URI = None

DATA_VARIANT_CONTEXT_EMAILS_MODE = "all" # "without_context", "with_context", "all"

DATA_VARIANT_EVAL_SOURCES = ["web", "email"] # ["web"], ["email"], ["web", "email"]

DUMP_COLLECTION_PAYLOADS_SIZE_COMPARISON_PLOT_FILENAME = (
    "dump_collection_size_comparison.png"
)
DUMP_COLLECTION_PAYLOADS_DECODER_TOKEN_DISTRIBUTION_PLOT_FILENAME = (
    "dump_collection_decoder_token_distribution.png"
)
DUMP_COLLECTION_PAYLOADS_ENCODER_TOKEN_DISTRIBUTION_PLOT_FILENAME = (
    "dump_collection_encoder_token_distribution.png"
)

if USE_FINETUNED_M3:
    M3_EVAL_ENCODER = {"bge_m3_muia": {"batch_size": 64, "rrf_weight": 1.0}}
else:
    M3_EVAL_ENCODER = {"bge_m3": {"batch_size": 64, "rrf_weight": 1.0}}

COMMON_EVAL_ENCODERS = {
    "bm25": {"batch_size": 64, "rrf_weight": 1.0},
    "splade": {"batch_size": 64, "rrf_weight": 1.0},
    # "colbert": {"batch_size": 64, "rrf_weight": 1.0},
    # "bge_small": {"batch_size": 64, "rrf_weight": 1.0},
    # "bge_large": {"batch_size": 64, "rrf_weight": 1.0},
    **M3_EVAL_ENCODER,
    "qwen3_embedding_0_6b": {"batch_size": 32, "rrf_weight": 1.0},
    "jina_v5_text_small": {"batch_size": 64, "rrf_weight": 1.0},
    # "jina_base_es": {"batch_size": 64, "rrf_weight": 1.0},
}

DATA_VARIANT_TEST_EVAL_VARIANTS = {
    # "raw_chunks": {
    #     "encoders": COMMON_EVAL_ENCODERS,
    # },
    # "manually_cleaned_chunks": {
    #     "encoders": COMMON_EVAL_ENCODERS,
    # },
    "lm_cleaned_text_chunks": {
        "encoders": COMMON_EVAL_ENCODERS,
    },
    # "lm_summary_chunks": {
    #     "encoders": COMMON_EVAL_ENCODERS,
    # },
    # "lm_q_and_a_chunks": {
    #     "encoders": COMMON_EVAL_ENCODERS,
    # },
    # "lm_q_and_a_for_q_only_chunks": {
    #     "encoders": COMMON_EVAL_ENCODERS,
    # },
}
