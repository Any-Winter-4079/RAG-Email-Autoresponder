import modal

MAX_EMAILS = 0
# < 0 to keep all tokens
MAX_UNQUOTED_TOKENS_PER_CURRENT_EMAIL = 2500
MAX_UNQUOTED_TOKENS_PER_CONTEXT_EMAIL = 1500
MAX_QUOTED_TOKENS_PER_CURRENT_EMAIL = -1
MAX_QUOTED_TOKENS_PER_CONTEXT_EMAIL = 0
EMAIL_WRITER_THREAD_CONTEXT_TOKEN_RATIO = 0.365
EMAIL_WRITER_RAG_CONTEXT_TOKEN_RATIO = 0.446
INBOX_FOLDER = "Inbox"
SENT_FOLDER = "Sent"
UNREAD_ONLY = True
LEAVE_UNREAD = False # NOTE: if True, replies may be rewritten if they are not read before next execution
LAST_N_DAYS = 120
SEND_TO_SELF = True
SAVE_AS_DRAFT = True
DRAFTS_FOLDER = "Drafts"

TOP_K_PER_QUERY = 15
TOP_K_AFTER_QUERY_FUSION = None
USE_MAX_SIMILARITY_QUERY_FUSION_BEFORE_RRF = True
TOP_K_AFTER_SOURCE_RRF = 5
TOP_K_AFTER_RERANK = 5
RERANKER_NAME = "bge_reranker_v2_m3"
BASE_DATA_VARIANT = "lm_cleaned_text_chunks"
SOURCE_TO_ENCODER_SETTINGS = {
    "web": {
        "bm25": {"batch_size": 64, "rrf_weight": 0.064},
        "splade": {"batch_size": 64, "rrf_weight": 0.026},
        "bge_m3_muia": {
            "batch_size": 64,
            "rrf_weights": {
                "sparse": 0.236,
                "dense": 0.577,
            },
        },
        "qwen3_embedding_0_6b": {"batch_size": 32, "rrf_weight": 0.029},
        "jina_v5_text_small": {"batch_size": 64, "rrf_weight": 0.068},
    },
    "email": {
        "bm25": {"batch_size": 64, "rrf_weight": 0.064},
        "splade": {"batch_size": 64, "rrf_weight": 0.026},
        "bge_m3_muia": {
            "batch_size": 64,
            "rrf_weights": {
                "sparse": 0.236,
                "dense": 0.577,
            },
        },
        "qwen3_embedding_0_6b": {"batch_size": 32, "rrf_weight": 0.029},
        "jina_v5_text_small": {"batch_size": 64, "rrf_weight": 0.068},
    },
}

EMAIL_HOUR = 9
EMAIL_MINUTE = 0

PYTHON_VERSION = "3.11"

PACKAGES = [
    "transformers==5.5.0"
]

MODAL_TIMEOUT = 5400 # seconds

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*PACKAGES)
    .add_local_python_source("config", "helpers")
)
