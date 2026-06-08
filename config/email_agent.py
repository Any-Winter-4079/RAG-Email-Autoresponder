import modal

MAX_EMAILS = 2
# < 0 to keep all tokens
MAX_UNQUOTED_TOKENS_PER_CURRENT_EMAIL = 2500
MAX_UNQUOTED_TOKENS_PER_CONTEXT_EMAIL = 1500
MAX_QUOTED_TOKENS_PER_CURRENT_EMAIL = -1
MAX_QUOTED_TOKENS_PER_CONTEXT_EMAIL = 0
EMAIL_WRITER_BODY_TOKEN_RATIO = 0.20
EMAIL_WRITER_THREAD_CONTEXT_TOKEN_RATIO = 0.35
EMAIL_WRITER_RAG_CONTEXT_TOKEN_RATIO = 0.45
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
TOP_K_AFTER_SOURCE_RRF = 5
TOP_K_AFTER_RERANK = 5
SOURCE_TO_RRF_ENCODER_WEIGHTS = {
    "web": {
        "splade": 1.0,
        "bge_m3": 1.0,
        "qwen3_embedding_0_6b": 1.0,
        "jina_v5_text_small": 1.0,
    },
    "email": {
        "splade": 1.0,
        "bge_m3": 1.0,
        "qwen3_embedding_0_6b": 1.0,
        "jina_v5_text_small": 1.0,
    },
}

EMAIL_HOUR = 9
EMAIL_MINUTE = 0

PYTHON_VERSION = "3.11"

PACKAGES = [
    "transformers==4.57.0"
]

MODAL_TIMEOUT = 600 # seconds

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*PACKAGES)
    .add_local_python_source("config", "helpers")
)
