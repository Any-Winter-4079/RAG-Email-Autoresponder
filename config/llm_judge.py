import modal

ORACLE_DISCRIMINATOR_VARIANT = "lm_cleaned_text_chunks"

# data variants are:
# raw_chunks
# manually_cleaned_chunks
# lm_cleaned_text_chunks
# lm_summary_chunks
# lm_q_and_a_chunks
# lm_q_and_a_for_q_only_chunks

ORACLE_DISCRIMINATOR_INPUT_MODE = "retrieval" # "corpus", "retrieval"
ORACLE_DISCRIMINATOR_RETRIEVAL_TIMESTAMP = "2026-06-30_11-37-23" # "2026-05-16_19-51-52" # for retrieval (else None)
ORACLE_DISCRIMINATOR_RETRIEVAL_OUTPUT_FILE_NAMES = [
    "bm25",
    "splade",
    "bge_m3_muia_sparse",
    "bge_m3_muia_dense",
    "qwen3_embedding_0_6b",
    "jina_v5_text_small"] # encoder names, rrf (from rrf.json) or reranker (from reranker.json)
ORACLE_DISCRIMINATOR_RETRIEVAL_OUTPUT_FILE_NAMES = ["reranker"]
ORACLE_DISCRIMINATOR_TOP_K_PER_RETRIEVAL_OUTPUT = 5

# encoder names are:
# bm25 (from bm25.json)
# splade (from splade.json)
# bge_m3_sparse (or bge_m3_muia_sparse) (from bge_m3_sparse.json/bge_m3_muia_sparse.json)
# bge_m3_dense (or bge_m3_muia_dense) (from bge_m3_dense.json/bge_m3_muia_dense.json)
# qwen3_embedding_0_6b (from qwen3_embedding_0_6b.json)
# jina_v5_text_small (from jina_v5_text_small.json)

# noting the disambiguation for m3 via _sparse, _dense

ORACLE_DISCRIMINATOR_DATA_SOURCES = ["web", "email"] # ["web"], ["email"], ["web", "email"]

ORACLE_DISCRIMINATOR_HUMAN_ALIGNMENT_MODE = False

ORACLE_DISCRIMINATOR_COLLECTION_DUMP_TIMESTAMP = None
ORACLE_DISCRIMINATOR_QUERY_REWRITE_CACHE_FILENAME = None

ORACLE_DISCRIMINATOR_N_EVAL_SAMPLES_PER_FOLDER_URI = None
ORACLE_DISCRIMINATOR_MAX_SAMPLES_PER_RUN = None

ORACLE_DISCRIMINATOR_DUMP_COLLECTION_PAYLOADS_SCRIPT_NAME = "run_dump_collection_payloads"

PYTHON_VERSION = "3.11"
SCALEDOWN_WINDOW = 60 # seconds
MODAL_TIMEOUT = 900 # seconds
MIN_CONTAINERS = 0 # 0 to make sure we don't pay 24/7

PACKAGES = [
    "openai",
]

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*PACKAGES)
    .add_local_python_source("config", "helpers")
)
