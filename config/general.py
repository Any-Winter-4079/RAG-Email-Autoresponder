import modal

SECRET_NAME = "MUIA-SECRET"

modal_secret = modal.Secret.from_name(SECRET_NAME)

VOLUME_NAME = "muia-rag-volume"
VOLUME_PATH = "/root/volume"
QDRANT_PATH = f"{VOLUME_PATH}/qdrant"

rag_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

LEGACY_COLLECTIONS = [
    "raw_chunks",
    "manually_cleaned_chunks",
    "lm_cleaned_text_chunks",
    "lm_summary_chunks",
    "lm_q_and_a_chunks",
    "lm_q_and_a_for_q_only_chunks",
]

LEGACY_VOLUME_FOLDERS = [
    "raw",
    "manually_cleaned",
    "raw_chunks",
    "manually_cleaned_chunks",
    "lm_cleaned_text_chunks",
    "lm_abstract_chunks",
    "lm_summary_chunks",
    "lm_q_and_a_chunks",
    "lm_cleaned_text_subchunks",
    "lm_summary_subchunks",
    "lm_q_and_a_valid_chunks",
    "lm_q_and_a_for_q_only_valid_chunks",
]
