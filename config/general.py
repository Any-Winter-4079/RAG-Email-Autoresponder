import modal

SECRET_NAME = "MUIA-SECRET"

modal_secret = modal.Secret.from_name(SECRET_NAME)

VOLUME_NAME = "muia-rag-volume"
VOLUME_PATH = "/root/volume"

rag_volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

COMMON_PACKAGES = [
    "torchvision",
    "accelerate",
    "Pillow",
    "requests",
    "hf_xet",
]
FLASH_ATTENTION_RELEASE = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
FLASH_ATTENTION_IMAGE = "anywinter4079/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04-runpod-clone"
FLASH_ATTENTION_RUN_COMMANDS = ("python -m pip install --upgrade pip && "
                                "pip config set global.extra-index-url https://download.pytorch.org/whl/cu128")
FLASH_ATTENTION_TORCH_VERSION = "torch==2.8.0+cu128"
USE_FLASH_ATTENTION_IMAGE = True
NO_FLASH_ATTENTION_PYTHON_VERSION = "3.11"
NO_FLASH_ATTENTION_TORCH_VERSION = "torch==2.8.0"

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
