HAS_FINETUNED_M3 = True
USE_FINETUNED_M3 = True

from config.decoder_legacy import base_image as _base_image

M3_REENCODING_MODAL_GPU = "L40S"
M3_REENCODING_MODAL_SCALEDOWN_WINDOW = 60 # seconds
M3_REENCODING_MODAL_TIMEOUT = 5400 # seconds
M3_REENCODING_MODAL_MIN_CONTAINERS = 0 # 0 to make sure we don't pay 24/7

# Run `image.add_local_*` commands last in your image build to avoid rebuilding images with every local file change.
M3_REENCODING_MODAL_IMAGE = _base_image.pip_install(
    "FlagEmbedding",
    "llama-index",
    "fastembed-gpu",
    "qdrant-client>=1.14.2",
    "huggingface_hub",
    "sentence-transformers>=2.7.0",
).add_local_python_source("config", "helpers")
