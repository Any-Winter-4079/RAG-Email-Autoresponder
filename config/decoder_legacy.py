import modal
from config.decoder import (
    COMMON_PACKAGES,
    FLASH_ATTENTION_RELEASE,
    FLASH_ATTENTION_IMAGE,
    FLASH_ATTENTION_RUN_COMMANDS,
    FLASH_ATTENTION_TORCH_VERSION,
    USE_FLASH_ATTENTION_IMAGE,
    NO_FLASH_ATTENTION_PYTHON_VERSION,
    NO_FLASH_ATTENTION_TORCH_VERSION,
)

GPU = "L40S"
TIMEOUT = 900 # seconds

_image_flash_attention_base = (
    modal.Image.from_registry(FLASH_ATTENTION_IMAGE)
    .run_commands(FLASH_ATTENTION_RUN_COMMANDS)
    .pip_install(
        FLASH_ATTENTION_TORCH_VERSION,
        "transformers==4.57.0",
        "peft==0.17.1",
        *COMMON_PACKAGES
        )
    .pip_install(FLASH_ATTENTION_RELEASE)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
_image_no_flash_attention_base = (
    modal.Image.debian_slim(python_version=NO_FLASH_ATTENTION_PYTHON_VERSION)
    .pip_install(
        NO_FLASH_ATTENTION_TORCH_VERSION,
        "transformers==4.57.0",
        "peft==0.17.1",
        *COMMON_PACKAGES
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
base_image = _image_flash_attention_base if USE_FLASH_ATTENTION_IMAGE else _image_no_flash_attention_base

# Run `image.add_local_*` commands last in your image build to avoid rebuilding images with every local file change.
image = base_image.add_local_python_source("config", "helpers")
