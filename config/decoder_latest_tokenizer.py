import modal

PYTHON_VERSION = "3.11"
SCALEDOWN_WINDOW = 60 # seconds
MODAL_TIMEOUT = 1800 # seconds
MIN_CONTAINERS = 0 # 0 to make sure we don't pay 24/7

PACKAGES = [
    "transformers==5.5.0"
]

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .pip_install(*PACKAGES)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .add_local_python_source("config", "helpers")
)
