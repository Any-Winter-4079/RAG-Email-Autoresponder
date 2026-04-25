import modal
from config.general import VOLUME_PATH

PYTHON_VERSION = "3.11"
QDRANT_VERSION = "v1.17.0"
QDRANT_RELEASE_ARCHIVE = "qdrant-x86_64-unknown-linux-musl.tar.gz"

QDRANT_EXTERNAL_PROXY_PORT = 443
QDRANT_INTERNAL_PROXY_PORT = 6333
QDRANT_SERVER_INTERNAL_HTTP_PORT = 6334
QDRANT_SERVER_INTERNAL_GRPC_PORT = 6335

QDRANT_WEB_LABEL = "muia-qdrant-server"
QDRANT_STARTUP_TIMEOUT = 300 # seconds

QDRANT_RUNTIME_STORAGE_PATH = "/tmp/qdrant/storage"
QDRANT_VOLUME_STORAGE_PATH = f"{VOLUME_PATH}/qdrant_server_storage"

SCALEDOWN_WINDOW = 900 # seconds
MODAL_TIMEOUT = 3600 # seconds
MIN_CONTAINERS = 0
MAX_CONTAINERS = 1
MAX_INPUTS = 100

image = (
    modal.Image.debian_slim(python_version=PYTHON_VERSION)
    .apt_install("ca-certificates", "curl", "tar")
    .run_commands(
        "curl -L --fail "
        f"https://github.com/qdrant/qdrant/releases/download/{QDRANT_VERSION}/{QDRANT_RELEASE_ARCHIVE} "
        "-o /tmp/qdrant.tar.gz && "
        "tar -xzf /tmp/qdrant.tar.gz -C /usr/local/bin qdrant && "
        "chmod +x /usr/local/bin/qdrant && "
        "rm /tmp/qdrant.tar.gz"
    )
    .add_local_python_source("config")
)
