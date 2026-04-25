##############################################
# Helper 1: Ensure qdrant-server is up&ready #
##############################################
def ensure_qdrant_server_ready(log_prefix):
    import os
    import requests
    import time
    from config.qdrant_server import QDRANT_EXTERNAL_PROXY_PORT

    qdrant_ready_url = f"{os.environ['QDRANT_URL'].rstrip('/')}:{QDRANT_EXTERNAL_PROXY_PORT}/readyz"
    start_time = time.time()
    print(f"{log_prefix}: checking Qdrant server readiness at {qdrant_ready_url}", flush=True)
    try:
        response = requests.get(
            qdrant_ready_url,
            headers={"api-key": os.environ["QDRANT_API_KEY"]},
            timeout=300,
        )
        elapsed = time.time() - start_time
        print(
            f"{log_prefix}: Qdrant readiness response {response.status_code} "
            f"after {elapsed:.1f}s",
            flush=True,
        )
        response.raise_for_status()
        print(f"{log_prefix}: Qdrant server is ready", flush=True)
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print(
            f"{log_prefix}: failed to reach Qdrant server at {qdrant_ready_url} "
            f"after {elapsed:.1f}s: {e}",
            flush=True,
        )
        return False

################################################################
# Helper 2: Copy local qdrant-server storage to network volume #
################################################################
def persist_qdrant_storage(log_prefix):
    import os
    import requests
    import time
    from config.qdrant_server import QDRANT_EXTERNAL_PROXY_PORT

    qdrant_persist_url = f"{os.environ['QDRANT_URL'].rstrip('/')}:{QDRANT_EXTERNAL_PROXY_PORT}/persist"
    start_time = time.time()
    print(f"{log_prefix}: requesting Qdrant storage persistence at {qdrant_persist_url}", flush=True)
    try:
        response = requests.post(
            qdrant_persist_url,
            headers={"api-key": os.environ["QDRANT_API_KEY"]},
            timeout=600,
        )
        elapsed = time.time() - start_time
        print(
            f"{log_prefix}: Qdrant persistence response {response.status_code} "
            f"after {elapsed:.1f}s",
            flush=True,
        )
        response.raise_for_status()
        print(f"{log_prefix}: Qdrant storage persisted", flush=True)
        return True
    except Exception as e:
        elapsed = time.time() - start_time
        print(
            f"{log_prefix}: failed to persist Qdrant storage at {qdrant_persist_url} "
            f"after {elapsed:.1f}s: {e}",
            flush=True,
        )
        return False
