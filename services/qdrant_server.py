from config.general import modal_secret, rag_volume, VOLUME_PATH
from config.modal_apps import QDRANT_SERVER_APP_NAME
from config.qdrant_server import (
    image,
    MODAL_TIMEOUT,
    SCALEDOWN_WINDOW,
    MIN_CONTAINERS,
    MAX_CONTAINERS,
    MAX_INPUTS,
    QDRANT_INTERNAL_PROXY_PORT,
    QDRANT_STARTUP_TIMEOUT,
    QDRANT_WEB_LABEL,
)
import modal

# Modal
app = modal.App(QDRANT_SERVER_APP_NAME)

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    max_containers=MAX_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
@modal.concurrent(max_inputs=MAX_INPUTS)
@modal.web_server(
    QDRANT_INTERNAL_PROXY_PORT,
    startup_timeout=QDRANT_STARTUP_TIMEOUT,
    label=QDRANT_WEB_LABEL,
)
def serve_qdrant_server():
    import os
    import shutil
    import subprocess
    import time
    import urllib.error
    import urllib.request
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from urllib.parse import urlsplit
    import threading

    from config.qdrant_server import (
        QDRANT_SERVER_INTERNAL_HTTP_PORT,
        QDRANT_SERVER_INTERNAL_GRPC_PORT,
        QDRANT_INTERNAL_PROXY_PORT,
        QDRANT_RUNTIME_STORAGE_PATH,
        QDRANT_STARTUP_TIMEOUT,
        QDRANT_VOLUME_STORAGE_PATH,
    )

    api_key = os.getenv("QDRANT_API_KEY")
    if not api_key:
        raise ValueError("serve_qdrant_server: missing QDRANT_API_KEY")

    rag_volume.reload()
    if os.path.exists(QDRANT_RUNTIME_STORAGE_PATH):
        shutil.rmtree(QDRANT_RUNTIME_STORAGE_PATH)
    os.makedirs(QDRANT_RUNTIME_STORAGE_PATH, exist_ok=True)

    if os.path.exists(QDRANT_VOLUME_STORAGE_PATH):
        shutil.copytree(
            QDRANT_VOLUME_STORAGE_PATH,
            QDRANT_RUNTIME_STORAGE_PATH,
            dirs_exist_ok=True,
        )
        print(
            "serve_qdrant_server: copied Qdrant storage from Modal volume:\n"
            f"\tfrom: {QDRANT_VOLUME_STORAGE_PATH}\n"
            f"\tto: {QDRANT_RUNTIME_STORAGE_PATH}",
            flush=True,
        )
    else:
        print(
            "serve_qdrant_server: no Qdrant server storage found on Modal volume, "
            "starting with empty storage",
            flush=True,
        )

    env = os.environ.copy()
    env["QDRANT__SERVICE__API_KEY"] = api_key
    env["QDRANT__SERVICE__HOST"] = "127.0.0.1"
    env["QDRANT__SERVICE__HTTP_PORT"] = str(QDRANT_SERVER_INTERNAL_HTTP_PORT)
    env["QDRANT__SERVICE__GRPC_PORT"] = str(QDRANT_SERVER_INTERNAL_GRPC_PORT)
    env["QDRANT__STORAGE__STORAGE_PATH"] = QDRANT_RUNTIME_STORAGE_PATH
    env["QDRANT__TELEMETRY_DISABLED"] = "true"

    subprocess.Popen(["qdrant"], env=env)

    qdrant_url = f"http://127.0.0.1:{QDRANT_SERVER_INTERNAL_HTTP_PORT}"
    deadline = time.time() + QDRANT_STARTUP_TIMEOUT
    while time.time() < deadline:
        try:
            request = urllib.request.Request(
                f"{qdrant_url}/readyz",
                headers={"api-key": api_key},
            )
            with urllib.request.urlopen(request, timeout=5) as response:
                if response.status == 200:
                    break
        except Exception:
            time.sleep(1)
    else:
        raise TimeoutError("serve_qdrant_server: Qdrant did not become ready")
    print(f"serve_qdrant_server: Qdrant ready at {qdrant_url}/readyz", flush=True)

    def persist_storage():
        print("serve_qdrant_server: persisting Qdrant storage", flush=True)
        staging_path = f"{QDRANT_VOLUME_STORAGE_PATH}.tmp"
        if os.path.exists(staging_path):
            shutil.rmtree(staging_path)
        os.makedirs(os.path.dirname(QDRANT_VOLUME_STORAGE_PATH), exist_ok=True)
        shutil.copytree(QDRANT_RUNTIME_STORAGE_PATH, staging_path)
        if os.path.exists(QDRANT_VOLUME_STORAGE_PATH):
            shutil.rmtree(QDRANT_VOLUME_STORAGE_PATH)
        os.rename(staging_path, QDRANT_VOLUME_STORAGE_PATH)
        rag_volume.commit()
        print("serve_qdrant_server: Qdrant storage committed to Modal volume", flush=True)

    class QdrantProxyHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            self._handle_request()

        do_POST = do_GET
        do_PUT = do_GET
        do_PATCH = do_GET
        do_DELETE = do_GET

        def log_message(self, format, *args):
            return

        def _send_text(self, status, text):
            body = text.encode("utf-8")
            self.send_response(status)
            self.send_header("content-type", "text/plain; charset=utf-8")
            self.send_header("content-length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _handle_request(self):
            print(
                f"serve_qdrant_server: proxy received {self.command} {self.path}",
                flush=True,
            )
            if urlsplit(self.path).path == "/persist":
                if self.headers.get("api-key") != api_key:
                    print("serve_qdrant_server: rejected /persist request", flush=True)
                    self._send_text(401, "missing or invalid api-key\n")
                    return
                try:
                    persist_storage()
                    self._send_text(200, "qdrant storage persisted\n")
                except Exception as e:
                    print(f"serve_qdrant_server: failed to persist Qdrant storage: {e}", flush=True)
                    self._send_text(500, f"failed to persist qdrant storage: {e}\n")
                return

            start_time = time.time()
            content_length = int(self.headers.get("content-length", "0"))
            body = self.rfile.read(content_length) if content_length else None
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in {"connection", "host", "transfer-encoding"}
            }
            request = urllib.request.Request(
                f"{qdrant_url}{self.path}",
                data=body,
                headers=headers,
                method=self.command,
            )
            try:
                with urllib.request.urlopen(request, timeout=300) as response:
                    elapsed = time.time() - start_time
                    print(
                        f"serve_qdrant_server: proxied {self.command} {self.path} "
                        f"with status {response.status} after {elapsed:.1f}s",
                        flush=True,
                    )
                    self.send_response(response.status)
                    for key, value in response.headers.items():
                        if key.lower() not in {"connection", "transfer-encoding"}:
                            self.send_header(key, value)
                    self.end_headers()
                    self.wfile.write(response.read())
            except urllib.error.HTTPError as e:
                elapsed = time.time() - start_time
                print(
                    f"serve_qdrant_server: proxied {self.command} {self.path} "
                    f"with status {e.code} after {elapsed:.1f}s",
                    flush=True,
                )
                self.send_response(e.code)
                for key, value in e.headers.items():
                    if key.lower() not in {"connection", "transfer-encoding"}:
                        self.send_header(key, value)
                self.end_headers()
                self.wfile.write(e.read())
            except Exception as e:
                elapsed = time.time() - start_time
                print(
                    f"serve_qdrant_server: failed to proxy {self.command} {self.path} "
                    f"after {elapsed:.1f}s: {e}",
                    flush=True,
                )
                self._send_text(502, f"failed to proxy qdrant request: {e}\n")

    proxy_server = ThreadingHTTPServer(("0.0.0.0", QDRANT_INTERNAL_PROXY_PORT), QdrantProxyHandler)
    print(
        f"serve_qdrant_server: proxy listening on 0.0.0.0:{QDRANT_INTERNAL_PROXY_PORT}",
        flush=True,
    )
    threading.Thread(
        target=proxy_server.serve_forever,
        daemon=True,
    ).start()
