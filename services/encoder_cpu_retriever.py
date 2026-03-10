from config.general import modal_secret, rag_volume, VOLUME_PATH
from config.modal_apps import ENCODER_CPU_RETRIEVER_APP_NAME
from config.encoder_cpu import (
    image,
    MODAL_TIMEOUT,
    SCALEDOWN_WINDOW,
    MIN_CONTAINERS
)
import modal

# Modal
app = modal.App(ENCODER_CPU_RETRIEVER_APP_NAME)

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def run_encoder_cpu_retriever(query_texts, variant, encoder_name, top_k):
    from helpers.encoder import run_encoder_retriever

    return run_encoder_retriever(
        query_texts=query_texts,
        variant=variant,
        encoder_name=encoder_name,
        top_k=top_k,
        worker_name="run_encoder_cpu_retriever",
    )
