from config.general import modal_secret, rag_volume, VOLUME_PATH
from config.modal_apps import ENCODER_GPU_UPSERTER_OR_UPDATER_APP_NAME
from config.encoder_gpu import (
    image,
    GPU,
    MODAL_TIMEOUT,
    SCALEDOWN_WINDOW,
    MIN_CONTAINERS
)
import modal

# Modal
app = modal.App(ENCODER_GPU_UPSERTER_OR_UPDATER_APP_NAME)

@app.function(
    image=image,
    gpu=GPU,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def run_encoder_gpu_upserter_or_updater(variant, timestamp, start_index, batch_size, encoder, upsert_or_update="upsert"):
    from helpers.encoder import run_encoder_upserter_or_updater

    # run upserter or updater on GPU
    run_encoder_upserter_or_updater(
        variant=variant,
        timestamp=timestamp,
        start_index=start_index,
        batch_size=batch_size,
        encoder=encoder,
        upsert_or_update=upsert_or_update,
        worker_name="run_encoder_gpu_upserter_or_updater",
    )
