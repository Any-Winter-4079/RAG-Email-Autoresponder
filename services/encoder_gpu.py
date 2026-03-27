from config.general import modal_secret, rag_volume, VOLUME_PATH
from config.modal_apps import ENCODER_GPU_APP_NAME
from config.encoder_gpu import (
    image,
    GPU,
    MODAL_TIMEOUT,
    SCALEDOWN_WINDOW,
    MIN_CONTAINERS
)
import modal

# Modal
app = modal.App(ENCODER_GPU_APP_NAME)

@app.function(
    image=image,
    gpu=GPU,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def run_encoder_gpu_batch_embedder(variant, batch, encoder):
    from helpers.encoder import run_encoder_batch_embedder

    return run_encoder_batch_embedder(
        variant=variant,
        batch=batch,
        encoder=encoder,
        worker_name="run_encoder_gpu_batch_embedder",
    )

@app.function(
    image=image,
    gpu=GPU,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def run_encoder_gpu_retriever(query_texts, variant, encoder_name, top_k):
    from helpers.encoder import run_encoder_retriever

    return run_encoder_retriever(
        query_texts=query_texts,
        variant=variant,
        encoder_name=encoder_name,
        top_k=top_k,
        worker_name="run_encoder_gpu_retriever",
    )

@app.function(
    image=image,
    gpu=GPU,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def run_encoder_gpu_reranker(reranker_name, query_text, chunk_texts):
    from config.encoder import RERANKER_ENCODERS
    from FlagEmbedding import FlagReranker

    reranker_config = RERANKER_ENCODERS[reranker_name]
    if reranker_config.get("backend") != "flag_embedding":
        raise ValueError(
            f"run_encoder_gpu_reranker does not support backend "
            f"'{reranker_config.get('backend')}' for reranker '{reranker_name}'"
        )

    if not hasattr(run_encoder_gpu_reranker, "reranker_models"):
        run_encoder_gpu_reranker.reranker_models = {}

    model_name = reranker_config["model_name"]
    if model_name not in run_encoder_gpu_reranker.reranker_models:
        run_encoder_gpu_reranker.reranker_models[model_name] = FlagReranker(
            model_name,
            use_fp16=reranker_config.get("use_fp16", False),
        )

    if not chunk_texts:
        return []

    print(f"run_encoder_gpu_reranker: reranker '{reranker_name}': scoring {len(chunk_texts)} chunks")
    reranker_model = run_encoder_gpu_reranker.reranker_models[model_name]
    sentence_pairs = [[query_text, chunk_text] for chunk_text in chunk_texts]
    scores = reranker_model.compute_score(
        sentence_pairs,
        normalize=reranker_config.get("normalize", False),
    )
    if not isinstance(scores, list):
        scores = [scores]
    return scores
