from config.modal_apps import ENCODER_CPU_APP_NAME, ENCODER_GPU_APP_NAME
from config.modal_functions import (
    RUN_ENCODER_CPU_BATCH_EMBEDDER_FUNCTION_NAME,
    RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
    RUN_ENCODER_GPU_RERANKER_FUNCTION_NAME,
)

# all embedding encoders
EMBEDDING_ENCODERS = {
    "bm25": {
        "model_name": "Qdrant/bm25",
        "service": ENCODER_CPU_APP_NAME,
        "function": RUN_ENCODER_CPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "fastembed_kind": "sparse",
        "modifier": "idf",
    },
    "splade": {
        "model_name": "prithivida/Splade_PP_en_v1",
        "service": ENCODER_GPU_APP_NAME,
        "function": RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "fastembed_kind": "sparse",
    },
    "colbert": {
        "model_name": "colbert-ir/colbertv2.0",
        "service": ENCODER_GPU_APP_NAME,
        "function": RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "fastembed_kind": "late",
        "vector_size": 128,
        "distance": "cosine",
        "max_recommended_input_size": 256,
    },
    "bge_small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "service": ENCODER_GPU_APP_NAME,
        "function": RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "fastembed_kind": "dense",
        "vector_size": 384,
        "distance": "cosine",
        "max_recommended_input_size": 512,
    },
    "bge_large": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "service": ENCODER_GPU_APP_NAME,
        "function": RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "fastembed_kind": "dense",
        "vector_size": 1024,
        "distance": "cosine",
        "max_recommended_input_size": 512,
    },
    "bge_m3": {
        "model_name": "BAAI/bge-m3",
        "service": ENCODER_GPU_APP_NAME,
        "function": RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "fastembed_kind": "dense",
        "embedding_backend": "flag_embedding",
        "vector_size": 1024,
        "distance": "cosine",
        "max_recommended_input_size": 8192,
        "use_fp16": True,
    },
    "jina_base_es": {
        "model_name": "jinaai/jina-embeddings-v2-base-es",
        "service": ENCODER_GPU_APP_NAME,
        "function": RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "fastembed_kind": "dense",
        "needs_custom_fastembed_registration": True,
        "vector_size": 768,
        "distance": "cosine",
        "max_recommended_input_size": 8192,
        "fastembed_pooling": "mean",
        "normalization": True,
        "model_file": "onnx/model.onnx",
    },
}

# all reranker encoders
RERANKER_ENCODERS = {
    "bge_reranker_v2_m3": {
        "model_name": "BAAI/bge-reranker-v2-m3",
        "service": ENCODER_GPU_APP_NAME,
        "function": RUN_ENCODER_GPU_RERANKER_FUNCTION_NAME,
        "backend": "flag_embedding",
        "use_fp16": True,
        "normalize": False,
    },
}
