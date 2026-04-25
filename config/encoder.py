from config.modal_apps import ENCODER_CPU_APP_NAME, ENCODER_GPU_APP_NAME
from config.modal_functions import (
    RUN_ENCODER_CPU_BATCH_EMBEDDER_FUNCTION_NAME,
    RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
    RUN_ENCODER_GPU_RERANKER_FUNCTION_NAME,
)

DEFAULT_RRF_ENCODER_WEIGHT = 1.0

# all embedding encoders
EMBEDDING_ENCODERS = {
    "bm25": {
        "model_name": "Qdrant/bm25",
        "service": ENCODER_CPU_APP_NAME,
        "function": RUN_ENCODER_CPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "embedding_kinds": ["sparse"],
        "modifier": "idf",
    },
    "splade": {
        "model_name": "prithivida/Splade_PP_en_v1",
        "service": ENCODER_CPU_APP_NAME,
        "function": RUN_ENCODER_CPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "embedding_kinds": ["sparse"],
    },
    "colbert": {
        "model_name": "colbert-ir/colbertv2.0",
        "service": ENCODER_GPU_APP_NAME,
        "function": RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "embedding_kinds": ["late"],
        "vector_size": 128,
        "distance": "cosine",
        "max_recommended_input_size": 256,
    },
    "bge_small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "service": ENCODER_GPU_APP_NAME,
        "function": RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "embedding_kinds": ["dense"],
        "vector_size": 384,
        "distance": "cosine",
        "max_recommended_input_size": 512,
    },
    "bge_large": {
        "model_name": "BAAI/bge-large-en-v1.5",
        "service": ENCODER_GPU_APP_NAME,
        "function": RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "embedding_kinds": ["dense"],
        "vector_size": 1024,
        "distance": "cosine",
        "max_recommended_input_size": 512,
    },
    "bge_m3": {
        "model_name": "BAAI/bge-m3",
        "service": ENCODER_GPU_APP_NAME,
        "function": RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "embedding_kinds": ["dense"],
        "embedding_backend": "flag_embedding",
        "vector_size": 1024,
        "distance": "cosine",
        "max_recommended_input_size": 8192,
        "use_fp16": True,
    },
    "bge_m3_muia": {
        "model_name": "Edue3r4t5y6/bge-m3-MUIA",
        "service": ENCODER_GPU_APP_NAME,
        "function": RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "embedding_kinds": ["dense"],
        "embedding_backend": "flag_embedding",
        "vector_size": 1024,
        "distance": "cosine",
        "max_recommended_input_size": 8192,
        "use_fp16": True,
    },
    "jina_v5_text_small": {
        "model_name": "jinaai/jina-embeddings-v5-text-small",
        "service": ENCODER_GPU_APP_NAME,
        "function": RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "embedding_kinds": ["dense"],
        "embedding_backend": "transformers",
        "vector_size": 1024,
        "distance": "cosine",
        "max_recommended_input_size": 32768,
    },
    "jina_base_es": {
        "model_name": "jinaai/jina-embeddings-v2-base-es",
        "service": ENCODER_GPU_APP_NAME,
        "function": RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "embedding_kinds": ["dense"],
        "needs_custom_fastembed_registration": True,
        "vector_size": 768,
        "distance": "cosine",
        "max_recommended_input_size": 8192,
        "fastembed_pooling": "mean",
        "normalization": True,
        "model_file": "onnx/model.onnx",
    },
    "qwen3_embedding_0_6b": {
        "model_name": "Qwen/Qwen3-Embedding-0.6B",
        "service": ENCODER_GPU_APP_NAME,
        "function": RUN_ENCODER_GPU_BATCH_EMBEDDER_FUNCTION_NAME,
        "embedding_kinds": ["dense"],
        "embedding_backend": "sentence_transformers",
        "vector_size": 1024,
        "distance": "cosine",
        "max_recommended_input_size": 32768,
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
