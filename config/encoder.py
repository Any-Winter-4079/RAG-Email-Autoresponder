from config.modal_apps import ENCODER_CPU_UPSERTER_OR_UPDATER_APP_NAME, ENCODER_GPU_UPSERTER_OR_UPDATER_APP_NAME
from config.modal_functions import RUN_ENCODER_CPU_UPSERTER_OR_UPDATER_FUNCTION_NAME, RUN_ENCODER_GPU_UPSERTER_OR_UPDATER_FUNCTION_NAME

# all encoders
ENCODERS = {
    "bm25": {
        "model_name": "Qdrant/bm25",
        "service": ENCODER_CPU_UPSERTER_OR_UPDATER_APP_NAME,
        "function": RUN_ENCODER_CPU_UPSERTER_OR_UPDATER_FUNCTION_NAME,
        "fastembed_kind": "sparse",
        "modifier": "idf",
    },
    "splade": {
        "model_name": "prithivida/Splade_PP_en_v1",
        "service": ENCODER_GPU_UPSERTER_OR_UPDATER_APP_NAME,
        "function": RUN_ENCODER_GPU_UPSERTER_OR_UPDATER_FUNCTION_NAME,
        "fastembed_kind": "sparse",
    },
    "colbert": {
        "model_name": "colbert-ir/colbertv2.0",
        "service": ENCODER_GPU_UPSERTER_OR_UPDATER_APP_NAME,
        "function": RUN_ENCODER_GPU_UPSERTER_OR_UPDATER_FUNCTION_NAME,
        "fastembed_kind": "late",
        "vector_size": 128,
        "distance": "cosine",
        "max_recommended_input_size": 256,
    },
    "bge_small": {
        "model_name": "BAAI/bge-small-en-v1.5",
        "service": ENCODER_GPU_UPSERTER_OR_UPDATER_APP_NAME,
        "function": RUN_ENCODER_GPU_UPSERTER_OR_UPDATER_FUNCTION_NAME,
        "fastembed_kind": "dense",
        "vector_size": 384,
        "distance": "cosine",
        "max_recommended_input_size": 8192,
    },
}
