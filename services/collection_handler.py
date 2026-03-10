from config.general import modal_secret, rag_volume, VOLUME_PATH
from config.modal_apps import COLLECTION_HANDLER_APP_NAME
from config.encoder_cpu import (
    image,
    MODAL_TIMEOUT,
    SCALEDOWN_WINDOW,
    MIN_CONTAINERS
)
import modal

# modal run services/collection_handler.py::drop_legacy_collections

# Modal
app = modal.App(COLLECTION_HANDLER_APP_NAME)

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def drop_legacy_collections():
    from qdrant_client import QdrantClient
    from config.general import QDRANT_PATH, LEGACY_COLLECTIONS

    rag_volume.reload()
    client = QdrantClient(path=QDRANT_PATH)
    for collection_name in LEGACY_COLLECTIONS:
        if client.collection_exists(collection_name=collection_name):
            client.delete_collection(collection_name=collection_name)
            print(f"drop_legacy_collections: deleted '{collection_name}'")
        else:
            print(f"drop_legacy_collections: '{collection_name}' does not exist, skipping")
    rag_volume.commit()

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def create_collections(variants, recreate):
    from config.encoder import ENCODERS
    from config.general import QDRANT_PATH
    from qdrant_client import QdrantClient, models

    rag_volume.reload()
    client = QdrantClient(path=QDRANT_PATH)
    for variant in variants:
        if not recreate and client.collection_exists(collection_name=variant):
            continue
        vectors_config = {}
        sparse_vectors_config = {}
        for encoder, encoder_config in ENCODERS.items():
            fastembed_kind = encoder_config["fastembed_kind"]
            if fastembed_kind == "sparse":
                if encoder_config.get("modifier") == "idf":
                    sparse_vectors_config[encoder] = models.SparseVectorParams(
                        modifier=models.Modifier.IDF,
                    )
                else:
                    sparse_vectors_config[encoder] = models.SparseVectorParams()
            elif fastembed_kind == "dense":
                distance_str = encoder_config["distance"]
                if distance_str == "cosine":
                    distance = models.Distance.COSINE
                elif distance_str == "dot":
                    distance = models.Distance.DOT
                elif distance_str == "euclid":
                    distance = models.Distance.EUCLID
                elif distance_str == "manhattan":
                    distance = models.Distance.MANHATTAN
                else:
                    raise ValueError(f"create_collections: unsupported distance '{distance_str}' for encoder '{encoder}'")
                vectors_config[encoder] = models.VectorParams(
                    size=encoder_config["vector_size"],
                    distance=distance,
                )
            elif fastembed_kind == "late":
                distance_str = encoder_config["distance"]
                if distance_str == "cosine":
                    distance = models.Distance.COSINE
                elif distance_str == "dot":
                    distance = models.Distance.DOT
                elif distance_str == "euclid":
                    distance = models.Distance.EUCLID
                elif distance_str == "manhattan":
                    distance = models.Distance.MANHATTAN
                else:
                    raise ValueError(f"create_collections: unsupported distance '{distance_str}' for encoder '{encoder}'")
                vectors_config[encoder] = models.VectorParams(
                    size=encoder_config["vector_size"],
                    distance=distance,
                    multivector_config=models.MultiVectorConfig(
                        comparator=models.MultiVectorComparator.MAX_SIM
                    ),
                )
            else:
                raise ValueError(f"create_collections: unsupported fastembed_kind '{fastembed_kind}' for encoder '{encoder}'")
        client.recreate_collection(
            collection_name=variant,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
        )
        print(f"create_collections: recreated '{variant}'")
    rag_volume.commit()
