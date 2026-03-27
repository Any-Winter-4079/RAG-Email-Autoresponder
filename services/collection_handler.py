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
    from config.encoder import EMBEDDING_ENCODERS
    from config.general import QDRANT_PATH
    from qdrant_client import QdrantClient, models

    rag_volume.reload()
    client = QdrantClient(path=QDRANT_PATH)
    for variant in variants:
        if not recreate and client.collection_exists(collection_name=variant):
            continue
        vectors_config = {}
        sparse_vectors_config = {}
        for encoder, encoder_config in EMBEDDING_ENCODERS.items():
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

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def dump_collection_payloads(collection_name, include_vectors=False, page_size=1024):
    from config.general import QDRANT_PATH
    from qdrant_client import QdrantClient

    rag_volume.reload()
    client = QdrantClient(path=QDRANT_PATH)
    if not client.collection_exists(collection_name=collection_name):
        raise ValueError(
            "dump_collection_payloads: collection does not exist:\n"
            f"\t{collection_name}"
        )

    dumped_points = []
    next_offset = None
    while True:
        points, next_offset = client.scroll(
            collection_name=collection_name,
            offset=next_offset,
            limit=page_size,
            with_payload=True,
            with_vectors=include_vectors,
        )
        for point in points:
            dumped_point = {
                "id": point.id,
                "payload": point.payload or {},
            }
            if include_vectors:
                dumped_point["vector"] = point.vector
            dumped_points.append(dumped_point)
        if next_offset is None:
            break

    print(
        "dump_collection_payloads: dumped collection:\n"
        f"\tcollection: {collection_name}\n"
        f"\tinclude_vectors: {include_vectors}\n"
        f"\tpage_size: {page_size}\n"
        f"\tn points: {len(dumped_points)}"
    )
    return dumped_points

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def write_batch_points(variant, batch, encoder, embeddings, upsert_or_update):
    from config.encoder import EMBEDDING_ENCODERS
    from config.general import QDRANT_PATH
    from qdrant_client import QdrantClient, models

    point_ids = batch["point_ids"]
    payloads = batch["payloads"]
    encoder_config = EMBEDDING_ENCODERS[encoder]
    fastembed_kind = encoder_config["fastembed_kind"]

    rag_volume.reload()
    client = QdrantClient(path=QDRANT_PATH)

    points = []
    for point_id, payload, embedding in zip(point_ids, payloads, embeddings):
        if fastembed_kind == "sparse":
            vectors = {
                encoder: models.SparseVector(
                    indices=embedding.indices,
                    values=embedding.values,
                )
            }
        else:
            vectors = {encoder: embedding}

        if upsert_or_update == "upsert":
            points.append(
                models.PointStruct(
                    id=point_id,
                    payload=payload,
                    vector=vectors,
                )
            )
        elif upsert_or_update == "update":
            points.append(
                models.PointVectors(
                    id=point_id,
                    vector=vectors,
                )
            )
        else:
            raise ValueError(f"write_batch_points: unsupported upsert_or_update '{upsert_or_update}'")

    if upsert_or_update == "upsert":
        client.upsert(collection_name=variant, points=points)
        print(f"write_batch_points: {variant}: encoder '{encoder}': upserted {len(points)} points")
    else:
        client.update_vectors(collection_name=variant, points=points)
        print(f"write_batch_points: {variant}: encoder '{encoder}': updated {len(points)} points")

    rag_volume.commit()
