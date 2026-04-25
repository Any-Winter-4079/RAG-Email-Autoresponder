from config.general import modal_secret
from config.modal_apps import COLLECTION_HANDLER_APP_NAME
from config.encoder_cpu import image
from config.collection_handler import (
    MODAL_TIMEOUT,
    SCALEDOWN_WINDOW,
    MIN_CONTAINERS,
    COLLECTION_HANDLER_QDRANT_CLIENT_TIMEOUT,
)
from config.qdrant_server import QDRANT_EXTERNAL_PROXY_PORT
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
)
def drop_legacy_collections():
    import os
    from qdrant_client import QdrantClient
    from config.general import LEGACY_COLLECTIONS

    client = QdrantClient(
        url=f"{os.environ['QDRANT_URL'].rstrip('/')}:{QDRANT_EXTERNAL_PROXY_PORT}",
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=COLLECTION_HANDLER_QDRANT_CLIENT_TIMEOUT,
    )
    for collection_name in LEGACY_COLLECTIONS:
        if client.collection_exists(collection_name=collection_name):
            client.delete_collection(collection_name=collection_name)
            print(f"drop_legacy_collections: deleted '{collection_name}'")
        else:
            print(f"drop_legacy_collections: '{collection_name}' does not exist, skipping")

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
)
def create_collections(variants, recreate):
    import os
    from config.collection_handler import COLLECTION_HNSW_CONFIG
    from config.encoder import EMBEDDING_ENCODERS
    from qdrant_client import QdrantClient, models

    client = QdrantClient(
        url=f"{os.environ['QDRANT_URL'].rstrip('/')}:{QDRANT_EXTERNAL_PROXY_PORT}",
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=COLLECTION_HANDLER_QDRANT_CLIENT_TIMEOUT,
    )
    hnsw_config = models.HnswConfigDiff(**COLLECTION_HNSW_CONFIG)
    for variant in variants:
        if not recreate and client.collection_exists(collection_name=variant):
            continue
        vectors_config = {}
        sparse_vectors_config = {}
        for encoder, encoder_config in EMBEDDING_ENCODERS.items():
            embedding_kinds = encoder_config["embedding_kinds"]
            for embedding_kind in embedding_kinds:
                vector_name = encoder if len(embedding_kinds) == 1 else f"{encoder}_{embedding_kind}"
                if embedding_kind == "sparse":
                    if encoder_config.get("modifier") == "idf":
                        sparse_vectors_config[vector_name] = models.SparseVectorParams(
                            modifier=models.Modifier.IDF,
                        )
                    else:
                        sparse_vectors_config[vector_name] = models.SparseVectorParams()
                elif embedding_kind in {"dense", "late"}:
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
                    vector_params = {
                        "size": encoder_config["vector_size"],
                        "distance": distance,
                    }
                    if embedding_kind == "late":
                        vector_params["multivector_config"] = models.MultiVectorConfig(
                            comparator=models.MultiVectorComparator.MAX_SIM
                        )
                    vectors_config[vector_name] = models.VectorParams(**vector_params)
                else:
                    raise ValueError(f"create_collections: unsupported embedding kind '{embedding_kind}' for encoder '{encoder}'")
        client.recreate_collection(
            collection_name=variant,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            hnsw_config=hnsw_config,
        )
        print(f"create_collections: recreated '{variant}'")

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
)
def dump_collection_payloads(collection_name, include_vectors=False, page_size=1024):
    import os
    from qdrant_client import QdrantClient

    client = QdrantClient(
        url=f"{os.environ['QDRANT_URL'].rstrip('/')}:{QDRANT_EXTERNAL_PROXY_PORT}",
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=COLLECTION_HANDLER_QDRANT_CLIENT_TIMEOUT,
    )
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
)
def write_batch_points(variant, batch, encoder, embeddings, upsert_or_update):
    import os
    from config.encoder import EMBEDDING_ENCODERS
    from qdrant_client import QdrantClient, models

    point_ids = batch["point_ids"]
    payloads = batch["payloads"]
    encoder_config = EMBEDDING_ENCODERS[encoder]
    embedding_kinds = encoder_config["embedding_kinds"]
    embedding_backend = encoder_config.get("embedding_backend", "fastembed")

    for embedding_kind in embedding_kinds:
        if embedding_kind not in embeddings:
            raise ValueError(
                f"write_batch_points: encoder '{encoder}' did not return "
                f"embeddings for embedding kind '{embedding_kind}'"
            )
        if len(embeddings[embedding_kind]) != len(point_ids):
            raise ValueError(
                f"write_batch_points: encoder '{encoder}' returned "
                f"{len(embeddings[embedding_kind])} embeddings for embedding kind "
                f"'{embedding_kind}', expected {len(point_ids)}"
            )

    client = QdrantClient(
        url=f"{os.environ['QDRANT_URL'].rstrip('/')}:{QDRANT_EXTERNAL_PROXY_PORT}",
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=COLLECTION_HANDLER_QDRANT_CLIENT_TIMEOUT,
    )

    points = []
    for point_index, (point_id, payload) in enumerate(zip(point_ids, payloads)):
        vectors = {}
        for embedding_kind in embedding_kinds:
            vector_name = encoder if len(embedding_kinds) == 1 else f"{encoder}_{embedding_kind}"
            embedding = embeddings[embedding_kind][point_index]
            if embedding_kind == "sparse":
                if embedding_backend == "flag_embedding":
                    vectors[vector_name] = models.SparseVector(
                        indices=[int(token_id) for token_id in embedding.keys()],
                        values=list(embedding.values()),
                    )
                elif embedding_backend == "fastembed":
                    vectors[vector_name] = models.SparseVector(
                        indices=embedding.indices,
                        values=embedding.values,
                    )
                else:
                    raise ValueError(
                        f"write_batch_points: embedding_backend '{embedding_backend}' "
                        f"does not support sparse embeddings"
                    )
            elif embedding_kind in {"dense", "late"}:
                vectors[vector_name] = embedding
            else:
                raise ValueError(
                    f"write_batch_points: unsupported embedding kind '{embedding_kind}' "
                    f"for encoder '{encoder}'"
                )

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
