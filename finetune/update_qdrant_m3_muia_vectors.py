import json
from pathlib import Path

import modal

from config.general import VOLUME_PATH, modal_secret, rag_volume
from config.m3 import (
    M3_REENCODING_MODAL_GPU,
    M3_REENCODING_MODAL_IMAGE,
    M3_REENCODING_MODAL_MIN_CONTAINERS,
    M3_REENCODING_MODAL_SCALEDOWN_WINDOW,
    M3_REENCODING_MODAL_TIMEOUT,
)

# modal run finetune/update_qdrant_m3_muia_vectors.py

app = modal.App("qdrant-m3-muia-vector-update")

target_collections = [
    "raw_chunks",
    "manually_cleaned_chunks",
    "lm_cleaned_text_chunks",
    "lm_summary_chunks",
    "lm_q_and_a_chunks",
    "lm_q_and_a_for_q_only_chunks",
    "email_lm_cleaned_text_chunks",
    "email_lm_summary_chunks",
    "email_lm_q_and_a_chunks",
    "email_lm_q_and_a_for_q_only_chunks",
]
encoder_name = "bge_m3_muia"
batch_size = 256
post_sft_suffix = "_post_sft"
summary_path = Path(__file__).resolve().parent / "qdrant_m3_muia_vector_update_summary.json"

@app.function(
    image=M3_REENCODING_MODAL_IMAGE,
    gpu=M3_REENCODING_MODAL_GPU,
    secrets=[modal_secret],
    timeout=M3_REENCODING_MODAL_TIMEOUT,
    scaledown_window=M3_REENCODING_MODAL_SCALEDOWN_WINDOW,
    min_containers=M3_REENCODING_MODAL_MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def update_qdrant_m3_muia_vectors(collection_names, page_size):
    import os
    from config.collection_handler import (
        COLLECTION_HANDLER_QDRANT_CLIENT_TIMEOUT,
        COLLECTION_INGESTION_HNSW_CONFIG,
        COLLECTION_INGESTION_OPTIMIZERS_CONFIG,
    )
    from config.encoder import EMBEDDING_ENCODERS
    from config.qdrant_server import QDRANT_EXTERNAL_PROXY_PORT
    from helpers.encoder import run_encoder_batch_document_embedder
    from helpers.general import get_text_from_payload
    from helpers.qdrant import ensure_qdrant_server_ready, persist_qdrant_storage
    from qdrant_client import QdrantClient, models

    def get_qdrant_vectors(embeddings, point_index):
        sparse_embedding = embeddings["sparse"][point_index]
        return {
            f"{encoder_name}_dense": embeddings["dense"][point_index],
            f"{encoder_name}_sparse": models.SparseVector(
                indices=[int(token_id) for token_id in sparse_embedding.keys()],
                values=list(sparse_embedding.values()),
            ),
        }

    def get_collection_vector_configs(collection_info):
        return {
            vector_name: models.VectorParams(
                size=vector_config.size,
                distance=vector_config.distance,
                multivector_config=vector_config.multivector_config,
            )
            for vector_name, vector_config in collection_info.config.params.vectors.items()
        }, {
            sparse_vector_name: models.SparseVectorParams(
                modifier=sparse_vector_config.modifier,
            )
            for sparse_vector_name, sparse_vector_config in collection_info.config.params.sparse_vectors.items()
        }

    def add_encoder_vector_configs(vectors_config, sparse_vectors_config, encoder_name):
        encoder_config = EMBEDDING_ENCODERS[encoder_name]
        for embedding_kind in encoder_config["embedding_kinds"]:
            vector_name = f"{encoder_name}_{embedding_kind}"
            if embedding_kind == "dense":
                vectors_config[vector_name] = models.VectorParams(
                    size=encoder_config["vector_size"],
                    distance=models.Distance.COSINE,
                )
            elif embedding_kind == "sparse":
                sparse_vectors_config[vector_name] = models.SparseVectorParams()
            else:
                raise ValueError(
                    "update_qdrant_m3_muia_vectors: unsupported embedding kind:\n"
                    f"\tencoder: {encoder_name}\n"
                    f"\tembedding kind: {embedding_kind}"
                )

    def create_post_sft_collection(client, collection_name, post_sft_collection_name):
        collection_info = client.get_collection(collection_name=collection_name)
        vectors_config, sparse_vectors_config = get_collection_vector_configs(collection_info)
        add_encoder_vector_configs(vectors_config, sparse_vectors_config, encoder_name)
        hnsw_config = models.HnswConfigDiff(**COLLECTION_INGESTION_HNSW_CONFIG)
        optimizers_config = models.OptimizersConfigDiff(**COLLECTION_INGESTION_OPTIMIZERS_CONFIG)
        client.recreate_collection(
            collection_name=post_sft_collection_name,
            vectors_config=vectors_config,
            sparse_vectors_config=sparse_vectors_config,
            hnsw_config=hnsw_config,
            optimizers_config=optimizers_config,
        )

    if not ensure_qdrant_server_ready("update_qdrant_m3_muia_vectors"):
        raise RuntimeError("update_qdrant_m3_muia_vectors: Qdrant server is not ready")

    client = QdrantClient(
        url=f"{os.environ['QDRANT_URL'].rstrip('/')}:{QDRANT_EXTERNAL_PROXY_PORT}",
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=COLLECTION_HANDLER_QDRANT_CLIENT_TIMEOUT,
    )
    summaries = []
    for collection_name in collection_names:
        post_sft_collection_name = f"{collection_name}{post_sft_suffix}"
        create_post_sft_collection(client, collection_name, post_sft_collection_name)
        next_offset = None
        n_points = 0
        while True:
            points, next_offset = client.scroll(
                collection_name=collection_name,
                offset=next_offset,
                limit=page_size,
                with_payload=True,
                with_vectors=True,
            )
            if not points:
                break
            texts = [
                get_text_from_payload(point.payload).strip()
                for point in points
            ]
            embeddings = run_encoder_batch_document_embedder(
                collection_name,
                {"texts": texts},
                encoder_name,
                "update_qdrant_m3_muia_vectors",
            )
            client.upsert(
                collection_name=post_sft_collection_name,
                points=[
                    models.PointStruct(
                        id=point.id,
                        payload=point.payload,
                        vector={
                            **point.vector,
                            **get_qdrant_vectors(embeddings, point_index),
                        },
                    )
                    for point_index, point in enumerate(points)
                ],
            )
            n_points += len(points)
            if next_offset is None:
                break

        sample_points, _ = client.scroll(
            collection_name=post_sft_collection_name,
            limit=1,
            with_payload=False,
            with_vectors=True,
        )
        vector_names = sorted(sample_points[0].vector)
        summaries.append({
            "source_collection": collection_name,
            "post_sft_collection": post_sft_collection_name,
            "n_points": n_points,
            "vector_names": vector_names,
        })
        print(
            "update_qdrant_m3_muia_vectors: updated collection:\n"
            f"\tsource collection: {collection_name}\n"
            f"\tpost-SFT collection: {post_sft_collection_name}\n"
            f"\tn points: {n_points}\n"
            f"\tvector names: {vector_names}"
        )
        if not persist_qdrant_storage("update_qdrant_m3_muia_vectors"):
            raise RuntimeError("update_qdrant_m3_muia_vectors: failed to persist Qdrant storage")

    return summaries

@app.local_entrypoint()
def main():
    summaries = update_qdrant_m3_muia_vectors.remote(target_collections, batch_size)
    with open(summary_path, "w", encoding="utf-8") as summary_file:
        json.dump(summaries, summary_file, ensure_ascii=False, indent=2)
    print(f"wrote update summary to {summary_path}")
