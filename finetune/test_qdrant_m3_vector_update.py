import json
import sys
from pathlib import Path

import modal

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FlagEmbedding"))

from config.encoder_gpu import GPU, MIN_CONTAINERS, MODAL_TIMEOUT, SCALEDOWN_WINDOW, image
from config.general import VOLUME_PATH, modal_secret, rag_volume
from helpers.general import get_text_from_payload, resolve_oracle_discriminator_path

# modal run finetune/test_qdrant_m3_vector_update.py

app = modal.App("qdrant-m3-vector-update-test")
collection_name = "dummy_m3_vector_update"
post_sft_collection_name = f"{collection_name}_post_sft"
base_encoder_name = "bge_m3"
tuned_encoder_name = "bge_m3_muia"
n_points = 3
split_name = "dev"
data_variant = "lm_summary_chunks"
oracle_timestamps = [
    "2026-05-14_19-31-38",
    "2026-05-15_01-52-23",
]
dump_path = project_root / "finetune" / "qdrant_m3_vector_update_dummy_dump.json"

def load_texts():
    texts = []
    seen_texts = set()
    for oracle_timestamp in oracle_timestamps:
        oracle_path = resolve_oracle_discriminator_path(
            project_root=project_root,
            split_name=split_name,
            variant=data_variant,
            timestamp=oracle_timestamp,
        )
        with open(oracle_path, "r", encoding="utf-8") as oracle_file:
            oracle_output = json.load(oracle_file)
        for result in oracle_output["results"]:
            for subquery in result["discriminator_result"].get("subqueries") or []:
                for chunk in subquery["supporting_chunks"] + subquery["insufficient_chunks"]:
                    if chunk["payload"] is None:
                        continue
                    text = get_text_from_payload(chunk["payload"]).strip()
                    if text and text not in seen_texts:
                        texts.append(text)
                        seen_texts.add(text)
                    if len(texts) == n_points:
                        return texts
    return texts

@app.function(
    image=image,
    gpu=GPU,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def test_qdrant_m3_vector_update(texts):
    import os
    from config.collection_handler import COLLECTION_HANDLER_QDRANT_CLIENT_TIMEOUT
    from config.encoder import EMBEDDING_ENCODERS
    from config.qdrant_server import QDRANT_EXTERNAL_PROXY_PORT
    from helpers.encoder import run_encoder_batch_document_embedder
    from qdrant_client import QdrantClient, models

    def get_vector_configs(encoder_name):
        encoder_config = EMBEDDING_ENCODERS[encoder_name]
        return {
            f"{encoder_name}_dense": models.VectorParams(
                size=encoder_config["vector_size"],
                distance=models.Distance.COSINE,
            )
        }, {
            f"{encoder_name}_sparse": models.SparseVectorParams()
        }

    def get_qdrant_vectors(encoder_name, embeddings, point_index):
        sparse_embedding = embeddings["sparse"][point_index]
        return {
            f"{encoder_name}_dense": embeddings["dense"][point_index],
            f"{encoder_name}_sparse": models.SparseVector(
                indices=[int(token_id) for token_id in sparse_embedding.keys()],
                values=list(sparse_embedding.values()),
            ),
        }

    def get_vector_configs_for_encoders(encoder_names):
        vectors_config = {}
        sparse_vectors_config = {}
        for encoder_name in encoder_names:
            dense_config, sparse_config = get_vector_configs(encoder_name)
            vectors_config.update(dense_config)
            sparse_vectors_config.update(sparse_config)
        return vectors_config, sparse_vectors_config

    def summarize_vector(vector):
        if isinstance(vector, list):
            return {
                "kind": "dense",
                "length": len(vector),
            }
        return {
            "kind": "sparse",
            "n_indices": len(vector.indices),
            "n_values": len(vector.values),
        }

    def sparse_vectors_are_equal(left_vector, right_vector):
        return (
            left_vector.indices == right_vector.indices
            and left_vector.values == right_vector.values
        )

    client = QdrantClient(
        url=f"{os.environ['QDRANT_URL'].rstrip('/')}:{QDRANT_EXTERNAL_PROXY_PORT}",
        api_key=os.environ["QDRANT_API_KEY"],
        timeout=COLLECTION_HANDLER_QDRANT_CLIENT_TIMEOUT,
    )
    vectors_config, sparse_vectors_config = get_vector_configs_for_encoders([base_encoder_name])

    client.recreate_collection(
        collection_name=collection_name,
        vectors_config=vectors_config,
        sparse_vectors_config=sparse_vectors_config,
    )

    batch = {
        "texts": texts,
    }
    base_embeddings = run_encoder_batch_document_embedder(
        collection_name,
        batch,
        base_encoder_name,
        "test_qdrant_m3_vector_update",
    )
    tuned_embeddings = run_encoder_batch_document_embedder(
        collection_name,
        batch,
        tuned_encoder_name,
        "test_qdrant_m3_vector_update",
    )

    client.upsert(
        collection_name=collection_name,
        points=[
            models.PointStruct(
                id=point_index,
                payload={
                    "text": text,
                },
                vector=get_qdrant_vectors(base_encoder_name, base_embeddings, point_index),
            )
            for point_index, text in enumerate(texts)
        ],
    )

    source_points, _ = client.scroll(
        collection_name=collection_name,
        limit=len(texts),
        with_payload=True,
        with_vectors=True,
    )

    post_sft_vectors_config, post_sft_sparse_vectors_config = get_vector_configs_for_encoders([
        base_encoder_name,
        tuned_encoder_name,
    ])
    client.recreate_collection(
        collection_name=post_sft_collection_name,
        vectors_config=post_sft_vectors_config,
        sparse_vectors_config=post_sft_sparse_vectors_config,
    )
    client.upsert(
        collection_name=post_sft_collection_name,
        points=[
            models.PointStruct(
                id=point.id,
                payload=point.payload,
                vector={
                    **point.vector,
                    **get_qdrant_vectors(tuned_encoder_name, tuned_embeddings, point_index),
                },
            )
            for point_index, point in enumerate(source_points)
        ],
    )

    post_sft_points, _ = client.scroll(
        collection_name=post_sft_collection_name,
        limit=len(texts),
        with_payload=True,
        with_vectors=True,
    )
    source_points_by_id = {
        point.id: point
        for point in source_points
    }
    return [
        {
            "id": point.id,
            "payload": point.payload,
            "copied_base_dense_matches": (
                point.vector[f"{base_encoder_name}_dense"]
                == source_points_by_id[point.id].vector[f"{base_encoder_name}_dense"]
            ),
            "copied_base_sparse_matches": sparse_vectors_are_equal(
                point.vector[f"{base_encoder_name}_sparse"],
                source_points_by_id[point.id].vector[f"{base_encoder_name}_sparse"],
            ),
            "vectors": {
                vector_name: summarize_vector(vector)
                for vector_name, vector in sorted(point.vector.items())
            },
        }
        for point in post_sft_points
    ]

@app.local_entrypoint()
def main():
    texts = load_texts()
    if len(texts) != n_points:
        raise ValueError(f"main: expected {n_points} texts, got {len(texts)}")
    dump = test_qdrant_m3_vector_update.remote(texts)
    with open(dump_path, "w", encoding="utf-8") as dump_file:
        json.dump(dump, dump_file, indent=2)

    print(f"wrote {len(dump)} point summaries to {dump_path.relative_to(project_root)}")
    for point_summary in dump:
        print(f"point {point_summary['id']}: {sorted(point_summary['vectors'])}")
