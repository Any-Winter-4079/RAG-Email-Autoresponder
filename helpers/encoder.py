#############################################
# Helper 1: Run encoder upserter or updater #
#############################################
def run_encoder_upserter_or_updater(variant, timestamp, start_index, batch_size, encoder, worker_name, upsert_or_update="upsert"):
    from config.encoder import ENCODERS
    from config.general import QDRANT_PATH, rag_volume
    from config.crawler_agent import (
        FILE_START,
        RAW_CHUNKS_PATH,
        MANUALLY_CLEANED_CHUNKS_PATH,
        LM_CLEANED_TEXT_SUBCHUNKS_PATH,
        LM_SUMMARY_SUBCHUNKS_PATH,
        LM_Q_AND_A_VALID_CHUNKS_PATH,
        LM_Q_AND_A_FOR_Q_ONLY_VALID_CHUNKS_PATH
    )
    import os
    import json
    from qdrant_client import QdrantClient, models
    from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

    encode_paths = {
        "raw_chunks": RAW_CHUNKS_PATH,
        "manually_cleaned_chunks": MANUALLY_CLEANED_CHUNKS_PATH,
        "lm_cleaned_text_chunks": LM_CLEANED_TEXT_SUBCHUNKS_PATH,
        "lm_summary_chunks": LM_SUMMARY_SUBCHUNKS_PATH,
        "lm_q_and_a_chunks": LM_Q_AND_A_VALID_CHUNKS_PATH,
        "lm_q_and_a_for_q_only_chunks": LM_Q_AND_A_FOR_Q_ONLY_VALID_CHUNKS_PATH
    }

    rag_volume.reload()
    collection_name = variant
    client = QdrantClient(path=QDRANT_PATH)
    if not client.collection_exists(collection_name=collection_name):
        print(f"{worker_name}: {variant}: encoder '{encoder}': collection '{collection_name}' does not exist")
        return

    # load files to encode
    file_path = os.path.join(encode_paths[variant], f"{FILE_START}{timestamp}.jsonl")
    with open(file_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]
    batch = records[start_index:start_index + batch_size]
    print(f"{worker_name}: {variant}: encoder '{encoder}': loaded {len(batch)} records")

    # encode and upsert or update
    texts = []
    payloads = []
    point_ids = []
    if variant in ["lm_q_and_a_chunks", "lm_q_and_a_for_q_only_chunks"]:
        pair_start = sum(len(record["pairs"]) for record in records[:start_index])
        pair_offset = 0
        for record in batch:
            for pair_index, pair in enumerate(record["pairs"], start=1):
                if variant == "lm_q_and_a_chunks":
                    text = f"Q: {pair['question']}\nA: {pair['answer']}"
                else:
                    text = pair["question"]
                texts.append(text)
                payloads.append({
                    "variant": variant,
                    "timestamp": timestamp,
                    "url": record["url"],
                    "category": record.get("category", "university"),
                    "depth": record.get("depth"),
                    "chunk_index": record["chunk_index"],
                    "pair_index": pair_index,
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "decoder_token_count": pair["decoder_token_count"],
                    "encoder_token_count": pair["encoder_token_count"],
                })
                point_ids.append(pair_start + pair_offset)
                pair_offset += 1
    else:
        for record_offset, record in enumerate(batch):
            record_index = start_index + record_offset
            texts.append(record["text"])
            payloads.append({
                **record,
                "variant": variant,
                "timestamp": timestamp,
            })
            point_ids.append(record_index)

    encoder_config = ENCODERS[encoder]
    fastembed_kind = encoder_config["fastembed_kind"]
    if fastembed_kind == "sparse":
        encoder_model = SparseTextEmbedding(model_name=encoder_config["model_name"])
    elif fastembed_kind == "late":
        encoder_model = LateInteractionTextEmbedding(model_name=encoder_config["model_name"])
    elif fastembed_kind == "dense":
        encoder_model = TextEmbedding(model_name=encoder_config["model_name"])
    else:
        raise ValueError(f"{worker_name} does not support fastembed_kind '{fastembed_kind}' for encoder '{encoder}'")
    encoder_embeddings = list(encoder_model.embed(texts))

    points = []
    for i in range(len(texts)):
        embedding = encoder_embeddings[i]
        if fastembed_kind == "sparse":
            vectors = {
                encoder: models.SparseVector(
                    indices=embedding.indices,
                    values=embedding.values
                )
            }
        else:
            vectors = {
                encoder: embedding
            }
        if upsert_or_update == "upsert":
            points.append(
                models.PointStruct(
                    id=point_ids[i],
                    payload=payloads[i],
                    vector=vectors,
                )
            )
        else:
            points.append(
                models.PointVectors(
                    id=point_ids[i],
                    vector=vectors,
                )
            )
    if upsert_or_update == "upsert":
        client.upsert(collection_name=collection_name, points=points)
        print(f"{worker_name}: {variant}: encoder '{encoder}': upserted {len(points)} points into '{collection_name}'")
    elif upsert_or_update == "update":
        # https://qdrant.tech/documentation/concepts/points/?q=update_vec
        client.update_vectors(collection_name=collection_name, points=points)
        print(f"{worker_name}: {variant}: encoder '{encoder}': updated vectors for {len(points)} points in '{collection_name}'")
    else:
        raise ValueError(f"{worker_name}: unsupported upsert_or_update '{upsert_or_update}'")
    rag_volume.commit()

###################################
# Helper 2: Run encoder retriever #
###################################
def run_encoder_retriever(query_texts, variant, encoder_name, top_k, worker_name):
    from config.encoder import ENCODERS
    from config.general import QDRANT_PATH, rag_volume
    from qdrant_client import QdrantClient, models
    from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding
    from time import perf_counter

    total_start = perf_counter()
    encoder_config = ENCODERS[encoder_name]
    fastembed_kind = encoder_config["fastembed_kind"]
    model_name = encoder_config["model_name"]
    # NOTE: we use `QdrantClient(path=...)` in local mode because for periodic jobs,
    # keeping a server up 24/7 does not make much sense.
    # This means, however, that client initialization is very expensive (~60+ s),
    # because even though rag_volume is mounted, we still need to load the whole
    # collection into process memory.
    # To avoid each individual call to this function having to load the collection
    # into process memory, we cache the client in the worker process so that as long
    # as the container is running (scaledown window has not been reached in between
    # calls), we can reuse that cache and avoid the penalty.
    if not hasattr(run_encoder_retriever, "client"):
        reload_start = perf_counter()
        rag_volume.reload()
        reload_seconds = perf_counter() - reload_start

        client_init_start = perf_counter()
        run_encoder_retriever.client = QdrantClient(path=QDRANT_PATH)
        client_init_seconds = perf_counter() - client_init_start
        print(f"{worker_name}: initialized Qdrant client from '{QDRANT_PATH}'")
    else:
        reload_seconds = 0.0
        client_init_seconds = 0.0

    client = run_encoder_retriever.client

    embed_start = perf_counter()
    if fastembed_kind == "sparse":
        model = SparseTextEmbedding(model_name=model_name)
        query_embeddings = list(model.query_embed(query_texts))
        queries = [
            models.SparseVector(
                indices=query_embedding.indices,
                values=query_embedding.values
            )
            for query_embedding in query_embeddings
        ]
    elif fastembed_kind == "late":
        model = LateInteractionTextEmbedding(model_name=model_name)
        query_embeddings = list(model.query_embed(query_texts))
        queries = []
        for query_embedding in query_embeddings:
            queries.append(query_embedding)
    elif fastembed_kind == "dense":
        model = TextEmbedding(model_name=model_name)
        queries = list(model.query_embed(query_texts))
    else:
        raise ValueError(f"{worker_name} does not support fastembed_kind '{fastembed_kind}' for encoder '{encoder_name}'")
    embed_seconds = perf_counter() - embed_start

    requests = [
        models.QueryRequest(
            query=query,
            using=encoder_name,
            limit=top_k,
            with_payload=True,
        )
        for query in queries
    ]
    # https://python-client.qdrant.tech/qdrant_client.qdrant_client
    query_start = perf_counter()
    query_responses = client.query_batch_points(
        collection_name=variant,
        requests=requests,
    )
    query_seconds = perf_counter() - query_start

    results = [
        [
            {"score": point.score, "payload": point.payload or {}}
            for point in query_response.points
        ]
        for query_response in query_responses
    ]
    timings = {
        "reload_seconds": reload_seconds,
        "client_init_seconds": client_init_seconds,
        "embed_seconds": embed_seconds,
        "query_seconds": query_seconds,
        "total_seconds": perf_counter() - total_start,
    }
    return {"results": results, "timings": timings}
