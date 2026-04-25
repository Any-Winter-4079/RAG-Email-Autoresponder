###################################################
# Helper 1: Register custom FastEmbed dense model #
###################################################
def register_custom_fastembed_dense_model(encoder_config):
    from fastembed import TextEmbedding
    from fastembed.common.model_description import PoolingType, ModelSource

    def is_model_in_fastembed_registry(model_name):
        if not hasattr(TextEmbedding, "list_supported_models"):
            return False
        for supported_model in TextEmbedding.list_supported_models():
            if isinstance(supported_model, str):
                registered_model_name = supported_model
            elif isinstance(supported_model, dict):
                registered_model_name = supported_model.get("model") or supported_model.get("model_name")
            else:
                registered_model_name = (
                    getattr(supported_model, "model", None)
                    or getattr(supported_model, "model_name", None)
                )
            if registered_model_name == model_name:
                return True
        return False

    model_name = encoder_config["model_name"]
    if is_model_in_fastembed_registry(model_name):
        return

    if encoder_config.get("fastembed_pooling") == "mean":
        TextEmbedding.add_custom_model(
            model=model_name,
            pooling=PoolingType.MEAN,
            normalization=encoder_config["normalization"],
            sources=ModelSource(hf=model_name),
            dim=encoder_config["vector_size"],
            model_file=encoder_config["model_file"],
        )

#######################################
# Helper 2: Get or load encoder model #
#######################################
def get_or_load_encoder_model(function_cache_owner, encoder_config):
    from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

    embedding_kinds = encoder_config["embedding_kinds"]
    embedding_backend = encoder_config.get("embedding_backend", "fastembed")
    model_name = encoder_config["model_name"]
    unsupported_backend_message = (
        f"Unsupported embedding_backend '{embedding_backend}' "
        f"for embedding_kinds '{embedding_kinds}'"
    )

    if not hasattr(function_cache_owner, "encoder_models"):
        function_cache_owner.encoder_models = {}

    model_cache_key = (embedding_backend, model_name)
    if model_cache_key in function_cache_owner.encoder_models:
        return function_cache_owner.encoder_models[model_cache_key]

    if embedding_backend == "flag_embedding":
        from FlagEmbedding import BGEM3FlagModel

        function_cache_owner.encoder_models[model_cache_key] = BGEM3FlagModel(
            model_name,
            use_fp16=encoder_config.get("use_fp16", False),
        )
    else:
        if len(embedding_kinds) > 1:
            raise ValueError(
                f"get_or_load_encoder_model does not support multiple embedding kinds "
                f"for embedding_backend '{embedding_backend}'"
            )

        embedding_kind = embedding_kinds[0]
        if embedding_kind == "sparse":
            if embedding_backend != "fastembed":
                raise ValueError(unsupported_backend_message)
            function_cache_owner.encoder_models[model_cache_key] = SparseTextEmbedding(model_name=model_name)
        elif embedding_kind == "late":
            if embedding_backend != "fastembed":
                raise ValueError(unsupported_backend_message)
            function_cache_owner.encoder_models[model_cache_key] = LateInteractionTextEmbedding(model_name=model_name)
        elif embedding_kind == "dense":
            if embedding_backend == "fastembed":
                if encoder_config.get("needs_custom_fastembed_registration", False):
                    register_custom_fastembed_dense_model(encoder_config)
                function_cache_owner.encoder_models[model_cache_key] = TextEmbedding(model_name=model_name)
            elif embedding_backend == "transformers":
                import torch
                from transformers import AutoModel

                encoder_model = AutoModel.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    dtype=torch.bfloat16,
                )
                function_cache_owner.encoder_models[model_cache_key] = encoder_model.to(
                    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
                )
            elif embedding_backend == "sentence_transformers":
                import torch
                from sentence_transformers import SentenceTransformer

                function_cache_owner.encoder_models[model_cache_key] = SentenceTransformer(
                    model_name,
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    model_kwargs={"trust_remote_code": True},
                    tokenizer_kwargs={"padding_side": "left"},
                )
            else:
                raise ValueError(unsupported_backend_message)
        else:
            raise ValueError(f"Unsupported embedding kind '{embedding_kind}' for encoder '{model_name}'")

    return function_cache_owner.encoder_models[model_cache_key]

##########################################################
# Helper 3: Normalize different-backend dense embeddings #
##########################################################
def normalize_dense_embeddings(embeddings):
    if hasattr(embeddings, "detach") and hasattr(embeddings, "cpu"):
        embeddings = embeddings.detach().cpu()
    if hasattr(embeddings, "float") and hasattr(embeddings, "numpy"):
        embeddings = embeddings.float().numpy()
    return embeddings

########################################
# Helper 4: Run encoder batch embedder #
########################################
def run_encoder_batch_embedder(variant, batch, encoder, worker_name):
    from config.encoder import EMBEDDING_ENCODERS
    texts = batch["texts"]
    print(f"{worker_name}: {variant}: encoder '{encoder}': embedding {len(texts)} texts")
    encoder_config = EMBEDDING_ENCODERS[encoder]
    embedding_kinds = encoder_config["embedding_kinds"]
    embedding_backend = encoder_config.get("embedding_backend", "fastembed")
    encoder_model = get_or_load_encoder_model(run_encoder_batch_embedder, encoder_config)
    if embedding_backend == "flag_embedding":
        raw_embeddings = encoder_model.encode(
            texts,
            batch_size=len(texts),
            max_length=encoder_config["max_recommended_input_size"],
            return_dense="dense" in embedding_kinds,
            return_sparse="sparse" in embedding_kinds,
            return_colbert_vecs="late" in embedding_kinds,
        )
        embeddings = {}
        if "dense" in embedding_kinds and "dense_vecs" in raw_embeddings:
            embeddings["dense"] = list(normalize_dense_embeddings(raw_embeddings["dense_vecs"]))
        if "sparse" in embedding_kinds and "lexical_weights" in raw_embeddings:
            embeddings["sparse"] = raw_embeddings["lexical_weights"]
        if "late" in embedding_kinds and "colbert_vecs" in raw_embeddings:
            embeddings["late"] = raw_embeddings["colbert_vecs"]
        return embeddings

    if embedding_backend == "fastembed":
        embeddings = list(encoder_model.embed(texts))
        if embedding_kinds[0] == "dense":
            embeddings = list(normalize_dense_embeddings(embeddings))
        return {embedding_kinds[0]: embeddings}

    if "dense" not in embedding_kinds:
        raise ValueError(
            f"{worker_name}: embedding_backend '{embedding_backend}' only supports "
            f"dense embeddings, got embedding_kinds '{embedding_kinds}'"
        )

    if embedding_backend == "transformers":
        embeddings = encoder_model.encode(
            texts=texts,
            task="retrieval",
            prompt_name="document",
            max_length=encoder_config["max_recommended_input_size"],
        )
    elif embedding_backend == "sentence_transformers":
        embeddings = encoder_model.encode(
            texts,
            batch_size=len(texts),
        )
    else:
        raise ValueError(
            f"{worker_name}: unsupported dense embedding_backend '{embedding_backend}'"
        )

    embeddings = normalize_dense_embeddings(embeddings)
    return {"dense": list(embeddings)}

###################################
# Helper 5: Run encoder retriever #
###################################
def run_encoder_retriever(query_texts, variant, encoder_name, top_k, worker_name):
    import os
    from config.encoder import EMBEDDING_ENCODERS
    from config.qdrant_server import QDRANT_EXTERNAL_PROXY_PORT
    from qdrant_client import QdrantClient, models
    from time import perf_counter
    import json

    total_start = perf_counter()
    n_queries = len(query_texts)
    encoder_config = EMBEDDING_ENCODERS[encoder_name]
    embedding_kinds = encoder_config["embedding_kinds"]
    embedding_backend = encoder_config.get("embedding_backend", "fastembed")
    print(
        f"{worker_name}: starting retrieval for encoder='{encoder_name}', "
        f"variant='{variant}', n_queries={n_queries}, top_k={top_k}",
        flush=True,
    )
    encoder_model = get_or_load_encoder_model(run_encoder_retriever, encoder_config)
    # cache the HTTP client in the worker process so warm containers can reuse it
    if not hasattr(run_encoder_retriever, "client"):
        print(f"{worker_name}: Qdrant client cache miss", flush=True)
        client_init_start = perf_counter()
        qdrant_url = f"{os.environ['QDRANT_URL'].rstrip('/')}:{QDRANT_EXTERNAL_PROXY_PORT}"
        run_encoder_retriever.client = QdrantClient(
            url=qdrant_url,
            api_key=os.environ["QDRANT_API_KEY"],
        )
        client_init_seconds = perf_counter() - client_init_start
        print(
            f"{worker_name}: initialized Qdrant client from '{qdrant_url}' "
            f"(init={client_init_seconds:.2f}s)",
            flush=True,
        )
    else:
        client_init_seconds = 0.0
        print(f"{worker_name}: Qdrant client cache hit", flush=True)

    client = run_encoder_retriever.client

    embed_start = perf_counter()
    print(
        f"{worker_name}: embedding {n_queries} queries with "
        f"embedding_kinds='{embedding_kinds}', backend='{embedding_backend}'",
        flush=True,
    )
    if embedding_backend == "flag_embedding":
        raw_query_embeddings = encoder_model.encode(
            query_texts,
            batch_size=len(query_texts),
            max_length=encoder_config["max_recommended_input_size"],
            return_dense="dense" in embedding_kinds,
            return_sparse="sparse" in embedding_kinds,
            return_colbert_vecs="late" in embedding_kinds,
        )
        queries_by_embedding_kind = {}
        if "dense" in embedding_kinds:
            queries = list(normalize_dense_embeddings(raw_query_embeddings["dense_vecs"]))
            queries_by_embedding_kind["dense"] = queries
        if "sparse" in embedding_kinds:
            queries_by_embedding_kind["sparse"] = [
                models.SparseVector(
                    indices=[int(token_id) for token_id in lexical_weights.keys()],
                    values=list(lexical_weights.values()),
                )
                for lexical_weights in raw_query_embeddings["lexical_weights"]
            ]
        if "late" in embedding_kinds:
            queries_by_embedding_kind["late"] = raw_query_embeddings["colbert_vecs"]
    elif embedding_backend == "fastembed":
        if len(embedding_kinds) != 1:
            raise ValueError(
                f"{worker_name}: embedding_backend 'fastembed' does not support "
                f"multiple embedding kinds in one encoder config: {embedding_kinds}"
            )
        embedding_kind = embedding_kinds[0]
        query_embeddings = list(encoder_model.query_embed(query_texts))
        if embedding_kind == "sparse":
            queries_by_embedding_kind = {
                "sparse": [
                    models.SparseVector(
                        indices=query_embedding.indices,
                        values=query_embedding.values,
                    )
                    for query_embedding in query_embeddings
                ]
            }
        elif embedding_kind in {"dense", "late"}:
            if embedding_kind == "dense":
                query_embeddings = list(normalize_dense_embeddings(query_embeddings))
            queries_by_embedding_kind = {embedding_kind: query_embeddings}
        else:
            raise ValueError(
                f"{worker_name}: unsupported embedding kind '{embedding_kind}' "
                f"for embedding_backend 'fastembed'"
            )
    elif embedding_backend == "transformers":
        if embedding_kinds != ["dense"]:
            raise ValueError(
                f"{worker_name}: embedding_backend 'transformers' only supports "
                f"embedding_kinds ['dense'], got {embedding_kinds}"
            )
        query_embeddings = encoder_model.encode(
            texts=query_texts,
            task="retrieval",
            prompt_name="query",
            max_length=encoder_config["max_recommended_input_size"],
        )
        query_embeddings = normalize_dense_embeddings(query_embeddings)
        queries_by_embedding_kind = {"dense": list(query_embeddings)}
    elif embedding_backend == "sentence_transformers":
        if embedding_kinds != ["dense"]:
            raise ValueError(
                f"{worker_name}: embedding_backend 'sentence_transformers' only supports "
                f"embedding_kinds ['dense'], got {embedding_kinds}"
            )
        query_embeddings = encoder_model.encode(
            query_texts,
            batch_size=len(query_texts),
            prompt_name="query",
        )
        query_embeddings = normalize_dense_embeddings(query_embeddings)
        queries_by_embedding_kind = {"dense": list(query_embeddings)}
    else:
        raise ValueError(
            f"{worker_name}: unsupported embedding_backend '{embedding_backend}'"
        )

    for embedding_kind, queries in queries_by_embedding_kind.items():
        if len(queries) != n_queries:
            raise ValueError(
                f"{worker_name}: encoder '{encoder_name}' returned {len(queries)} "
                f"queries for embedding kind '{embedding_kind}', expected {n_queries}"
            )

    embed_seconds = perf_counter() - embed_start
    print(
        f"{worker_name}: finished embedding {n_queries} queries in {embed_seconds:.2f}s",
        flush=True,
    )

    requests = []
    request_metadata = []
    for embedding_kind, queries in queries_by_embedding_kind.items():
        vector_name = encoder_name if len(embedding_kinds) == 1 else f"{encoder_name}_{embedding_kind}"
        for query_index, query in enumerate(queries):
            requests.append(
                models.QueryRequest(
                    query=query,
                    using=vector_name,
                    limit=top_k,
                    with_payload=True,
                )
            )
            request_metadata.append((query_index, embedding_kind))

    # https://python-client.qdrant.tech/qdrant_client.qdrant_client
    query_start = perf_counter()
    print(
        f"{worker_name}: querying Qdrant for {len(requests)} requests "
        f"against collection '{variant}'",
        flush=True,
    )
    query_responses = client.query_batch_points(
        collection_name=variant,
        requests=requests,
    )
    query_seconds = perf_counter() - query_start
    print(
        f"{worker_name}: finished Qdrant query_batch_points in {query_seconds:.2f}s",
        flush=True,
    )

    results = [
        [
            {"score": point.score, "payload": point.payload or {}}
            for point in query_response.points
        ]
        for query_response in query_responses
    ]
    timings = {
        "client_init_seconds": client_init_seconds,
        "embed_seconds": embed_seconds,
        "query_seconds": query_seconds,
        "total_seconds": perf_counter() - total_start,
    }
    print(
        f"{worker_name}: retrieval complete "
        f"(total={timings['total_seconds']:.2f}s, "
        f"client_init={client_init_seconds:.2f}s, embed={embed_seconds:.2f}s, "
        f"query={query_seconds:.2f}s)",
        flush=True,
    )

    if len(embedding_kinds) == 1:
        fused_results = results
    else:
        payload_key_to_candidate_by_query = [{} for _ in range(n_queries)]
        for (query_index, embedding_kind), query_top_k_chunks in zip(request_metadata, results):
            for rank, chunk in enumerate(query_top_k_chunks, start=1):
                payload_key = json.dumps(chunk["payload"], sort_keys=True, ensure_ascii=False)
                query_candidates = payload_key_to_candidate_by_query[query_index]
                if payload_key not in query_candidates:
                    query_candidates[payload_key] = {
                        **chunk,
                        "score": 0.0,
                        "embedding_kinds": [],
                    }
                query_candidates[payload_key]["score"] += 1.0 / (60 + rank)
                query_candidates[payload_key]["embedding_kinds"].append(embedding_kind)
        fused_results = [
            sorted(
                query_candidates.values(),
                key=lambda candidate: candidate["score"],
                reverse=True,
            )[:top_k]
            for query_candidates in payload_key_to_candidate_by_query
        ]

    return {"results": fused_results, "timings": timings}
