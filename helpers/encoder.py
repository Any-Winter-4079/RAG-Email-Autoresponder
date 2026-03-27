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

######################################################
# Helper 2: Get FlagEmbedding dense model max length #
######################################################
def get_flag_embedding_max_length(encoder_config, variant):
    from config.decoder import MODEL_PROFILES as DECODER_MODEL_PROFILES, DATA_CLEANER_PROFILE

    encoder_max_length = encoder_config["max_recommended_input_size"]
    if variant in {"raw_chunks", "manually_cleaned_chunks"}:
        return encoder_max_length

    data_cleaner_chunk_size = DECODER_MODEL_PROFILES[DATA_CLEANER_PROFILE]["max_chunk_size"]
    return min(encoder_max_length, data_cleaner_chunk_size + 256)

#######################################
# Helper 3: Get or load encoder model #
#######################################
def get_or_load_encoder_model(function_cache_owner, encoder_config):
    from fastembed import TextEmbedding, SparseTextEmbedding, LateInteractionTextEmbedding

    fastembed_kind = encoder_config["fastembed_kind"]
    embedding_backend = encoder_config.get("embedding_backend", "fastembed")
    model_name = encoder_config["model_name"]
    unsupported_backend_message = (
        f"Unsupported embedding_backend '{embedding_backend}' "
        f"for fastembed_kind '{fastembed_kind}' and encoder '{model_name}'"
    )

    if not hasattr(function_cache_owner, "encoder_models"):
        function_cache_owner.encoder_models = {}

    model_cache_key = (fastembed_kind, embedding_backend, model_name)
    if model_cache_key in function_cache_owner.encoder_models:
        return function_cache_owner.encoder_models[model_cache_key]

    if fastembed_kind == "sparse":
        if embedding_backend != "fastembed":
            raise ValueError(unsupported_backend_message)
        function_cache_owner.encoder_models[model_cache_key] = SparseTextEmbedding(model_name=model_name)
    elif fastembed_kind == "late":
        if embedding_backend != "fastembed":
            raise ValueError(unsupported_backend_message)
        function_cache_owner.encoder_models[model_cache_key] = LateInteractionTextEmbedding(model_name=model_name)
    elif fastembed_kind == "dense":
        if embedding_backend == "fastembed":
            if encoder_config.get("needs_custom_fastembed_registration", False):
                register_custom_fastembed_dense_model(encoder_config)
            function_cache_owner.encoder_models[model_cache_key] = TextEmbedding(model_name=model_name)
        elif embedding_backend == "flag_embedding":
            from FlagEmbedding import BGEM3FlagModel

            function_cache_owner.encoder_models[model_cache_key] = BGEM3FlagModel(
                model_name,
                use_fp16=encoder_config.get("use_fp16", False),
            )
        else:
            raise ValueError(unsupported_backend_message)
    else:
        raise ValueError(f"Unsupported fastembed_kind '{fastembed_kind}' for encoder '{model_name}'")

    return function_cache_owner.encoder_models[model_cache_key]

########################################
# Helper 4: Run encoder batch embedder #
########################################
def run_encoder_batch_embedder(variant, batch, encoder, worker_name):
    from config.encoder import EMBEDDING_ENCODERS
    texts = batch["texts"]
    print(f"{worker_name}: {variant}: encoder '{encoder}': embedding {len(texts)} texts")
    encoder_config = EMBEDDING_ENCODERS[encoder]
    fastembed_kind = encoder_config["fastembed_kind"]
    embedding_backend = encoder_config.get("embedding_backend", "fastembed")
    encoder_model = get_or_load_encoder_model(run_encoder_batch_embedder, encoder_config)
    if fastembed_kind == "sparse":
        embeddings = list(encoder_model.embed(texts))
    elif fastembed_kind == "late":
        embeddings = list(encoder_model.embed(texts))
    elif fastembed_kind == "dense":
        if embedding_backend == "fastembed":
            embeddings = list(encoder_model.embed(texts))
        elif embedding_backend == "flag_embedding":
            embeddings = encoder_model.encode(
                texts,
                batch_size=len(texts),
                max_length=get_flag_embedding_max_length(encoder_config, variant),
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )["dense_vecs"]
        else:
            raise ValueError(
                f"{worker_name} does not support embedding_backend '{embedding_backend}' "
                f"for fastembed_kind '{fastembed_kind}' and encoder '{encoder}'"
            )
    else:
        raise ValueError(f"{worker_name} does not support fastembed_kind '{fastembed_kind}' for encoder '{encoder}'")

    return embeddings

###################################
# Helper 5: Run encoder retriever #
###################################
def run_encoder_retriever(query_texts, variant, encoder_name, top_k, worker_name):
    from config.encoder import EMBEDDING_ENCODERS
    from config.general import QDRANT_PATH, rag_volume
    from qdrant_client import QdrantClient, models
    from time import perf_counter

    total_start = perf_counter()
    n_queries = len(query_texts)
    encoder_config = EMBEDDING_ENCODERS[encoder_name]
    fastembed_kind = encoder_config["fastembed_kind"]
    embedding_backend = encoder_config.get("embedding_backend", "fastembed")
    print(
        f"{worker_name}: starting retrieval for encoder='{encoder_name}', "
        f"variant='{variant}', n_queries={n_queries}, top_k={top_k}",
        flush=True,
    )
    encoder_model = get_or_load_encoder_model(run_encoder_retriever, encoder_config)
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
        print(f"{worker_name}: Qdrant client cache miss, reloading volume", flush=True)
        reload_start = perf_counter()
        rag_volume.reload()
        reload_seconds = perf_counter() - reload_start

        client_init_start = perf_counter()
        run_encoder_retriever.client = QdrantClient(path=QDRANT_PATH)
        client_init_seconds = perf_counter() - client_init_start
        print(
            f"{worker_name}: initialized Qdrant client from '{QDRANT_PATH}' "
            f"(reload={reload_seconds:.2f}s, init={client_init_seconds:.2f}s)",
            flush=True,
        )
    else:
        reload_seconds = 0.0
        client_init_seconds = 0.0
        print(f"{worker_name}: Qdrant client cache hit", flush=True)

    client = run_encoder_retriever.client

    embed_start = perf_counter()
    print(
        f"{worker_name}: embedding {n_queries} queries with "
        f"fastembed_kind='{fastembed_kind}', backend='{embedding_backend}'",
        flush=True,
    )
    if fastembed_kind == "sparse":
        query_embeddings = list(encoder_model.query_embed(query_texts))
        queries = [
            models.SparseVector(
                indices=query_embedding.indices,
                values=query_embedding.values
            )
            for query_embedding in query_embeddings
        ]
    elif fastembed_kind == "late":
        query_embeddings = list(encoder_model.query_embed(query_texts))
        queries = []
        for query_embedding in query_embeddings:
            queries.append(query_embedding)
    elif fastembed_kind == "dense":
        if embedding_backend == "fastembed":
            queries = list(encoder_model.query_embed(query_texts))
        elif embedding_backend == "flag_embedding":
            queries = encoder_model.encode(
                query_texts,
                batch_size=len(query_texts),
                max_length=get_flag_embedding_max_length(encoder_config, variant),
                return_dense=True,
                return_sparse=False,
                return_colbert_vecs=False,
            )["dense_vecs"]
            if hasattr(queries, "tolist"):
                queries = queries.tolist()
        else:
            raise ValueError(
                f"{worker_name} does not support embedding_backend '{embedding_backend}' "
                f"for fastembed_kind '{fastembed_kind}' and encoder '{encoder_name}'"
            )
    else:
        raise ValueError(f"{worker_name} does not support fastembed_kind '{fastembed_kind}' for encoder '{encoder_name}'")
    embed_seconds = perf_counter() - embed_start
    print(
        f"{worker_name}: finished embedding {n_queries} queries in {embed_seconds:.2f}s",
        flush=True,
    )

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
        "reload_seconds": reload_seconds,
        "client_init_seconds": client_init_seconds,
        "embed_seconds": embed_seconds,
        "query_seconds": query_seconds,
        "total_seconds": perf_counter() - total_start,
    }
    print(
        f"{worker_name}: retrieval complete "
        f"(total={timings['total_seconds']:.2f}s, reload={reload_seconds:.2f}s, "
        f"client_init={client_init_seconds:.2f}s, embed={embed_seconds:.2f}s, "
        f"query={query_seconds:.2f}s)",
        flush=True,
    )
    return {"results": results, "timings": timings}
