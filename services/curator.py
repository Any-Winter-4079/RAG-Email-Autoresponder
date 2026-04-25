from config.general import modal_secret, rag_volume, VOLUME_PATH
from config.modal_apps import CURATOR_APP_NAME
from config.curator import (
    image,
    MODAL_TIMEOUT,
    SCALEDOWN_WINDOW,
    MIN_CONTAINERS,
    NONPREEMPTIBLE,
)
import modal

# Modal
app = modal.App(CURATOR_APP_NAME)

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    nonpreemptible=NONPREEMPTIBLE,
    volumes={VOLUME_PATH: rag_volume},
)
def run_email_knowledge_base_curator_pipeline(
    threads=None,
    reuse_curation=False,
    reuse_timestamp=None,
):
    import asyncio
    import datetime
    import os
    from llama_index.core.node_parser import SentenceSplitter
    from transformers import AutoTokenizer
    from config.crawler_agent import (
        CHUNK_OVERLAP,
        ENCODE_VARIANTS,
    )
    from config.data import (
        EMAIL_KNOWLEDGE_BASE_FILE_START,
        EMAIL_KNOWLEDGE_BASE_MAX_EMAILS,
        EMAIL_KNOWLEDGE_BASE_MAX_THREADS,
        EMAIL_KNOWLEDGE_BASE_RECREATE_COLLECTIONS,
        EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX,
    )
    from config.decoder import (
        EMAIL_KNOWLEDGE_BASE_CURATOR_PROFILE,
        EMAIL_WRITER_PROFILE,
        MAX_CONCURRENT_BATCHES,
        MODEL_PROFILES,
    )
    from config.encoder import EMBEDDING_ENCODERS
    from config.modal_apps import (
        COLLECTION_HANDLER_APP_NAME,
        DECODER_LATEST_TOKENIZER_APP_NAME,
        STORAGE_HANDLER_APP_NAME,
    )
    from config.modal_functions import (
        COUNT_DECODER_LATEST_TOKENS_FUNCTION_NAME,
        CREATE_COLLECTIONS_FUNCTION_NAME,
        READ_JSONL_RECORDS_FUNCTION_NAME,
        WRITE_BATCH_POINTS_FUNCTION_NAME,
        WRITE_CHUNK_RECORDS_FUNCTION_NAME,
    )
    from helpers.data import (
        build_email_thread_knowledge_base_chunks,
        prepare_batches_for_data_variant,
        run_email_knowledge_base_curator_on_threads,
    )
    from helpers.decoder import count_tokens
    from helpers.qdrant import ensure_qdrant_server_ready, persist_qdrant_storage

    variant_to_reuse_folder_name = {
        f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}lm_cleaned_text_chunks": (
            f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}lm_cleaned_text_subchunks"
        ),
        f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}lm_summary_chunks": (
            f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}lm_summary_subchunks"
        ),
        f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}lm_q_and_a_chunks": (
            f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}lm_q_and_a_valid_chunks"
        ),
        f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}lm_q_and_a_for_q_only_chunks": (
            f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}lm_q_and_a_for_q_only_valid_chunks"
        ),
    }

    variant_to_records = {}
    encode_timestamp = None
    curator_run_data = None

    # first, either reuse the stored curator stage or run the curator again
    if reuse_curation:
        read_jsonl_records = modal.Function.from_name(
            STORAGE_HANDLER_APP_NAME,
            READ_JSONL_RECORDS_FUNCTION_NAME,
        )
        anchor_variant = next(iter(variant_to_reuse_folder_name))
        anchor_folder_name = variant_to_reuse_folder_name[anchor_variant]
        anchor_result = read_jsonl_records.remote(
            variant_path=os.path.join(VOLUME_PATH, anchor_folder_name),
            file_start=EMAIL_KNOWLEDGE_BASE_FILE_START,
            timestamp=reuse_timestamp,
        )
        encode_timestamp = anchor_result["timestamp"]
        variant_to_records[anchor_variant] = anchor_result["records"]

        for variant, folder_name in list(variant_to_reuse_folder_name.items())[1:]:
            reuse_result = read_jsonl_records.remote(
                variant_path=os.path.join(VOLUME_PATH, folder_name),
                file_start=EMAIL_KNOWLEDGE_BASE_FILE_START,
                timestamp=encode_timestamp,
            )
            variant_to_records[variant] = reuse_result["records"]

        print(
            "run_email_knowledge_base_curator_pipeline: reusing stored curator outputs: "
            f"timestamp {encode_timestamp}; skipping curator plots"
        )
    else:
        if threads is None:
            raise ValueError(
                "run_email_knowledge_base_curator_pipeline: threads are required:\n"
                "\treuse_curation is false"
            )

        email_knowledge_base_curator_profile_config = MODEL_PROFILES[
            EMAIL_KNOWLEDGE_BASE_CURATOR_PROFILE
        ].copy()
        prompt_template = email_knowledge_base_curator_profile_config.pop("prompt_template")
        max_input_tokens = email_knowledge_base_curator_profile_config.pop(
            "max_input_tokens",
            None,
        )
        provider = email_knowledge_base_curator_profile_config.get("provider")
        if provider != "local":
            raise ValueError(
                "run_email_knowledge_base_curator_pipeline: unsupported provider for private data:\n"
                f"\t{provider}"
            )
        decoder_app_name = email_knowledge_base_curator_profile_config.pop(
            "decoder_app_name"
        )
        decoder_function_name = email_knowledge_base_curator_profile_config.pop(
            "decoder_function_name"
        )
        email_knowledge_base_curator_profile_config["decoder_profile"] = (
            EMAIL_KNOWLEDGE_BASE_CURATOR_PROFILE
        )

        run_email_knowledge_base_curator = modal.Function.from_name(
            decoder_app_name,
            decoder_function_name,
        )
        run_decoder_latest_tokenizer = modal.Function.from_name(
            DECODER_LATEST_TOKENIZER_APP_NAME,
            COUNT_DECODER_LATEST_TOKENS_FUNCTION_NAME,
        )

        curator_run_data = run_email_knowledge_base_curator_on_threads(
            threads,
            run_email_knowledge_base_curator,
            run_decoder_latest_tokenizer,
            email_knowledge_base_curator_profile_config,
            prompt_template,
            EMAIL_KNOWLEDGE_BASE_MAX_EMAILS,
            EMAIL_KNOWLEDGE_BASE_MAX_THREADS,
            MAX_CONCURRENT_BATCHES,
            max_input_tokens,
            return_run_data=True,
        )
        curated_thread_outputs = curator_run_data.pop("curated_thread_outputs")
        curator_run_data["curator_statistics"]["n_curated_thread_outputs"] = len(
            curated_thread_outputs
        )
        encode_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # then convert each curated thread output into the stored chunk variants
        thread_variant_to_chunks = {
            "lm_abstract_chunks": [],
            "lm_summary_chunks": [],
            "lm_cleaned_text_chunks": [],
            "lm_q_and_a_chunks": [],
        }
        for curated_thread_chunk in curated_thread_outputs:
            if curated_thread_chunk["no_useful_information"]:
                continue
            curated_thread_chunks_by_variant = build_email_thread_knowledge_base_chunks(
                curated_thread_chunk
            )
            for variant, chunks in curated_thread_chunks_by_variant.items():
                thread_variant_to_chunks[variant].extend(chunks)

        encoders = set(
            encoder
            for _, encoder_data in ENCODE_VARIANTS.items()
            for encoder in encoder_data["encoders"]
        )
        encoder_sizes = {
            encoder: EMBEDDING_ENCODERS[encoder]["max_recommended_input_size"]
            for encoder in encoders
            if (
                encoder in EMBEDDING_ENCODERS
                and "max_recommended_input_size" in EMBEDDING_ENCODERS[encoder]
            )
        }
        chunking_encoder_name = min(encoder_sizes, key=encoder_sizes.get)
        encoder_path = EMBEDDING_ENCODERS[chunking_encoder_name]["model_name"]
        encoder_tokenizer = AutoTokenizer.from_pretrained(
            encoder_path,
            trust_remote_code=True,
        )

        decoder_path = MODEL_PROFILES[EMAIL_WRITER_PROFILE]["model_name_or_path"]
        decoder_tokenizer = AutoTokenizer.from_pretrained(
            decoder_path,
            trust_remote_code=True,
        )
        embedding_chunk_size = encoder_sizes[chunking_encoder_name]
        embedding_splitter = SentenceSplitter(
            chunk_size=embedding_chunk_size,
            chunk_overlap=CHUNK_OVERLAP,
            tokenizer=lambda text: encoder_tokenizer.encode(
                text,
                add_special_tokens=False,
            ),
        )

        for chunks in thread_variant_to_chunks.values():
            for chunk_index, chunk in enumerate(chunks, start=1):
                chunk["chunk_index"] = chunk_index
                if "text" in chunk:
                    chunk["decoder_token_count"] = count_tokens(
                        decoder_tokenizer,
                        chunk["text"],
                    )
                    chunk["encoder_token_count"] = count_tokens(
                        encoder_tokenizer,
                        chunk["text"],
                    )
                elif "pairs" in chunk:
                    for pair in chunk["pairs"]:
                        pair["decoder_token_count_q"] = count_tokens(
                            decoder_tokenizer,
                            pair["question"],
                        )
                        pair["encoder_token_count_q"] = count_tokens(
                            encoder_tokenizer,
                            pair["question"],
                        )
                        pair["decoder_token_count_a"] = count_tokens(
                            decoder_tokenizer,
                            pair["answer"],
                        )
                        pair["encoder_token_count_a"] = count_tokens(
                            encoder_tokenizer,
                            pair["answer"],
                        )

        write_chunk_records = modal.Function.from_name(
            STORAGE_HANDLER_APP_NAME,
            WRITE_CHUNK_RECORDS_FUNCTION_NAME,
        )
        for (variant, chunks), label in zip(
            thread_variant_to_chunks.items(),
            [
                "EMAIL LM ABSTRACT",
                "EMAIL LM SUMMARY",
                "EMAIL LM CLEANED TEXT",
                "EMAIL LM Q&A",
            ],
        ):
            variant_path = os.path.join(
                VOLUME_PATH,
                f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}{variant}",
            )
            json_path = os.path.join(
                variant_path,
                f"{EMAIL_KNOWLEDGE_BASE_FILE_START}{encode_timestamp}.jsonl",
            )
            txt_path = os.path.join(
                variant_path,
                f"{EMAIL_KNOWLEDGE_BASE_FILE_START}{encode_timestamp}.txt",
            )
            write_chunk_records.remote(
                chunks,
                json_path,
                txt_path,
                label,
                decoder_path,
                encoder_path,
            )

            if variant == "lm_abstract_chunks":
                continue

            if variant in ["lm_cleaned_text_chunks", "lm_summary_chunks"]:
                subchunks = []
                for chunk in chunks:
                    split_texts = embedding_splitter.split_text(chunk["text"].strip())
                    for split_index, split_text in enumerate(split_texts, start=1):
                        subchunks.append({
                            "thread_id": chunk["thread_id"],
                            "chunk_index": chunk["chunk_index"],
                            "subchunk_index": split_index,
                            "text": split_text,
                            "decoder_token_count": count_tokens(
                                decoder_tokenizer,
                                split_text,
                            ),
                            "encoder_token_count": count_tokens(
                                encoder_tokenizer,
                                split_text,
                            ),
                        })

                if not subchunks:
                    continue

                subchunk_path = os.path.join(
                    VOLUME_PATH,
                    f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}"
                    f"{variant.removesuffix('_chunks')}_subchunks",
                )
                json_path = os.path.join(
                    subchunk_path,
                    f"{EMAIL_KNOWLEDGE_BASE_FILE_START}{encode_timestamp}.jsonl",
                )
                txt_path = os.path.join(
                    subchunk_path,
                    f"{EMAIL_KNOWLEDGE_BASE_FILE_START}{encode_timestamp}.txt",
                )
                write_chunk_records.remote(
                    subchunks,
                    json_path,
                    txt_path,
                    label,
                    decoder_path,
                    encoder_path,
                )
                variant_to_records[
                    f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}{variant}"
                ] = subchunks
                continue

            valid_q_and_a_chunks = []
            valid_q_only_chunks = []
            for chunk in chunks:
                valid_q_and_a_pairs = []
                valid_q_only_pairs = []
                for pair in chunk["pairs"]:
                    q_and_a_text = f"Q: {pair['question']}\nA: {pair['answer']}".strip()
                    q_only_text = pair["question"].strip()
                    q_and_a_token_ids = encoder_tokenizer.encode(
                        q_and_a_text,
                        add_special_tokens=False,
                    )
                    q_only_token_ids = encoder_tokenizer.encode(
                        q_only_text,
                        add_special_tokens=False,
                    )
                    if len(q_and_a_token_ids) <= embedding_chunk_size:
                        valid_q_and_a_pairs.append(pair)
                    if len(q_only_token_ids) <= embedding_chunk_size:
                        valid_q_only_pairs.append(pair)
                if valid_q_and_a_pairs:
                    valid_q_and_a_chunks.append({
                        "thread_id": chunk["thread_id"],
                        "chunk_index": chunk["chunk_index"],
                        "pairs": valid_q_and_a_pairs,
                    })
                if valid_q_only_pairs:
                    valid_q_only_chunks.append({
                        "thread_id": chunk["thread_id"],
                        "chunk_index": chunk["chunk_index"],
                        "pairs": valid_q_only_pairs,
                    })

            if valid_q_only_chunks:
                q_only_valid_path = os.path.join(
                    VOLUME_PATH,
                    f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}"
                    "lm_q_and_a_for_q_only_valid_chunks",
                )
                q_only_json_path = os.path.join(
                    q_only_valid_path,
                    f"{EMAIL_KNOWLEDGE_BASE_FILE_START}{encode_timestamp}.jsonl",
                )
                q_only_txt_path = os.path.join(
                    q_only_valid_path,
                    f"{EMAIL_KNOWLEDGE_BASE_FILE_START}{encode_timestamp}.txt",
                )
                write_chunk_records.remote(
                    valid_q_only_chunks,
                    q_only_json_path,
                    q_only_txt_path,
                    "EMAIL LM Q&A FOR Q ONLY",
                    decoder_path,
                    encoder_path,
                )
                variant_to_records[
                    f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}"
                    "lm_q_and_a_for_q_only_chunks"
                ] = valid_q_only_chunks

            if not valid_q_and_a_chunks:
                continue

            q_and_a_valid_path = os.path.join(
                VOLUME_PATH,
                f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}lm_q_and_a_valid_chunks",
            )
            json_path = os.path.join(
                q_and_a_valid_path,
                f"{EMAIL_KNOWLEDGE_BASE_FILE_START}{encode_timestamp}.jsonl",
            )
            txt_path = os.path.join(
                q_and_a_valid_path,
                f"{EMAIL_KNOWLEDGE_BASE_FILE_START}{encode_timestamp}.txt",
            )
            write_chunk_records.remote(
                valid_q_and_a_chunks,
                json_path,
                txt_path,
                label,
                decoder_path,
                encoder_path,
            )
            variant_to_records[
                f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}{variant}"
            ] = valid_q_and_a_chunks

    # finally, encode each stored variant and write the points into the collections
    create_collections = modal.Function.from_name(
        COLLECTION_HANDLER_APP_NAME,
        CREATE_COLLECTIONS_FUNCTION_NAME,
    )
    write_batch_points = modal.Function.from_name(
        COLLECTION_HANDLER_APP_NAME,
        WRITE_BATCH_POINTS_FUNCTION_NAME,
    )

    create_collections.remote(
        list(variant_to_records.keys()),
        EMAIL_KNOWLEDGE_BASE_RECREATE_COLLECTIONS,
    )
    if not persist_qdrant_storage("run_email_knowledge_base_curator_pipeline"):
        return

    variant_and_size_to_prepared_batches = {}
    encoder_to_batch_jobs = {}
    for variant, records in variant_to_records.items():
        base_variant = variant.removeprefix(EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX)
        variant_config = ENCODE_VARIANTS[base_variant]
        first_encoder_name = list(variant_config["encoders"].keys())[0]
        for encoder_name, encoder_config_for_variant in variant_config["encoders"].items():
            batch_size = encoder_config_for_variant["batch_size"]
            variant_and_size = (variant, batch_size)
            if variant_and_size not in variant_and_size_to_prepared_batches:
                variant_and_size_to_prepared_batches[variant_and_size] = (
                    prepare_batches_for_data_variant(
                        variant=variant,
                        records=records,
                        batch_size=batch_size,
                        encode_timestamp=encode_timestamp,
                    )
                )
            encoder_to_batch_jobs.setdefault(encoder_name, [])
            upsert_or_update = (
                "upsert"
                if encoder_name == first_encoder_name else "update"
            )
            for batch in variant_and_size_to_prepared_batches[variant_and_size]:
                encoder_to_batch_jobs[encoder_name].append(
                    (variant, batch, upsert_or_update)
                )

    encoder_functions = {}
    for encoder_name, batch_jobs in encoder_to_batch_jobs.items():
        encoder_config = EMBEDDING_ENCODERS[encoder_name]
        service_name = encoder_config["service"]
        function_name = encoder_config["function"]
        service_key = (service_name, function_name)
        if service_key not in encoder_functions:
            encoder_functions[service_key] = modal.Function.from_name(
                service_name,
                function_name,
            )
        run_encoder = encoder_functions[service_key]

        async def gather_encoder_embeddings():
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)

            async def run_one_encoder_batch(variant, batch):
                async with semaphore:
                    return await run_encoder.remote.aio(variant, batch, encoder_name)

            return await asyncio.gather(*[
                run_one_encoder_batch(variant, batch)
                for variant, batch, _ in batch_jobs
            ])

        embeddings_by_batch = asyncio.run(gather_encoder_embeddings())
        if not ensure_qdrant_server_ready("run_email_knowledge_base_curator_pipeline"):
            return
        for (variant, batch, upsert_or_update), embeddings in zip(
            batch_jobs,
            embeddings_by_batch,
        ):
            write_batch_points.remote(
                variant,
                batch,
                encoder_name,
                embeddings,
                upsert_or_update,
            )
        if not persist_qdrant_storage("run_email_knowledge_base_curator_pipeline"):
            return

    return {
        "encode_timestamp": encode_timestamp,
        "curator_run_data": curator_run_data,
        "variant_to_n_records": {
            variant: len(records)
            for variant, records in variant_to_records.items()
        },
    }
