import asyncio
from collections import Counter
from time import perf_counter

import modal

from config.encoder import DEFAULT_RRF_ENCODER_WEIGHT, EMBEDDING_ENCODERS
from config.modal_apps import (
    DECODER_LATEST_TOKENIZER_APP_NAME,
    ENCODER_CPU_APP_NAME,
    ENCODER_GPU_APP_NAME,
)
from config.modal_functions import (
    COUNT_DECODER_LATEST_TOKENS_FUNCTION_NAME,
    RUN_ENCODER_CPU_BATCH_QUERY_EMBEDDER_AND_QDRANT_RETRIEVER_FUNCTION_NAME,
    RUN_ENCODER_GPU_BATCH_QUERY_EMBEDDER_AND_QDRANT_RETRIEVER_FUNCTION_NAME,
    RUN_ENCODER_GPU_RERANKER_FUNCTION_NAME,
    TRUNCATE_DECODER_LATEST_TEXTS_FUNCTION_NAME,
)
from config.retrieval_pipeline import (
    QUERY_REWRITER_BODY_TOKEN_RATIO,
    QUERY_REWRITER_LOG_MESSAGE_START_CHARS,
    QUERY_REWRITER_THREAD_CONTEXT_TOKEN_RATIO,
)
from helpers.data import format_email_prompt_block, get_unquoted_text
from helpers.decoder import count_tokens, get_tokenizer, truncate_to_tokens
from helpers.general import get_text_from_payload

#########################################
# Helper 1: Build query rewriter prompt #
#########################################
async def build_query_rewriter_prompt(
        query_rewriter_runtime,
        context_emails,
        subject,
        body,
        log_prefix="run_data_variant_eval",
        log_token_counts=False,
        ):
    prompt_template = query_rewriter_runtime["query_rewriter_prompt_template"]
    thread_context = format_thread_context_for_query_rewriter(context_emails)
    prompt = prompt_template.format(
        thread_context=thread_context,
        subject=subject,
        body=body,
    )
    max_input_tokens = query_rewriter_runtime["query_rewriter_max_input_tokens"]
    tokenizer = query_rewriter_runtime["query_rewriter_tokenizer"]
    if max_input_tokens is None:
        return prompt
    use_runtime_tokenizer = query_rewriter_runtime["use_runtime_query_rewriter_tokenizer"]
    model_name_or_path = query_rewriter_runtime["query_rewriter_model_name_or_path"]

    async def count_text(text):
        if use_runtime_tokenizer:
            return count_tokens(tokenizer, text)
        return (await query_rewriter_runtime["run_query_rewriter_token_counter"].remote.aio([text], model_name_or_path))[0]

    async def truncate_text(text, max_tokens):
        if use_runtime_tokenizer:
            return truncate_to_tokens(tokenizer, text, max_tokens)
        return (await query_rewriter_runtime["run_query_rewriter_text_truncator"].remote.aio([text], max_tokens, model_name_or_path))[0]

    empty_context_prompt = prompt_template.format(
        thread_context="",
        subject=subject,
        body="",
    )
    empty_context_prompt_tokens = await count_text(empty_context_prompt)

    prompt_tokens = await count_text(prompt)

    if log_token_counts:
        thread_context_tokens = await count_text(thread_context)
        body_tokens = await count_text(body)
        message_start = body[:QUERY_REWRITER_LOG_MESSAGE_START_CHARS]
        print(
            f"{log_prefix}: query rewriter token counts before truncation: "
            f"prompt template={empty_context_prompt_tokens:,} | "
            f"thread={thread_context_tokens:,} | "
            f"body={body_tokens:,}"
        )
        print(
            f"{log_prefix}: query rewriter message start:\n"
            f"{message_start}"
        )
    if prompt_tokens <= max_input_tokens:
        return prompt

    thread_context_without_quoted_text = format_thread_context_for_query_rewriter(
        context_emails,
        remove_quoted_text_from_bodies=True,
    )
    body_without_quoted_text = get_unquoted_text(body)
    prompt = prompt_template.format(
        thread_context=thread_context_without_quoted_text,
        subject=subject,
        body=body_without_quoted_text,
    )
    prompt_tokens = await count_text(prompt)
    if log_token_counts:
        thread_context_without_quoted_text_tokens = await count_text(thread_context_without_quoted_text)
        body_without_quoted_text_tokens = await count_text(body_without_quoted_text)
        print(
            f"{log_prefix}: query rewriter token counts after quoted-text removal: "
            f"prompt template={empty_context_prompt_tokens:,} | "
            f"thread={thread_context_without_quoted_text_tokens:,} | "
            f"body={body_without_quoted_text_tokens:,}"
        )
    if prompt_tokens <= max_input_tokens:
        return prompt

    available_input_tokens = max_input_tokens - empty_context_prompt_tokens
    thread_context_token_budget = int(
        available_input_tokens * QUERY_REWRITER_THREAD_CONTEXT_TOKEN_RATIO
    )
    body_token_budget = int(
        available_input_tokens * QUERY_REWRITER_BODY_TOKEN_RATIO
    )
    if log_token_counts:
        print(
            f"{log_prefix}: query rewriter token budgets after prompt template: "
            f"prompt template={empty_context_prompt_tokens:,} | "
            f"thread budget={thread_context_token_budget:,} | "
            f"body budget={body_token_budget:,}"
        )
    truncated_thread_context = await truncate_text(
        thread_context_without_quoted_text,
        thread_context_token_budget,
    )
    prompt = prompt_template.format(
        thread_context=truncated_thread_context,
        subject=subject,
        body=body_without_quoted_text,
    )
    prompt_tokens = await count_text(prompt)
    if log_token_counts:
        truncated_thread_context_tokens = await count_text(truncated_thread_context)
        print(
            f"{log_prefix}: query rewriter token counts after thread truncation: "
            f"prompt template={empty_context_prompt_tokens:,} | "
            f"thread={truncated_thread_context_tokens:,} | "
            f"body={body_without_quoted_text_tokens:,}"
        )
    if prompt_tokens <= max_input_tokens:
        return prompt

    truncated_body = await truncate_text(
        body_without_quoted_text,
        body_token_budget,
    )
    prompt = prompt_template.format(
        thread_context=truncated_thread_context,
        subject=subject,
        body=truncated_body,
    )
    if log_token_counts:
        prompt_tokens = await count_text(prompt)
        truncated_body_tokens = await count_text(truncated_body)
        print(
            f"{log_prefix}: query rewriter token counts after body truncation: "
            f"prompt template={empty_context_prompt_tokens:,} | "
            f"thread={truncated_thread_context_tokens:,} | "
            f"body={truncated_body_tokens:,}"
        )
    return prompt

#########################################
# Helper 2: Load query rewriter runtime #
#########################################
def load_query_rewriter_runtime(use_runtime_query_rewriter_tokenizer=False):
    from config.decoder import (
        MODEL_PROFILES,
        QUERY_REWRITER_PROFILE,
    )

    query_rewriter_profile_config = MODEL_PROFILES[QUERY_REWRITER_PROFILE].copy()
    query_rewriter_app_name = query_rewriter_profile_config.pop("decoder_app_name")
    query_rewriter_function_name = query_rewriter_profile_config.pop("decoder_function_name")
    query_rewriter_prompt_template = query_rewriter_profile_config.pop("prompt_template")
    query_rewriter_max_input_tokens = query_rewriter_profile_config.pop(
        "max_input_tokens",
        None,
    )
    query_rewriter_model_name_or_path = query_rewriter_profile_config["model_name_or_path"]
    query_rewriter_tokenizer = (
        get_tokenizer(query_rewriter_model_name_or_path)
        if query_rewriter_max_input_tokens is not None and use_runtime_query_rewriter_tokenizer
        else None
    )
    query_rewriter_runtime = {
        "use_runtime_query_rewriter_tokenizer": use_runtime_query_rewriter_tokenizer,
        "query_rewriter_model_name_or_path": query_rewriter_model_name_or_path,
        "query_rewriter_tokenizer": query_rewriter_tokenizer,
    }
    if query_rewriter_max_input_tokens is not None and not use_runtime_query_rewriter_tokenizer:
        query_rewriter_runtime["run_query_rewriter_token_counter"] = modal.Function.from_name(
            DECODER_LATEST_TOKENIZER_APP_NAME,
            COUNT_DECODER_LATEST_TOKENS_FUNCTION_NAME,
        )
        query_rewriter_runtime["run_query_rewriter_text_truncator"] = modal.Function.from_name(
            DECODER_LATEST_TOKENIZER_APP_NAME,
            TRUNCATE_DECODER_LATEST_TEXTS_FUNCTION_NAME,
        )

    return {
        **query_rewriter_runtime,
        "run_query_rewriter": modal.Function.from_name(
            query_rewriter_app_name,
            query_rewriter_function_name,
        ),
        "query_rewriter_profile_name": QUERY_REWRITER_PROFILE,
        "query_rewriter_model_config": query_rewriter_profile_config,
        "query_rewriter_prompt_template": query_rewriter_prompt_template,
        "query_rewriter_max_input_tokens": query_rewriter_max_input_tokens,
    }

######################################################################
# Helper 3: Get encoder query embedder and Qdrant retriever function #
######################################################################
def get_run_encoder_batch_query_embedder_and_qdrant_retriever(encoder_name):
    service_name = EMBEDDING_ENCODERS[encoder_name]["service"]
    if service_name == ENCODER_GPU_APP_NAME:
        return modal.Function.from_name(
            ENCODER_GPU_APP_NAME,
            RUN_ENCODER_GPU_BATCH_QUERY_EMBEDDER_AND_QDRANT_RETRIEVER_FUNCTION_NAME,
        )
    if service_name == ENCODER_CPU_APP_NAME:
        return modal.Function.from_name(
            ENCODER_CPU_APP_NAME,
            RUN_ENCODER_CPU_BATCH_QUERY_EMBEDDER_AND_QDRANT_RETRIEVER_FUNCTION_NAME,
        )
    raise ValueError(f"unsupported service '{service_name}' for encoder '{encoder_name}'")

##################################
# Helper 4: Get encoder reranker #
##################################
def get_run_encoder_gpu_reranker():
    return modal.Function.from_name(
        ENCODER_GPU_APP_NAME,
        RUN_ENCODER_GPU_RERANKER_FUNCTION_NAME,
    )

####################################################
# Helper 5: Dedupe query type to rewritten queries #
####################################################
def dedupe_query_type_to_rewritten_queries(query_type_to_rewritten_queries):
    n_duplicate_queries_removed = 0
    deduped_query_type_to_rewritten_queries = {}
    for query_type, queries in query_type_to_rewritten_queries.items():
        seen_queries = set()
        deduped_queries = []
        query_counts = Counter(queries)
        n_duplicate_queries_removed += sum(
            query_count - 1
            for query_count in query_counts.values()
            if query_count > 1
        )
        for query in queries:
            if query in seen_queries:
                continue
            seen_queries.add(query)
            deduped_queries.append(query)
        deduped_query_type_to_rewritten_queries[query_type] = deduped_queries
    return deduped_query_type_to_rewritten_queries, n_duplicate_queries_removed

#################################################
# Helper 6: Cap query type to rewritten queries #
#################################################
def cap_query_type_to_rewritten_queries(query_type_to_rewritten_queries, n_max_queries):
    n_queries = sum(len(queries) for queries in query_type_to_rewritten_queries.values())
    if n_queries <= n_max_queries:
        return query_type_to_rewritten_queries, 0

    n_capped_queries_removed = n_queries - n_max_queries
    for _ in range(n_capped_queries_removed):
        longest_query_type = max(
            query_type_to_rewritten_queries,
            key=lambda query_type: len(query_type_to_rewritten_queries[query_type]),
        )
        # note this mutates the dictionary
        query_type_to_rewritten_queries[longest_query_type].pop()
    return query_type_to_rewritten_queries, n_capped_queries_removed

#########################################
# Helper 7: Format context for rewriter #
#########################################
def format_thread_context_for_query_rewriter(
        context_emails,
        remove_quoted_text_from_bodies=False,
        ):
    if not context_emails:
        return "(no prior messages found)"

    formatted_context_emails = []
    for email in context_emails:
        body = email.get("body") or email.get("message_body") or ""
        if remove_quoted_text_from_bodies:
            body = get_unquoted_text(body)
        formatted_context_emails.append(
            format_email_prompt_block(
                {
                    "author": email.get("author") or email.get("from"),
                    "recipients": email.get("recipients") or email.get("to"),
                    "subject": email.get("subject"),
                    "body": body,
                    "date": email.get("date"),
                },
                date_key="date",
            )
        )

    return "\n[END MESSAGE]\n".join(formatted_context_emails)

###########################################
# Helper 8: Build source query from email #
###########################################
def build_source_query_from_email(email):
    source_subject = (email.get("subject") or "").strip()
    source_body = (email.get("message_body") or "").strip()
    return f"Subject:\n{source_subject}\n\nBody:\n{source_body}"

################################################
# Helper 9: Post-process query rewriter output #
################################################
def post_process_query_rewriter_output(
        query_rewriter_output,
        n_max_queries,
        log_prefix,
        did_raise_exception=False,
        ):
    from config.decoder import QUERY_TYPE_TO_N_MAX_QUERIES

    rewrite_result = {
        "no_request": False,
        "query_type_to_rewritten_queries": None,
        "reranker_query": None,
        "anonymized_request": None,
        "did_raise_exception": False,
        "did_return_empty_output": False,
        "did_return_no_usable_queries": False,
        "did_return_no_reranker_query": False,
        "did_return_no_anonymized_request": False,
        "n_duplicate_queries_removed": 0,
        "did_hit_query_cap": False,
        "n_capped_queries_removed": 0,
    }

    if did_raise_exception:
        rewrite_result["did_raise_exception"] = True
    elif not query_rewriter_output:
        print(f"{log_prefix}: query rewrite returned no output")
        rewrite_result["did_return_empty_output"] = True
    elif query_rewriter_output["no_request"]:
        rewrite_result["no_request"] = True
    else:
        reranker_query = query_rewriter_output.get("reranker_query", None)
        anonymized_request = query_rewriter_output.get("anonymized_request", None)
        query_type_to_rewritten_queries = {
            query_type: query_rewriter_output[f"{query_type}_queries"]
            for query_type in QUERY_TYPE_TO_N_MAX_QUERIES
        }
        if not reranker_query:
            print(f"{log_prefix}: query rewrite returned no reranker query")
            rewrite_result["did_return_no_reranker_query"] = True
        elif not anonymized_request:
            print(f"{log_prefix}: query rewrite returned no anonymized request")
            rewrite_result["did_return_no_anonymized_request"] = True
        else:
            query_type_to_rewritten_queries["reranker"] = [reranker_query]
            if not any(query_type_to_rewritten_queries.values()):
                print(f"{log_prefix}: query rewrite returned no usable queries")
                rewrite_result["did_return_no_usable_queries"] = True
            else:
                deduped_query_type_to_rewritten_queries, n_duplicate_queries_removed = dedupe_query_type_to_rewritten_queries(
                    query_type_to_rewritten_queries=query_type_to_rewritten_queries,
                )
                query_type_to_rewritten_queries, n_capped_queries_removed = cap_query_type_to_rewritten_queries(
                    query_type_to_rewritten_queries=deduped_query_type_to_rewritten_queries,
                    n_max_queries=n_max_queries,
                )
                rewrite_result["query_type_to_rewritten_queries"] = query_type_to_rewritten_queries
                rewrite_result["reranker_query"] = reranker_query
                rewrite_result["anonymized_request"] = anonymized_request
                rewrite_result["n_duplicate_queries_removed"] = n_duplicate_queries_removed
                rewrite_result["did_hit_query_cap"] = n_capped_queries_removed > 0
                rewrite_result["n_capped_queries_removed"] = n_capped_queries_removed

    return rewrite_result

######################################
# Helper 10: Rewrite one email async #
######################################
async def rewrite_one_email_async(
        email,
        query_rewriter_runtime,
        n_max_queries,
        log_prefix,
        ):
    body = (email.get("message_body") or "").strip()
    query_rewriter_prompt = await build_query_rewriter_prompt(
        query_rewriter_runtime=query_rewriter_runtime,
        context_emails=email.get("context_emails") or [],
        subject=(email.get("subject") or "").strip(),
        body=body,
        log_prefix=log_prefix,
        log_token_counts=log_prefix == "run_email_agent",
    )
    try:
        query_rewriter_output, _ = await query_rewriter_runtime["run_query_rewriter"].remote.aio(
            context=[],
            current_turn_input_text=query_rewriter_prompt,
            current_turn_image_in_bytes=None,
            **query_rewriter_runtime["query_rewriter_model_config"],
            decoder_profile=query_rewriter_runtime["query_rewriter_profile_name"],
        )
        did_raise_exception = False
    except Exception as e:
        print(f"{log_prefix}: query rewrite raised exception: {e}")
        query_rewriter_output = None
        did_raise_exception = True

    rewrite_result = post_process_query_rewriter_output(
        query_rewriter_output=query_rewriter_output,
        n_max_queries=n_max_queries,
        log_prefix=log_prefix,
        did_raise_exception=did_raise_exception,
    )
    return rewrite_result

###################################
# Helper 11: Rewrite emails async #
###################################
async def rewrite_emails_async(
        emails,
        n_max_queries=None,
        log_prefix="run_data_variant_eval",
        use_runtime_query_rewriter_tokenizer=False,
        max_concurrent_query_rewrites=None,
        ):
    from config.decoder import MAX_CONCURRENT_BATCHES, QUERY_TYPE_TO_N_MAX_QUERIES

    if n_max_queries is None:
        n_max_queries = sum(QUERY_TYPE_TO_N_MAX_QUERIES.values()) + 1

    query_rewriter_runtime = load_query_rewriter_runtime(
        use_runtime_query_rewriter_tokenizer=use_runtime_query_rewriter_tokenizer,
    )
    if max_concurrent_query_rewrites is None:
        max_concurrent_query_rewrites = MAX_CONCURRENT_BATCHES
    semaphore = asyncio.Semaphore(max_concurrent_query_rewrites)

    async def rewrite_one_email_with_semaphore(email):
        async with semaphore:
            return await rewrite_one_email_async(
                email=email,
                query_rewriter_runtime=query_rewriter_runtime,
                n_max_queries=n_max_queries,
                log_prefix=log_prefix,
            )

    rewrite_results = await asyncio.gather(*[
        rewrite_one_email_with_semaphore(email)
        for email in emails
    ])

    rewritten_emails = []
    no_request_emails = []
    n_query_rewriter_exceptions = 0
    n_empty_query_rewrite_outputs = 0
    n_no_usable_query_rewrite_outputs = 0
    n_no_reranker_query_outputs = 0
    n_no_anonymized_request_outputs = 0
    n_duplicate_queries_removed = 0
    n_query_cap_hits = 0
    n_capped_queries_removed = 0
    for email, rewrite_result in zip(emails, rewrite_results):
        if rewrite_result["did_raise_exception"]:
            n_query_rewriter_exceptions += 1
            continue
        if rewrite_result["did_return_empty_output"]:
            n_empty_query_rewrite_outputs += 1
            continue
        if rewrite_result["did_return_no_usable_queries"]:
            n_no_usable_query_rewrite_outputs += 1
            continue
        if rewrite_result["did_return_no_reranker_query"]:
            n_no_reranker_query_outputs += 1
            continue
        if rewrite_result["did_return_no_anonymized_request"]:
            n_no_anonymized_request_outputs += 1
            continue
        if rewrite_result["no_request"]:
            no_request_emails.append(email)
            continue
        n_duplicate_queries_removed += rewrite_result["n_duplicate_queries_removed"]
        n_query_cap_hits += int(rewrite_result["did_hit_query_cap"])
        n_capped_queries_removed += rewrite_result["n_capped_queries_removed"]
        rewritten_emails.append({
            "email": email,
            "query_type_to_rewritten_queries": rewrite_result["query_type_to_rewritten_queries"],
            "reranker_query": rewrite_result["reranker_query"],
            "anonymized_request": rewrite_result["anonymized_request"],
        })

    return {
        "rewritten_emails": rewritten_emails,
        "no_request_emails": no_request_emails,
        "n_query_rewriter_exceptions": n_query_rewriter_exceptions,
        "n_empty_query_rewrite_outputs": n_empty_query_rewrite_outputs,
        "n_no_usable_query_rewrite_outputs": n_no_usable_query_rewrite_outputs,
        "n_no_reranker_query_outputs": n_no_reranker_query_outputs,
        "n_no_anonymized_request_outputs": n_no_anonymized_request_outputs,
        "n_duplicate_queries_removed": n_duplicate_queries_removed,
        "n_query_cap_hits": n_query_cap_hits,
        "n_capped_queries_removed": n_capped_queries_removed,
    }

##########################################################
# Helper 12: Build retrieval query batches for one email #
##########################################################
def build_retrieval_query_batches_for_one_email(
        rewritten_email,
        retrieval_batch_size,
        ):
    retrieval_queries = [
        {"query": query, "query_type": query_type}
        for query_type, queries in rewritten_email["query_type_to_rewritten_queries"].items()
        for query in queries
    ]
    retrieval_query_batches = []
    for query_start_index in range(0, len(retrieval_queries), retrieval_batch_size):
        typed_queries = retrieval_queries[query_start_index:query_start_index + retrieval_batch_size]
        retrieval_query_batches.append({
            "typed_queries": typed_queries,
            "query_texts": [
                typed_query["query"]
                for typed_query in typed_queries
            ],
        })
    return retrieval_query_batches

###########################################
# Helper 13: Run single encoder retrieval #
###########################################
def run_single_encoder_retrieval(
        rewritten_emails,
        base_data_variant,
        source_name,
        collection_name,
        encoder_name,
        encoder_runtime_config,
        top_k_per_query,
        top_k_after_query_fusion,
        use_max_similarity_query_fusion_before_rrf,
        result_record_metadata,
        ):
    from config.decoder import MAX_CONCURRENT_BATCHES
    run_encoder_query_retriever = get_run_encoder_batch_query_embedder_and_qdrant_retriever(encoder_name)
    embedding_kinds = EMBEDDING_ENCODERS[encoder_name]["embedding_kinds"]
    embedding_kind_to_output_name = {
        embedding_kind: (
            encoder_name
            if len(embedding_kinds) == 1
            else f"{encoder_name}_{embedding_kind}"
        )
        for embedding_kind in embedding_kinds
    }
    encoder_name_to_output = {
        output_name: {
            **result_record_metadata,
            "base_data_variant": base_data_variant,
            "data_source": source_name,
            "data_variant": collection_name,
            "encoder_name": output_name,
            "top_k": top_k_after_query_fusion,
            "use_max_similarity_query_fusion_before_rrf": use_max_similarity_query_fusion_before_rrf,
            "first_batch_embed_seconds": None,
            "first_batch_query_seconds": None,
            "n_failed_emails": 0,
            "results": [],
        }
        for output_name in embedding_kind_to_output_name.values()
    }

    retrieval_jobs = []
    for email_index, rewritten_email in enumerate(rewritten_emails):
        retrieval_query_batches = build_retrieval_query_batches_for_one_email(
            rewritten_email=rewritten_email,
            retrieval_batch_size=encoder_runtime_config["batch_size"],
        )
        for retrieval_query_batch in retrieval_query_batches:
            retrieval_jobs.append({
                "email_index": email_index,
                "rewritten_email": rewritten_email,
                "typed_queries": retrieval_query_batch["typed_queries"],
                "query_texts": retrieval_query_batch["query_texts"],
            })

    async def run_retrieval_jobs_async():
        semaphore = asyncio.Semaphore(MAX_CONCURRENT_BATCHES)

        async def run_one_retrieval_job(retrieval_job):
            async with semaphore:
                try:
                    retrieval_response = await run_encoder_query_retriever.remote.aio(
                        retrieval_job["query_texts"],
                        collection_name,
                        encoder_name,
                        top_k_per_query,
                    )
                except Exception as e:
                    print(
                        "run_single_encoder_retrieval: retrieval failed"
                        f" | data variant {collection_name}"
                        f" | encoder {encoder_name}"
                        f" | error {e}"
                    )
                    return {**retrieval_job, "retrieval_response": None}
                return {**retrieval_job, "retrieval_response": retrieval_response}

        return await asyncio.gather(*[
            run_one_retrieval_job(retrieval_job)
            for retrieval_job in retrieval_jobs
        ])

    retrieval_job_results = asyncio.run(run_retrieval_jobs_async())
    email_index_to_retrieval_job_results = {
        email_index: []
        for email_index in range(len(rewritten_emails))
    }
    for retrieval_job_result in retrieval_job_results:
        email_index_to_retrieval_job_results[retrieval_job_result["email_index"]].append(
            retrieval_job_result
        )

    for email_index, rewritten_email in enumerate(rewritten_emails):
        retrieval_failed = False
        output_name_to_query_entries_with_top_k_chunks = {
            output_name: []
            for output_name in encoder_name_to_output
        }

        for retrieval_job_result in email_index_to_retrieval_job_results[email_index]:
            retrieval_response = retrieval_job_result["retrieval_response"]
            if retrieval_response is None:
                retrieval_failed = True
                break

            for output in encoder_name_to_output.values():
                if output["first_batch_embed_seconds"] is None:
                    output["first_batch_embed_seconds"] = retrieval_response["timings"]["embed_seconds"]
                    output["first_batch_query_seconds"] = retrieval_response["timings"]["query_seconds"]

            embedding_kind_to_results = retrieval_response["embedding_kind_to_results"]
            query_texts = retrieval_job_result["query_texts"]
            for embedding_kind in embedding_kinds:
                if embedding_kind not in embedding_kind_to_results:
                    print(
                        "run_single_encoder_retrieval: retrieval did not return embedding kind"
                        f" | data variant {collection_name}"
                        f" | encoder {encoder_name}"
                        f" | embedding kind {embedding_kind}"
                    )
                    retrieval_failed = True
                    break
                n_actual_query_responses = len(embedding_kind_to_results[embedding_kind])
                if n_actual_query_responses != len(query_texts):
                    print(
                        "run_single_encoder_retrieval: retrieval returned wrong number of query responses"
                        f" | data variant {collection_name}"
                        f" | encoder {encoder_name}"
                        f" | embedding kind {embedding_kind}"
                        f" | expected {len(query_texts)}"
                        f" | actual {n_actual_query_responses}"
                    )
                    retrieval_failed = True
                    break
            if retrieval_failed:
                break

            typed_query_batch = retrieval_job_result["typed_queries"]
            for embedding_kind, query_result_lists in embedding_kind_to_results.items():
                output_name = embedding_kind_to_output_name[embedding_kind]
                output_name_to_query_entries_with_top_k_chunks[output_name].extend([
                    {
                        "query": typed_query["query"],
                        "query_type": typed_query["query_type"],
                        "top_k_chunks": query_top_k_chunk_list,
                    }
                    for typed_query, query_top_k_chunk_list in zip(typed_query_batch, query_result_lists)
                ])

        if retrieval_failed:
            for output in encoder_name_to_output.values():
                output["n_failed_emails"] += 1
                output["results"].append({
                    "email": rewritten_email["email"],
                    "query_type_to_rewritten_queries": rewritten_email["query_type_to_rewritten_queries"],
                    "reranker_query": rewritten_email["reranker_query"],
                    "anonymized_request": rewritten_email.get("anonymized_request"),
                    "retrieval_failed": True,
                    "retrieval_results": [],
                })
            continue

        for output_name, query_entries_with_top_k_chunks in output_name_to_query_entries_with_top_k_chunks.items():
            result_entry = {
                "email": rewritten_email["email"],
                "query_type_to_rewritten_queries": rewritten_email["query_type_to_rewritten_queries"],
                "reranker_query": rewritten_email["reranker_query"],
                "anonymized_request": rewritten_email.get("anonymized_request"),
                "retrieval_failed": False,
            }
            if use_max_similarity_query_fusion_before_rrf:
                fused_chunks = fuse_multiple_query_types_for_one_sample(
                    query_entries_with_top_k_chunks,
                    top_k_after_query_fusion,
                )
                result_entry["retrieval_results"] = [
                    {**selected_chunk, "rank": rank}
                    for rank, selected_chunk in enumerate(fused_chunks, start=1)
                ]
            else:
                result_entry["retrieval_results"] = [
                    {
                        **retrieved_chunk,
                        "rank": rank,
                        "query_matching_retrieved_chunk": {
                            "query": query_entry["query"],
                            "query_type": query_entry["query_type"],
                        },
                    }
                    for query_entry in query_entries_with_top_k_chunks
                    for rank, retrieved_chunk in enumerate(
                        query_entry["top_k_chunks"],
                        start=1,
                    )
                ]
            encoder_name_to_output[output_name]["results"].append(result_entry)
    return encoder_name_to_output

#######################################################
# Helper 14: Fuse multiple query types for one sample #
#######################################################
def fuse_multiple_query_types_for_one_sample(sample_query_entries_with_top_k_chunks, top_k):
    import json

    # given we do query rewritting, having multiple queries per sample (email),
    # we get the top k results across all queries (which are 'rewrites' of
    # the original sample or email, hopefully optimized for retrieval)

    # sample_query_entries_with_top_k_chunks:
    # several query entries for one sample
    # and for each query entry, its top-k retrieved chunks
    # (with each chunk having its own payload)

    # multiple queries could return the same chunk,
    # so for a unique top k we index by payload
    payload_to_highest_scored_query_candidate = {}
    for sample_query_entry in sample_query_entries_with_top_k_chunks:
        rewritten_query = sample_query_entry["query"]
        query_type = sample_query_entry["query_type"]
        query_top_k_chunks = sample_query_entry["top_k_chunks"]
        for retrieved_chunk in query_top_k_chunks:
            payload_key = json.dumps(retrieved_chunk["payload"], sort_keys=True, ensure_ascii=False)
            score = retrieved_chunk["score"]
            best_query_candidate = payload_to_highest_scored_query_candidate.get(payload_key)
            if best_query_candidate is None or score > best_query_candidate["score"]:
                best_query_candidate = retrieved_chunk.copy()
                best_query_candidate["query_matching_retrieved_chunk"] = {
                    "query": rewritten_query,
                    "query_type": query_type,
                }
                payload_to_highest_scored_query_candidate[payload_key] = best_query_candidate
    sorted_candidates = sorted(
        payload_to_highest_scored_query_candidate.values(),
        key=lambda retrieved_chunk: retrieved_chunk["score"],
        reverse=True
    )
    if top_k is None:
        return sorted_candidates
    return sorted_candidates[:top_k]

####################################################################################
# Helper 15: Fuse ranked lists for one sample with weighted reciprocal rank fusion #
####################################################################################
def fuse_ranked_lists_with_weighted_rrf(ranked_list_name_to_chunks, top_k, ranked_list_name_to_weight=None, rank_constant=60):
    import json
    from config.encoder import DEFAULT_RRF_ENCODER_WEIGHT

    # each key is a ranked-list name and each value is that list's ranked chunks
    if ranked_list_name_to_weight is None:
        ranked_list_name_to_weight = {}
    
    # max-similarity query fusion:
    #   ranked_list_name_to_query_matching_retrieved_chunk = {
    #       "bm25": {"query": "...", "query_type": "hyde"},
    #       "bge_m3_dense": {"query": "...", "query_type": "natural"},
    #       ...
    #   }
    #   ranked_list_name_to_score = {"bm25": 22.46, "bge_m3_dense": 0.693, ...}
    #   ranked_list_name_to_rank = {"bm25": 4, "bge_m3_dense": 3, ...}

    # query-level RRF:
    #   ranked_list_name_to_query_matching_retrieved_chunk = {
    #       "bm25::keyword::master pre-registration dates": {"query": "...", "query_type": "keyword"},
    #       "bm25::hyde::annual admission calendar": {"query": "...", "query_type": "hyde"},
    #       ...
    #   }
    #   ranked_list_name_to_score = {"bm25::keyword::master pre-registration dates": 22.46, ...}
    #   ranked_list_name_to_rank = {"bm25::keyword::master pre-registration dates": 4, ...}
    payload_to_rrf_candidate = {}
    for ranked_list_name, top_k_chunks in ranked_list_name_to_chunks.items():
        ranked_list_weight = ranked_list_name_to_weight.get(
            ranked_list_name,
            DEFAULT_RRF_ENCODER_WEIGHT,
        )
        for retrieved_chunk in top_k_chunks:
            payload = retrieved_chunk["payload"]
            payload_key = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            if payload_key not in payload_to_rrf_candidate:
                payload_to_rrf_candidate[payload_key] = {
                    "score": 0.0,
                    "ranked_list_name_to_query_matching_retrieved_chunk": {},
                    "ranked_list_name_to_score": {},
                    "ranked_list_name_to_rank": {},
                    "payload": payload,
                }
            rrf_chunk_for_payload = payload_to_rrf_candidate[payload_key]
            rrf_chunk_for_payload["score"] += (
                ranked_list_weight / (rank_constant + retrieved_chunk["rank"])
            )
            rrf_chunk_for_payload["ranked_list_name_to_query_matching_retrieved_chunk"][ranked_list_name] = retrieved_chunk["query_matching_retrieved_chunk"]
            rrf_chunk_for_payload["ranked_list_name_to_score"][ranked_list_name] = retrieved_chunk["score"]
            rrf_chunk_for_payload["ranked_list_name_to_rank"][ranked_list_name] = retrieved_chunk["rank"]

    sorted_candidates = sorted(
        payload_to_rrf_candidate.values(),
        key=lambda retrieved_chunk: retrieved_chunk["score"],
        reverse=True,
    )
    if top_k is None:
        return sorted_candidates
    return sorted_candidates[:top_k]

#################################################################
# Helper 16: Keep category minimums from an already ranked list #
#################################################################
def keep_category_minimums_from_ranked_chunks(ranked_chunks, top_k, category_to_min_final_count):
    # ranked chunks are assummed to be **already** deduplicated
    # (e.g., from fuse_ranked_lists_with_weighted_rrf)
    plain_top_k_original_chunks = ranked_chunks[:top_k]

    if not category_to_min_final_count:
        return [
            {
                **ranked_chunk,
                "selection_reason": "top_k",
            }
            for ranked_chunk in ranked_chunks[:top_k]
        ]

    requested_min_count = sum(category_to_min_final_count.values())
    if requested_min_count > top_k:
        raise ValueError(
            "keep_category_minimums_from_ranked_chunks: requested minimum category count "
            f"({requested_min_count}) exceeds top_k ({top_k})"
        )

    selected_original_chunks = []
    selected_chunks = []
    for category, min_count in category_to_min_final_count.items():
        if min_count <= 0:
            continue
        category_chunks = [
            ranked_chunk
            for ranked_chunk in ranked_chunks
            if ranked_chunk["payload"].get("category") == category
        ][:min_count]
        selected_original_chunks.extend(category_chunks)
        selected_chunks.extend(
            {
                **ranked_chunk,
                "selection_reason": (
                    "category_minimum_and_top_k"
                    if ranked_chunk in plain_top_k_original_chunks
                    else "category_minimum_only"
                ),
            }
            for ranked_chunk in category_chunks
        )

    for ranked_chunk in ranked_chunks:
        if len(selected_chunks) == top_k:
            break
        if ranked_chunk in selected_original_chunks:
            continue
        selected_chunks.append({
            **ranked_chunk,
            "selection_reason": "top_k",
        })

    return selected_chunks

#################################################################
# Helper 17: Build reciprocal rank fusion output for one source #
#################################################################
def build_reciprocal_rank_fusion_output_for_one_source(
        base_data_variant,
        source_name,
        collection_name,
        encoder_name_to_output,
        encoder_name_to_rrf_weight,
        top_k_after_source_rrf,
        use_max_similarity_query_fusion_before_rrf,
        result_record_metadata,
        ):
    from config.encoder import DEFAULT_RRF_ENCODER_WEIGHT
    from config.eval import CATEGORY_TO_MIN_FINAL_COUNT_AFTER_RRF

    encoder_names = list(encoder_name_to_output)
    first_encoder_output = encoder_name_to_output[encoder_names[0]]
    reciprocal_rank_fusion_output_for_one_source = {
        **result_record_metadata,
        "base_data_variant": base_data_variant,
        "data_source": source_name,
        "data_variant": collection_name,
        "top_k": top_k_after_source_rrf,
        "use_max_similarity_query_fusion_before_rrf": use_max_similarity_query_fusion_before_rrf,
        "rrf_encoder_weights": {
            encoder_name: encoder_name_to_rrf_weight[encoder_name]
            for encoder_name in encoder_names
            if encoder_name in encoder_name_to_rrf_weight
        },
        "timings": {},
        "n_failed_emails": 0,
        "results": [],
    }

    rrf_seconds = 0.0
    category_minimum_seconds = 0.0
    rrf_start = perf_counter()
    for email_index, first_encoder_result in enumerate(first_encoder_output["results"]):
        encoder_results = {
            encoder_name: encoder_output["results"][email_index]
            for encoder_name, encoder_output in encoder_name_to_output.items()
        }
        successful_encoder_results = {
            encoder_name: encoder_result
            for encoder_name, encoder_result in encoder_results.items()
            if not encoder_result["retrieval_failed"]
        }

        if not successful_encoder_results:
            reciprocal_rank_fusion_output_for_one_source["n_failed_emails"] += 1
            reciprocal_rank_fusion_output_for_one_source["results"].append({
                "email": first_encoder_result["email"],
                "query_type_to_rewritten_queries": first_encoder_result["query_type_to_rewritten_queries"],
                "reranker_query": first_encoder_result["reranker_query"],
                "anonymized_request": first_encoder_result.get("anonymized_request"),
                "retrieval_failed": True,
                "retrieval_results": [],
            })
            continue

        rrf_step_start = perf_counter()
        if use_max_similarity_query_fusion_before_rrf:
            ranked_list_name_to_chunks = {
                encoder_name: encoder_result["retrieval_results"]
                for encoder_name, encoder_result in successful_encoder_results.items()
            }
            ranked_list_name_to_weight = {
                encoder_name: encoder_name_to_rrf_weight[encoder_name]
                for encoder_name in successful_encoder_results
                if encoder_name in encoder_name_to_rrf_weight
            }
        else:
            ranked_list_name_to_chunks = {}
            ranked_list_name_to_weight = {}
            for encoder_name, encoder_result in successful_encoder_results.items():
                encoder_weight = encoder_name_to_rrf_weight.get(
                    encoder_name,
                    DEFAULT_RRF_ENCODER_WEIGHT,
                )
                query_keys = {
                    (
                        retrieved_chunk["query_matching_retrieved_chunk"]["query_type"],
                        retrieved_chunk["query_matching_retrieved_chunk"]["query"],
                    )
                    for retrieved_chunk in encoder_result["retrieval_results"]
                }
                if not query_keys:
                    continue
                # query_result_weight = encoder_weight / len(query_keys)
                query_result_weight = encoder_weight
                for query_type, query in query_keys:
                    ranked_list_name = (
                        f"{encoder_name}::{query_type}::"
                        f"{query}"
                    )
                    ranked_list_name_to_chunks[ranked_list_name] = [
                        retrieved_chunk
                        for retrieved_chunk in encoder_result["retrieval_results"]
                        if (
                            retrieved_chunk["query_matching_retrieved_chunk"]["query_type"],
                            retrieved_chunk["query_matching_retrieved_chunk"]["query"],
                        ) == (query_type, query)
                    ]
                    ranked_list_name_to_weight[ranked_list_name] = query_result_weight

        if len(ranked_list_name_to_chunks) == 0:
            ranked_chunks_after_rrf = []
        elif len(ranked_list_name_to_chunks) == 1:
            ranked_chunks_after_rrf = next(iter(ranked_list_name_to_chunks.values()))
        else:
            ranked_chunks_after_rrf = fuse_ranked_lists_with_weighted_rrf(
                ranked_list_name_to_chunks,
                top_k=None,
                ranked_list_name_to_weight=ranked_list_name_to_weight,
            )
        rrf_seconds += perf_counter() - rrf_step_start

        if top_k_after_source_rrf is None:
            selected_chunks = ranked_chunks_after_rrf
        else:
            category_step_start = perf_counter()
            selected_chunks = keep_category_minimums_from_ranked_chunks(
                ranked_chunks_after_rrf,
                top_k=top_k_after_source_rrf,
                category_to_min_final_count=CATEGORY_TO_MIN_FINAL_COUNT_AFTER_RRF,
            )
            category_minimum_seconds += perf_counter() - category_step_start

        reciprocal_rank_fusion_output_for_one_source["results"].append({
            "email": first_encoder_result["email"],
            "query_type_to_rewritten_queries": first_encoder_result["query_type_to_rewritten_queries"],
            "reranker_query": first_encoder_result["reranker_query"],
            "anonymized_request": first_encoder_result.get("anonymized_request"),
            "retrieval_failed": False,
            "retrieval_results": [
                {**selected_chunk, "rank": rank}
                for rank, selected_chunk in enumerate(selected_chunks, start=1)
            ],
        })

    reciprocal_rank_fusion_output_for_one_source["timings"] = {
        "rrf_seconds": rrf_seconds,
        "category_minimum_seconds": category_minimum_seconds,
        "total_seconds": perf_counter() - rrf_start,
    }
    return reciprocal_rank_fusion_output_for_one_source

########################################################
# Helper 18: Build reranker output from source results #
########################################################
def build_reranker_output_from_source_results(
        base_data_variant,
        source_name_to_rrf_output,
        reranker_name,
        top_k_after_rerank,
        result_record_metadata,
        ):
    source_names = list(source_name_to_rrf_output)
    first_source_output = source_name_to_rrf_output[source_names[0]]
    reranker_output = {
        **result_record_metadata,
        "base_data_variant": base_data_variant,
        "top_k": top_k_after_rerank,
        "reranker_name": reranker_name,
        "timings": {},
        "n_failed_emails": 0,
        "results": [],
    }

    reranker_seconds = 0.0
    reranker_start = perf_counter()
    run_encoder_gpu_reranker = None
    if reranker_name is not None:
        run_encoder_gpu_reranker = get_run_encoder_gpu_reranker()

    for email_index, first_source_result in enumerate(first_source_output["results"]):
        source_results = {
            source_name: source_rrf["results"][email_index]
            for source_name, source_rrf in source_name_to_rrf_output.items()
        }
        successful_source_results = {
            source_name: source_result
            for source_name, source_result in source_results.items()
            if not source_result["retrieval_failed"]
        }

        if not successful_source_results:
            reranker_output["n_failed_emails"] += 1
            reranker_output["results"].append({
                "email": first_source_result["email"],
                "query_type_to_rewritten_queries": first_source_result["query_type_to_rewritten_queries"],
                "reranker_query": first_source_result["reranker_query"],
                "anonymized_request": first_source_result.get("anonymized_request"),
                "retrieval_failed": True,
                "retrieval_results": [],
            })
            continue

        source_selected_chunks = []
        for source_name, source_result in successful_source_results.items():
            source_selected_chunks.extend([
                {
                    **selected_chunk,
                    "source": source_name,
                    "source_rank": selected_chunk["rank"],
                }
                for selected_chunk in source_result["retrieval_results"]
            ])

        if not source_selected_chunks:
            reranked_chunks = []
        elif reranker_name is None:
            reranked_chunks = source_selected_chunks
        else:
            reranker_query = first_source_result["reranker_query"]
            if not reranker_query:
                print(
                    "build_reranker_output: missing reranker query"
                    f" | data variant {base_data_variant}"
                    f" | email index {email_index}"
                )
                reranker_output["n_failed_emails"] += 1
                reranker_output["results"].append({
                    "email": first_source_result["email"],
                    "query_type_to_rewritten_queries": first_source_result["query_type_to_rewritten_queries"],
                    "reranker_query": first_source_result["reranker_query"],
                    "anonymized_request": first_source_result.get("anonymized_request"),
                    "retrieval_failed": True,
                    "retrieval_results": [],
                })
                continue
            reranker_chunk_texts = [
                get_text_from_payload(selected_chunk["payload"])
                for selected_chunk in source_selected_chunks
            ]
            reranker_step_start = perf_counter()
            reranker_scores = run_encoder_gpu_reranker.remote(
                reranker_name,
                reranker_query,
                reranker_chunk_texts,
            )
            reranker_seconds += perf_counter() - reranker_step_start
            if len(reranker_scores) != len(source_selected_chunks):
                print(
                    "build_reranker_output: reranker returned wrong number of scores"
                    f" | data variant {base_data_variant}"
                    f" | email index {email_index}"
                    f" | expected {len(source_selected_chunks)}"
                    f" | actual {len(reranker_scores)}"
                )
                reranker_output["n_failed_emails"] += 1
                reranker_output["results"].append({
                    "email": first_source_result["email"],
                    "query_type_to_rewritten_queries": first_source_result["query_type_to_rewritten_queries"],
                    "reranker_query": first_source_result["reranker_query"],
                    "anonymized_request": first_source_result.get("anonymized_request"),
                    "retrieval_failed": True,
                    "retrieval_results": [],
                })
                continue
            reranked_chunks = [
                {**selected_chunk, "reranker_score": reranker_score}
                for selected_chunk, reranker_score in zip(source_selected_chunks, reranker_scores)
            ]
            reranked_chunks.sort(
                key=lambda selected_chunk: selected_chunk["reranker_score"],
                reverse=True,
            )

        if top_k_after_rerank is not None:
            reranked_chunks = reranked_chunks[:top_k_after_rerank]

        reranker_output["results"].append({
            "email": first_source_result["email"],
            "query_type_to_rewritten_queries": first_source_result["query_type_to_rewritten_queries"],
            "reranker_query": first_source_result["reranker_query"],
            "anonymized_request": first_source_result.get("anonymized_request"),
            "retrieval_failed": False,
            "retrieval_results": [
                {**selected_chunk, "rank": rank}
                for rank, selected_chunk in enumerate(reranked_chunks, start=1)
            ],
        })

    reranker_output["timings"] = {
        "reranker_seconds": reranker_seconds,
        "total_seconds": perf_counter() - reranker_start,
    }
    return reranker_output

###########################################################
# Helper 19: Run retrieval pipeline from rewritten emails #
###########################################################
def run_retrieval_pipeline_from_rewritten_emails(
        rewritten_emails,
        base_data_variant_to_source_to_encoder_settings,
        top_k_per_query,
        top_k_after_query_fusion,
        use_max_similarity_query_fusion_before_rrf=True,
        result_record_metadata=None,
        top_k_after_source_rrf=None,
        top_k_after_rerank=None,
        reranker_name=None,
        ):
    result_record_metadata = result_record_metadata or {}
    base_data_variant_to_source_to_encoder_output = {}
    base_data_variant_to_source_to_rrf_output = {}
    base_data_variant_to_reranker_output = {}

    for base_data_variant, source_to_encoder_settings in base_data_variant_to_source_to_encoder_settings.items():
        base_data_variant_to_source_to_encoder_output[base_data_variant] = {}
        base_data_variant_to_source_to_rrf_output[base_data_variant] = {}

        for source_name, encoder_name_to_settings in source_to_encoder_settings.items():
            base_data_variant_to_source_to_encoder_output[base_data_variant][source_name] = {}
            for encoder_name, encoder_settings in encoder_name_to_settings.items():
                encoder_name_to_output = run_single_encoder_retrieval(
                    rewritten_emails=rewritten_emails,
                    base_data_variant=base_data_variant,
                    source_name=source_name,
                    collection_name=encoder_settings["collection_name"],
                    encoder_name=encoder_name,
                    encoder_runtime_config=encoder_settings,
                    top_k_per_query=top_k_per_query,
                    top_k_after_query_fusion=top_k_after_query_fusion,
                    use_max_similarity_query_fusion_before_rrf=use_max_similarity_query_fusion_before_rrf,
                    result_record_metadata=result_record_metadata,
                )
                base_data_variant_to_source_to_encoder_output[base_data_variant][source_name].update(
                    encoder_name_to_output
                )

            if not encoder_name_to_settings:
                continue

            first_encoder_settings = next(iter(encoder_name_to_settings.values()))
            encoder_name_to_rrf_weight = {}
            for encoder_name, encoder_settings in encoder_name_to_settings.items():
                if (
                        "rrf_weight" not in encoder_settings
                        and "rrf_weights" not in encoder_settings
                        ):
                    continue
                embedding_kinds = EMBEDDING_ENCODERS[encoder_name]["embedding_kinds"]
                for embedding_kind in embedding_kinds:
                    output_name = (
                        encoder_name
                        if len(embedding_kinds) == 1
                        else f"{encoder_name}_{embedding_kind}"
                    )
                    encoder_name_to_rrf_weight[output_name] = (
                        encoder_settings.get("rrf_weights", {}).get(
                            embedding_kind,
                            encoder_settings.get(
                                "rrf_weight",
                                DEFAULT_RRF_ENCODER_WEIGHT,
                            ),
                        )
                    )
            base_data_variant_to_source_to_rrf_output[base_data_variant][source_name] = build_reciprocal_rank_fusion_output_for_one_source(
                base_data_variant=base_data_variant,
                source_name=source_name,
                collection_name=first_encoder_settings["collection_name"],
                encoder_name_to_output=base_data_variant_to_source_to_encoder_output[base_data_variant][source_name],
                encoder_name_to_rrf_weight=encoder_name_to_rrf_weight,
                top_k_after_source_rrf=top_k_after_source_rrf,
                use_max_similarity_query_fusion_before_rrf=use_max_similarity_query_fusion_before_rrf,
                result_record_metadata=result_record_metadata,
            )

        if base_data_variant_to_source_to_rrf_output[base_data_variant]:
            base_data_variant_to_reranker_output[base_data_variant] = build_reranker_output_from_source_results(
                base_data_variant=base_data_variant,
                source_name_to_rrf_output=base_data_variant_to_source_to_rrf_output[base_data_variant],
                reranker_name=reranker_name,
                top_k_after_rerank=top_k_after_rerank,
                result_record_metadata=result_record_metadata,
            )

    return {
        "base_data_variant_to_source_to_encoder_output": (
            base_data_variant_to_source_to_encoder_output
        ),
        "base_data_variant_to_source_to_rrf_output": (
            base_data_variant_to_source_to_rrf_output
        ),
        "base_data_variant_to_reranker_output": base_data_variant_to_reranker_output,
    }

#####################################
# Helper 20: Run retrieval pipeline #
#####################################
def run_retrieval_pipeline(
        emails,
        base_data_variant_to_source_to_encoder_settings,
        top_k_per_query,
        top_k_after_query_fusion,
        use_max_similarity_query_fusion_before_rrf=True,
        n_max_queries=None,
        result_record_metadata=None,
        max_query_rewrite_failure_rate=None,
        top_k_after_source_rrf=None,
        top_k_after_rerank=None,
        reranker_name=None,
        log_prefix="run_data_variant_eval",
        use_runtime_query_rewriter_tokenizer=False,
        max_concurrent_query_rewrites=None,
        ):
    rewrite_summary = asyncio.run(
        rewrite_emails_async(
            emails=emails,
            n_max_queries=n_max_queries,
            log_prefix=log_prefix,
            use_runtime_query_rewriter_tokenizer=use_runtime_query_rewriter_tokenizer,
            max_concurrent_query_rewrites=max_concurrent_query_rewrites,
        )
    )
    n_query_rewrite_failures = (
        rewrite_summary["n_query_rewriter_exceptions"]
        + rewrite_summary["n_empty_query_rewrite_outputs"]
        + rewrite_summary["n_no_usable_query_rewrite_outputs"]
        + rewrite_summary["n_no_reranker_query_outputs"]
        + rewrite_summary["n_no_anonymized_request_outputs"]
    )
    query_rewrite_failure_rate = (
        0.0 if not emails else n_query_rewrite_failures / len(emails)
    )
    if (
            max_query_rewrite_failure_rate is not None
            and query_rewrite_failure_rate > max_query_rewrite_failure_rate):
        raise RuntimeError(
            f"{log_prefix}: query rewrite failure rate exceeded threshold:\n"
            f"\tfailure rate: {query_rewrite_failure_rate:.2%}\n"
            f"\tmax failure rate: {max_query_rewrite_failure_rate:.2%}"
        )
    retrieval_summary = run_retrieval_pipeline_from_rewritten_emails(
        rewritten_emails=rewrite_summary["rewritten_emails"],
        base_data_variant_to_source_to_encoder_settings=base_data_variant_to_source_to_encoder_settings,
        top_k_per_query=top_k_per_query,
        top_k_after_query_fusion=top_k_after_query_fusion,
        use_max_similarity_query_fusion_before_rrf=use_max_similarity_query_fusion_before_rrf,
        result_record_metadata=result_record_metadata,
        top_k_after_source_rrf=top_k_after_source_rrf,
        top_k_after_rerank=top_k_after_rerank,
        reranker_name=reranker_name,
    )
    return {**rewrite_summary, **retrieval_summary}
