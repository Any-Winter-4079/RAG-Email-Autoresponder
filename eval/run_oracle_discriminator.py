import sys
from os.path import dirname, abspath

# NOTE: configure:
# ORACLE_DISCRIMINATOR_VARIANT (for data variant)
# ORACLE_DISCRIMINATOR_DATA_SOURCES = ["web", "email"] (for data sources,
# setting ["web"] for raw_chunks and manually_cleaned_chunks)
# ORACLE_DISCRIMINATOR_RETRIEVAL_OUTPUT_FILE_NAMES (for encoder names, rrf or reranker)
# if ORACLE_DISCRIMINATOR_INPUT_MODE = "retrieval"
# in config/llm_judge.py before running:
# modal run eval/run_oracle_discriminator.py

# encoder names are:
# bm25 (from bm25.json)
# splade (from splade.json)
# bge_m3_sparse (or bge_m3_muia_sparse) (from bge_m3_sparse.json/bge_m3_muia_sparse.json)
# bge_m3_dense (or bge_m3_muia_dense) (from bge_m3_dense.json/bge_m3_muia_dense.json)
# qwen3_embedding_0_6b (from qwen3_embedding_0_6b.json)
# jina_v5_text_small (from jina_v5_text_small.json)

# rrf name is:
# rrf (from rrf.json)

# reranker name is:
# reranker (from reranker.json)

# noting the disambiguation for m3 via _sparse, _dense

# NOTE: configure:
# ORACLE_DISCRIMINATOR_DATA_SOURCES = ["web", "email"] (for data sources)
# if ORACLE_DISCRIMINATOR_INPUT_MODE = "corpus"
# in config/llm_judge.py before running:
# modal run eval/run_oracle_discriminator.py
project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

from collections import Counter
from datetime import datetime
from pathlib import Path

import modal

from config.eval import (
    DATA_VARIANT_CONTEXT_EMAILS_MODE,
    DATA_VARIANT_TEST_SPLIT_NAME,
    QUERY_REWRITE_CACHE_DIR,
    RESULTS_DIR_NAME,
)
from config.llm_judge import (
    ORACLE_DISCRIMINATOR_COLLECTION_DUMP_TIMESTAMP,
    ORACLE_DISCRIMINATOR_DUMP_COLLECTION_PAYLOADS_SCRIPT_NAME,
    ORACLE_DISCRIMINATOR_DATA_SOURCES,
    ORACLE_DISCRIMINATOR_HUMAN_ALIGNMENT_MODE,
    ORACLE_DISCRIMINATOR_INPUT_MODE,
    ORACLE_DISCRIMINATOR_MAX_SAMPLES_PER_RUN,
    ORACLE_DISCRIMINATOR_N_EVAL_SAMPLES_PER_FOLDER_URI,
    ORACLE_DISCRIMINATOR_QUERY_REWRITE_CACHE_FILENAME,
    ORACLE_DISCRIMINATOR_RETRIEVAL_OUTPUT_FILE_NAMES,
    ORACLE_DISCRIMINATOR_RETRIEVAL_TIMESTAMP,
    ORACLE_DISCRIMINATOR_TOP_K_PER_RETRIEVAL_OUTPUT,
    ORACLE_DISCRIMINATOR_VARIANT,
)
from config.human_review import HUMAN_REVIEW_N_EVAL_SAMPLES_PER_FOLDER_URI
from config.decoder import MODEL_PROFILES, LLM_JUDGE_PROFILE
from config.modal_apps import LLM_JUDGE_APP_NAME
from config.modal_functions import RUN_LLM_JUDGE_FUNCTION_NAME
from helpers.eval import (
    attach_selected_chunks_to_discriminator_result,
    load_oracle_inputs,
    write_eval_output_to_file,
)

app = modal.App("oracle-discriminator-eval")

@app.local_entrypoint()
def run_oracle_discriminator():
    # load judge function
    judge_profile = MODEL_PROFILES[LLM_JUDGE_PROFILE]
    run_llm_judge = modal.Function.from_name(
        LLM_JUDGE_APP_NAME,
        RUN_LLM_JUDGE_FUNCTION_NAME,
    )

    # load oracle inputs
    n_eval_samples_per_folder_uri = (
        HUMAN_REVIEW_N_EVAL_SAMPLES_PER_FOLDER_URI
        if ORACLE_DISCRIMINATOR_HUMAN_ALIGNMENT_MODE
        else ORACLE_DISCRIMINATOR_N_EVAL_SAMPLES_PER_FOLDER_URI
    )
    oracle_entries, oracle_input_metadata = load_oracle_inputs(
        input_mode=ORACLE_DISCRIMINATOR_INPUT_MODE,
        project_root=project_root,
        query_rewrite_cache_dir=QUERY_REWRITE_CACHE_DIR,
        split_name=DATA_VARIANT_TEST_SPLIT_NAME,
        context_emails_mode=DATA_VARIANT_CONTEXT_EMAILS_MODE,
        n_eval_samples_per_folder_uri=n_eval_samples_per_folder_uri,
        configured_cache_filename=ORACLE_DISCRIMINATOR_QUERY_REWRITE_CACHE_FILENAME,
        max_samples=ORACLE_DISCRIMINATOR_MAX_SAMPLES_PER_RUN,
        data_variant=ORACLE_DISCRIMINATOR_VARIANT,
        dump_script_name=ORACLE_DISCRIMINATOR_DUMP_COLLECTION_PAYLOADS_SCRIPT_NAME,
        dump_timestamp=ORACLE_DISCRIMINATOR_COLLECTION_DUMP_TIMESTAMP,
        data_sources=ORACLE_DISCRIMINATOR_DATA_SOURCES,
        retrieval_output_names=ORACLE_DISCRIMINATOR_RETRIEVAL_OUTPUT_FILE_NAMES,
        retrieval_timestamp=ORACLE_DISCRIMINATOR_RETRIEVAL_TIMESTAMP,
        top_k_per_retrieval_output=ORACLE_DISCRIMINATOR_TOP_K_PER_RETRIEVAL_OUTPUT,
    )
    print(
        "run_oracle_discriminator: running oracle judge:\n"
        f"\tinput mode: {ORACLE_DISCRIMINATOR_INPUT_MODE}\n"
        f"\tdata variant: {ORACLE_DISCRIMINATOR_VARIANT}\n"
        f"\tdata sources: {ORACLE_DISCRIMINATOR_DATA_SOURCES}\n"
        f"\tretrieval output file names: {ORACLE_DISCRIMINATOR_RETRIEVAL_OUTPUT_FILE_NAMES}\n"
        f"\ttop k per retrieval output: {ORACLE_DISCRIMINATOR_TOP_K_PER_RETRIEVAL_OUTPUT}\n"
        f"\thuman alignment mode: {ORACLE_DISCRIMINATOR_HUMAN_ALIGNMENT_MODE}\n"
        f"\tn samples per folder uri: {n_eval_samples_per_folder_uri}\n"
        f"\tn samples: {len(oracle_entries)}"
    )

    # run oracle judge
    results = []
    for oracle_entry in oracle_entries:
        sample = oracle_entry["sample"]
        reranker_query = oracle_entry["reranker_query"]
        oracle_query = oracle_entry.get("anonymized_request") or reranker_query
        chunks = oracle_entry["chunks"]
        if not oracle_query or not chunks:
            continue
        discriminator_result = run_llm_judge.remote(oracle_query, chunks)
        if discriminator_result is None:
            results.append({
                "sample": sample,
                "reranker_query": reranker_query,
                "anonymized_request": oracle_entry.get("anonymized_request"),
                "generation_failed": True,
                "discriminator_result": None,
            })
            continue
        attach_selected_chunks_to_discriminator_result(
            discriminator_result=discriminator_result,
            id_to_chunk=oracle_entry["id_to_chunk"],
        )
        results.append({
            "sample": sample,
            "reranker_query": reranker_query,
            "anonymized_request": oracle_entry.get("anonymized_request"),
            "generation_failed": False,
            "discriminator_result": discriminator_result,
        })

    # summarize oracle outputs
    label_counts = Counter(
        result["discriminator_result"]["answerability"]
        for result in results
        if result.get("discriminator_result")
        and result["discriminator_result"].get("answerability") is not None
    )
    n_labeled_results = sum(label_counts.values())
    label_percentages = {
        label: round(100.0 * count / n_labeled_results, 1)
        for label, count in label_counts.items()
    } if n_labeled_results else {}
    subquery_confidences = [
        subquery["confidence"]
        for result in results
        if result.get("discriminator_result")
        for subquery in result["discriminator_result"].get("subqueries") or []
        if subquery.get("confidence") is not None
    ]
    average_subquery_confidence = (
        round(sum(subquery_confidences) / len(subquery_confidences), 3)
        if subquery_confidences
        else None
    )
    oracle_input_metadata = {
        key: [
            str(item.relative_to(project_root))
            if isinstance(item, Path)
            else item
            for item in value
        ] if isinstance(value, list) else (
            str(value.relative_to(project_root))
            if isinstance(value, Path)
            else value
        )
        for key, value in oracle_input_metadata.items()
    }

    # create results directory
    script_name = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = (
        Path(project_root)
        / "eval"
        / RESULTS_DIR_NAME
        / script_name
        / DATA_VARIANT_TEST_SPLIT_NAME
    )
    data_variant_results_dir = results_dir / timestamp / ORACLE_DISCRIMINATOR_VARIANT
    data_variant_results_dir.mkdir(parents=True, exist_ok=True)

    # store oracle outputs
    eval_output = {
        "eval_split": DATA_VARIANT_TEST_SPLIT_NAME,
        "data_variant": ORACLE_DISCRIMINATOR_VARIANT,
        "judge_settings": {
            "provider": judge_profile["provider"],
            "model_name_or_path": judge_profile["model_name_or_path"],
            "enable_thinking": judge_profile["enable_thinking"],
            "reasoning_effort": judge_profile["reasoning_effort"],
            "max_new_tokens": judge_profile["max_new_tokens"],
            "temperature": judge_profile["temperature"],
            "top_p": judge_profile["top_p"],
            "top_k": judge_profile["top_k"],
        },
        "oracle_input_mode": ORACLE_DISCRIMINATOR_INPUT_MODE,
        "oracle_input_metadata": oracle_input_metadata,
        "n_samples": len(results),
        "discriminator_label_counts": dict(label_counts),
        "discriminator_label_percentages": label_percentages,
        "average_subquery_confidence": average_subquery_confidence,
        "results": results,
    }
    write_eval_output_to_file(
        data_variant_results_dir,
        "oracle_discriminator",
        eval_output,
        ORACLE_DISCRIMINATOR_VARIANT,
    )
