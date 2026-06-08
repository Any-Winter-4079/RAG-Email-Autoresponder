import json
import sys
from datetime import datetime
from os.path import abspath, dirname
from pathlib import Path

import modal

# modal run eval/run_data_variant_eval.py

project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

from config.eval import (
    DATA_VARIANT_CONTEXT_EMAILS_MODE,
    DATA_VARIANT_N_EVAL_SAMPLES_PER_FOLDER_URI,
    DATA_VARIANT_TEST_SPLIT_NAME,
    MAX_QUERY_REWRITE_ERROR_RATE_THRES,
    QUERY_REWRITE_CACHE_DIR,
    RERANKER_NAME,
    RESULTS_DIR_NAME,
    TOP_K_AFTER_QUERY_FUSION,
    TOP_K_AFTER_RERANK,
    TOP_K_AFTER_SOURCE_RRF,
    TOP_K_PER_QUERY,
    USE_MAX_SIMILARITY_QUERY_FUSION_BEFORE_RRF,
)
from config.modal_apps import DATA_VARIANT_EVAL_APP_NAME
from helpers.eval import (
    attach_split_samples_to_retrieval_output,
    build_base_data_variant_to_source_to_encoder_settings,
    get_n_empty_result_emails,
    load_or_create_query_rewrite_data,
    save_query_rewrite_summary_plot,
    save_retrieval_summary_plot,
    write_eval_output_to_file,
)
from helpers.retrieval_pipeline import (
    run_retrieval_pipeline_from_rewritten_emails,
)

app = modal.App(DATA_VARIANT_EVAL_APP_NAME)

@app.local_entrypoint()
def run_data_variant_eval():
    # load or create query rewrites
    query_rewrite_data = load_or_create_query_rewrite_data(
        project_root=project_root,
        query_rewrite_cache_dir=QUERY_REWRITE_CACHE_DIR,
        split_name=DATA_VARIANT_TEST_SPLIT_NAME,
        context_emails_mode=DATA_VARIANT_CONTEXT_EMAILS_MODE,
        n_eval_samples_per_folder_uri=DATA_VARIANT_N_EVAL_SAMPLES_PER_FOLDER_URI,
        max_query_rewrite_failure_rate=MAX_QUERY_REWRITE_ERROR_RATE_THRES,
        log_prefix="run_data_variant_eval",
    )
    split_samples = query_rewrite_data["split_samples"]
    folder_uri_to_n_split_samples = query_rewrite_data["folder_uri_to_n_split_samples"]
    rewrite_summary = query_rewrite_data["rewrite_summary"]
    request_entries = query_rewrite_data["request_entries"]
    no_requests = query_rewrite_data["no_requests"]
    rewritten_split_samples = query_rewrite_data["rewritten_split_samples"]
    print(
        "run_data_variant_eval: selected samples per folder uri:\n"
        f"\tn samples per folder uri: {DATA_VARIANT_N_EVAL_SAMPLES_PER_FOLDER_URI}\n"
        f"\tcontext emails mode: {DATA_VARIANT_CONTEXT_EMAILS_MODE}\n"
        f"\tn total selected samples: {len(split_samples)}\n"
        + "\n".join(
            f"\t{folder_uri}: {n_selected}"
            for folder_uri, n_selected in folder_uri_to_n_split_samples.items()
        )
    )
    cache_action = (
        "cached"
        if query_rewrite_data["did_create_cache"]
        else "loaded"
    )
    print(
        f"run_data_variant_eval: {cache_action} usable query rewrites:\n"
        f"\tpath: {query_rewrite_data['query_rewrite_cache_path'].relative_to(project_root)}\n"
        f"\tn request samples: {len(request_entries)}"
    )
    print(
        f"run_data_variant_eval: {cache_action} no-request samples:\n"
        f"\tpath: {query_rewrite_data['no_requests_cache_path'].relative_to(project_root)}\n"
        f"\tn no-request samples: {len(no_requests)}"
    )

    # build retrieval settings
    base_data_variant_to_source_to_encoder_settings = (
        build_base_data_variant_to_source_to_encoder_settings()
    )

    # create results directory
    script_name = "run_data_variant_eval"
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = (
        Path(project_root)
        / "eval"
        / RESULTS_DIR_NAME
        / script_name
        / DATA_VARIANT_TEST_SPLIT_NAME
        / timestamp
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    # store query rewrite outputs
    if query_rewrite_data["did_create_cache"]:
        save_query_rewrite_summary_plot(
            n_emails=len(split_samples),
            n_query_rewriter_exceptions=rewrite_summary["n_query_rewriter_exceptions"],
            n_empty_query_rewrite_outputs=rewrite_summary["n_empty_query_rewrite_outputs"],
            n_no_usable_query_rewrite_outputs=rewrite_summary["n_no_usable_query_rewrite_outputs"],
            n_no_reranker_query_outputs=rewrite_summary["n_no_reranker_query_outputs"],
            n_no_request_emails=len(rewrite_summary["no_request_emails"]),
            n_rewritten_emails=len(rewrite_summary["rewritten_emails"]),
            n_duplicate_queries_removed=rewrite_summary["n_duplicate_queries_removed"],
            n_query_cap_hits=rewrite_summary["n_query_cap_hits"],
            n_capped_queries_removed=rewrite_summary["n_capped_queries_removed"],
            output_path=results_dir / "query_rewrite_summary.png",
        )
    no_requests_path = results_dir / "no_requests.json"
    with open(no_requests_path, "w", encoding="utf-8") as no_requests_file:
        json.dump(no_requests, no_requests_file, ensure_ascii=False, indent=2)
    print(
        "run_data_variant_eval: saved no-request samples:\n"
        f"\tpath: {no_requests_path.relative_to(project_root)}\n"
        f"\tn no-request samples: {len(no_requests)}"
    )

    # run retrieval on rewritten emails
    retrieval_summary = run_retrieval_pipeline_from_rewritten_emails(
        rewritten_emails=rewrite_summary["rewritten_emails"],
        base_data_variant_to_source_to_encoder_settings=base_data_variant_to_source_to_encoder_settings,
        top_k_per_query=TOP_K_PER_QUERY,
        top_k_after_query_fusion=TOP_K_AFTER_QUERY_FUSION,
        use_max_similarity_query_fusion_before_rrf=USE_MAX_SIMILARITY_QUERY_FUSION_BEFORE_RRF,
        result_record_metadata={"eval_split": DATA_VARIANT_TEST_SPLIT_NAME},
        top_k_after_source_rrf=TOP_K_AFTER_SOURCE_RRF,
        top_k_after_rerank=TOP_K_AFTER_RERANK,
        reranker_name=RERANKER_NAME,
    )

    # plot retrieval outputs
    label_to_n_failed_emails = {}
    label_to_n_empty_result_emails = {}
    for base_data_variant, source_to_encoder_output in (
            retrieval_summary["base_data_variant_to_source_to_encoder_output"].items()):
        for source_name, encoder_output_by_name in source_to_encoder_output.items():
            for encoder_name, encoder_output in encoder_output_by_name.items():
                output_label = f"{base_data_variant} / {source_name} / {encoder_name}"
                label_to_n_failed_emails[output_label] = encoder_output["n_failed_emails"]
                label_to_n_empty_result_emails[output_label] = get_n_empty_result_emails(
                    encoder_output
                )
    for base_data_variant, source_to_rrf_output in (
            retrieval_summary["base_data_variant_to_source_to_rrf_output"].items()):
        for source_name, rrf_output in source_to_rrf_output.items():
            output_label = f"{base_data_variant} / {source_name} / rrf"
            label_to_n_failed_emails[output_label] = rrf_output["n_failed_emails"]
            label_to_n_empty_result_emails[output_label] = get_n_empty_result_emails(rrf_output)
    for base_data_variant, reranker_output in (
            retrieval_summary["base_data_variant_to_reranker_output"].items()):
        output_label = f"{base_data_variant} / reranker"
        label_to_n_failed_emails[output_label] = reranker_output["n_failed_emails"]
        label_to_n_empty_result_emails[output_label] = get_n_empty_result_emails(reranker_output)

    save_retrieval_summary_plot(
        label_to_n_failed_emails=label_to_n_failed_emails,
        label_to_n_empty_result_emails=label_to_n_empty_result_emails,
        output_path=results_dir / "retrieval_summary.png",
    )

    # store retrieval outputs
    for base_data_variant, source_to_encoder_output in (
            retrieval_summary["base_data_variant_to_source_to_encoder_output"].items()):
        for source_name, encoder_output_by_name in source_to_encoder_output.items():
            source_results_dir = results_dir / base_data_variant / source_name
            source_results_dir.mkdir(parents=True, exist_ok=True)
            for encoder_name, encoder_output in encoder_output_by_name.items():
                write_eval_output_to_file(
                    data_variant_results_dir=source_results_dir,
                    output_name=encoder_name,
                    eval_output=attach_split_samples_to_retrieval_output(
                        encoder_output,
                        rewritten_split_samples,
                    ),
                    data_variant=f"{base_data_variant}/{source_name}",
                )

    for base_data_variant, source_to_rrf_output in (
            retrieval_summary["base_data_variant_to_source_to_rrf_output"].items()):
        for source_name, rrf_output in source_to_rrf_output.items():
            source_results_dir = results_dir / base_data_variant / source_name
            source_results_dir.mkdir(parents=True, exist_ok=True)
            write_eval_output_to_file(
                data_variant_results_dir=source_results_dir,
                output_name="rrf",
                eval_output=attach_split_samples_to_retrieval_output(
                    rrf_output,
                    rewritten_split_samples,
                ),
                data_variant=f"{base_data_variant}/{source_name}",
            )

    for base_data_variant, reranker_output in (
            retrieval_summary["base_data_variant_to_reranker_output"].items()):
        data_variant_results_dir = results_dir / base_data_variant
        data_variant_results_dir.mkdir(parents=True, exist_ok=True)
        write_eval_output_to_file(
            data_variant_results_dir=data_variant_results_dir,
            output_name="reranker",
            eval_output=attach_split_samples_to_retrieval_output(
                reranker_output,
                rewritten_split_samples,
            ),
            data_variant=base_data_variant,
        )

    print("run_data_variant_eval: done")
