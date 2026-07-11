import json
from pathlib import Path
from config.m3 import HAS_FINETUNED_M3

#####################################################
# Helper 1: Get query-rewrite cache suffix for mode #
#####################################################
def get_context_emails_mode_suffix(context_emails_mode):
    return f"_{context_emails_mode}"

###############################################################
# Helper 2: Get query-rewrite cache sample-cap suffix for run #
###############################################################
def get_n_eval_samples_per_folder_uri_suffix(n_eval_samples_per_folder_uri):
    if n_eval_samples_per_folder_uri is None:
        return "_nall"
    return f"_n{n_eval_samples_per_folder_uri}"

##########################
# Helper 3: Resolve path #
##########################
def resolve_path(
        search_root,
        candidate_pattern,
        missing_candidates_message,
        candidate_key,
        candidate_filter=None,
        ):
    candidate_paths = [
        path
        for path in search_root.glob(candidate_pattern)
        if path.is_file() and (candidate_filter(path) if candidate_filter else True)
    ]
    if not candidate_paths:
        raise FileNotFoundError(missing_candidates_message)

    return max(candidate_paths, key=candidate_key)

##############################################
# Helper 4: Resolve query rewrite cache path #
##############################################
def resolve_query_rewrite_cache_path(
        project_root,
        query_rewrite_cache_dir,
        split_name,
        context_emails_mode,
        n_eval_samples_per_folder_uri,
        configured_cache_filename=None,
        ):
    cache_dir = Path(project_root) / query_rewrite_cache_dir
    filter_mode_suffix = get_context_emails_mode_suffix(context_emails_mode)
    n_eval_samples_suffix = get_n_eval_samples_per_folder_uri_suffix(
        n_eval_samples_per_folder_uri
    )
    cache_pattern = f"{split_name}{filter_mode_suffix}_*{n_eval_samples_suffix}.json"
    if configured_cache_filename:
        return cache_dir / configured_cache_filename

    return resolve_path(
        search_root=cache_dir,
        candidate_pattern=cache_pattern,
        missing_candidates_message=(
            "resolve_query_rewrite_cache_path: no query rewrite cache files found:\n"
            f"\tdir: {cache_dir}\n"
            f"\tpattern: {cache_pattern}"
        ),
        candidate_key=lambda cache_path: cache_path.stat().st_mtime,
        candidate_filter=lambda cache_path: (
            not cache_path.name.endswith("_reranker_queries.json")
            and not cache_path.name.endswith("_no_requests.json")
        ),
    )

#############################################################
# Helper 5: Resolve dumped collection payload file for eval #
#############################################################
def resolve_dumped_collection_payloads_path(
        project_root,
        collection_name,
        dump_script_name,
        dump_timestamp=None,
        output_name="dump",
        ):
    dumps_root = (
        Path(project_root)
        / "eval"
        / "results"
        / dump_script_name
    )
    dump_filename = f"{output_name}.json"
    if dump_timestamp:
        return dumps_root / dump_timestamp / collection_name / dump_filename

    return resolve_path(
        search_root=dumps_root,
        candidate_pattern=f"*/{collection_name}/{dump_filename}",
        missing_candidates_message=(
            "resolve_dumped_collection_payloads_path: no dumped collection payload files found:\n"
            f"\tdir: {dumps_root}\n"
            f"\tcollection: {collection_name}\n"
            f"\toutput_name: {output_name}"
        ),
        candidate_key=lambda dump_path: dump_path.parent.parent.name,
    )

######################################################
# Helper 6: Resolve oracle discriminator result path #
######################################################
def resolve_oracle_discriminator_path(
        project_root,
        split_name,
        variant,
        timestamp=None,
        data_sources=None,
        input_mode=None,
        ):
    oracle_results_dir = (
        Path(project_root)
        / "eval"
        / "results"
        / "run_oracle_discriminator"
    )
    split_results_dir = oracle_results_dir / split_name
    if timestamp is not None:
        return split_results_dir / timestamp / variant / "oracle_discriminator.json"

    def candidate_matches_data_sources(path):
        if data_sources is None and input_mode is None:
            return True
        with open(path, "r", encoding="utf-8") as oracle_file:
            oracle_output = json.load(oracle_file)
        oracle_input_metadata = oracle_output.get("oracle_input_metadata") or {}
        if data_sources is not None and oracle_input_metadata.get("data_sources") != data_sources:
            return False
        oracle_input_mode = oracle_output.get(
            "oracle_input_mode",
            oracle_input_metadata.get("input_mode"),
        )
        if input_mode is not None and oracle_input_mode != input_mode:
            return False
        return True

    return resolve_path(
        search_root=split_results_dir,
        candidate_pattern=f"*/{variant}/oracle_discriminator.json",
        missing_candidates_message=(
            "resolve_oracle_discriminator_path: no oracle discriminator results found:\n"
            f"\tdir: {split_results_dir}\n"
            f"\tsplit: {split_name}\n"
            f"\tvariant: {variant}\n"
            f"\tdata_sources: {data_sources}\n"
            f"\tinput_mode: {input_mode}"
        ),
        candidate_key=lambda path: path.parent.parent.name,
        candidate_filter=candidate_matches_data_sources,
    )

#######################################################
# Helper 7: Resolve run_data_variant_eval output file #
#######################################################
def resolve_data_variant_eval_output_path(
        project_root,
        split_name,
        variant,
        output_name,
        source_name=None,
        timestamp=None,
        ):
    eval_results_dir = (
        Path(project_root)
        / "eval"
        / "results"
        / "run_data_variant_eval"
    )
    output_filename = f"{output_name}.json"
    split_results_dir = eval_results_dir / split_name
    if timestamp is not None:
        if source_name is None:
            return split_results_dir / timestamp / variant / output_filename
        return split_results_dir / timestamp / variant / source_name / output_filename

    if source_name is None:
        candidate_pattern = f"*/{variant}/{output_filename}"
    else:
        candidate_pattern = f"*/{variant}/{source_name}/{output_filename}"

    return resolve_path(
        search_root=split_results_dir,
        candidate_pattern=candidate_pattern,
        missing_candidates_message=(
            "resolve_data_variant_eval_output_path: no eval output files found:\n"
            f"\tdir: {split_results_dir}\n"
            f"\tsplit: {split_name}\n"
            f"\tvariant: {variant}\n"
            f"\tsource_name: {source_name}\n"
            f"\toutput_name: {output_name}"
        ),
        candidate_key=(
            (lambda path: path.parent.parent.name)
            if source_name is None
            else (lambda path: path.parent.parent.parent.name)
        ),
    )

###############################################
# Helper 8: Get Qdrant collection name to use #
###############################################
def get_qdrant_collection_name(collection_name):
    post_sft_collection_suffix = "_post_sft"
    if HAS_FINETUNED_M3 and not collection_name.endswith(post_sft_collection_suffix):
        return f"{collection_name}{post_sft_collection_suffix}"
    return collection_name

###################################################
# Helper 9: Get rerank text from collection point #
###################################################
def get_text_from_payload(payload):
    if "question" in payload and "answer" not in payload:
        return payload["question"]
    if "question" in payload and "answer" in payload:
        return f"Q: {payload['question']}\nA: {payload['answer']}"
    return payload["text"]
