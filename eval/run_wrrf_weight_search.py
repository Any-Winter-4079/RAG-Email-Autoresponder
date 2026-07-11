import json
import math
import sys
from os.path import abspath, dirname

# .venv/bin/python eval/run_wrrf_weight_search.py

project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

from helpers.oracle_support import (
    get_answerability_label,
    get_supporting_chunk_data,
    serialize_payload,
)
from helpers.general import (
    resolve_data_variant_eval_output_path,
    resolve_oracle_discriminator_path,
)

DATA_VARIANT = "lm_cleaned_text_chunks"
SPLIT_NAME = "dev"
ORACLE_TIMESTAMP = "2026-06-16_23-50-58"
RETRIEVAL_TIMESTAMP = "2026-06-28_17-43-26"

ENCODERS = [
    "bm25",
    "splade",
    "bge_m3_muia_sparse",
    "bge_m3_muia_dense",
    "qwen3_embedding_0_6b",
    "jina_v5_text_small",
]
DATA_SOURCES = ["web", "email"]
ENCODER_ALIASES = {
    "bge_m3_sparse": "bge_m3_muia_sparse",
    "bge_m3_dense": "bge_m3_muia_dense",
}

N_EVAL_SAMPLES = 302
TEMPERATURES = [1/(t+1) for t in range(20)]
TOP_K_AFTER_SOURCE_RRF = 5
CATEGORY_TO_MIN_FINAL_COUNT_AFTER_RRF = {"master": 2}
RRF_RANK_CONSTANT = 60

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)

def keep_category_minimums_from_ranked_chunks(
        ranked_chunks,
        top_k,
        category_to_min_final_count,
        ):
    plain_top_k_original_chunks = ranked_chunks[:top_k]
    if not category_to_min_final_count:
        return [
            {**ranked_chunk, "selection_reason": "top_k"}
            for ranked_chunk in plain_top_k_original_chunks
        ]

    selected_original_chunks = []
    selected_chunks = []
    for category, min_count in category_to_min_final_count.items():
        category_chunks = [
            ranked_chunk
            for ranked_chunk in ranked_chunks
            if ranked_chunk.get("category") == category
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
        selected_chunks.append({**ranked_chunk, "selection_reason": "top_k"})

    return selected_chunks

def format_weights(encoder_to_weight):
    return ", ".join(
        f"{encoder}={encoder_to_weight[encoder]:.3f}"
        for encoder in ENCODERS
    )

#####################
# Core calculations #
#####################
def build_single_encoder_scores(oracle_results):
    encoder_to_full = {encoder: 0.0 for encoder in ENCODERS}
    encoder_to_partial = {encoder: 0.0 for encoder in ENCODERS}

    for oracle_result in oracle_results:
        ans_label = get_answerability_label(oracle_result, {"1", "0"})
        if ans_label is None:
            continue

        supporting_chunk_data = get_supporting_chunk_data(
            oracle_result=oracle_result,
            encoder_order=ENCODERS,
            data_sources=DATA_SOURCES,
            encoder_aliases=ENCODER_ALIASES,
            include_payload=False,
        )
        if not supporting_chunk_data:
            continue

        supporting_chunk_weight = 1 / len(supporting_chunk_data)
        for supporting_chunk in supporting_chunk_data:
            for encoder in supporting_chunk["encoders"]:
                if ans_label == "1":
                    encoder_to_full[encoder] += supporting_chunk_weight
                else:
                    encoder_to_partial[encoder] += supporting_chunk_weight

    return {
        encoder: encoder_to_full[encoder] + 0.5 * encoder_to_partial[encoder]
        for encoder in ENCODERS
    }

def build_oracle_support_by_sample(oracle_results):
    support_by_sample = []
    for oracle_result in oracle_results:
        ans_label = get_answerability_label(oracle_result, {"1", "0"})
        payload_to_weight = {}
        supporting_chunk_data = get_supporting_chunk_data(
            oracle_result=oracle_result,
            encoder_order=ENCODERS,
            data_sources=DATA_SOURCES,
            encoder_aliases=ENCODER_ALIASES,
            include_payload=True,
        )
        if ans_label is not None and supporting_chunk_data:
            supporting_chunk_weight = 1 / len(supporting_chunk_data)
            for supporting_chunk in supporting_chunk_data:
                payload = supporting_chunk["payload"]
                if payload is not None:
                    payload_to_weight[payload] = (
                        payload_to_weight.get(payload, 0.0)
                        + supporting_chunk_weight
                    )
        support_by_sample.append({
            "ans_label": ans_label,
            "payload_to_weight": payload_to_weight,
        })
    return support_by_sample

def load_encoder_outputs():
    source_to_encoder_to_results = {}
    for source_name in DATA_SOURCES:
        source_to_encoder_to_results[source_name] = {}
        for encoder in ENCODERS:
            output_path = resolve_data_variant_eval_output_path(
                project_root=project_root,
                split_name=SPLIT_NAME,
                variant=DATA_VARIANT,
                output_name=encoder,
                source_name=source_name,
                timestamp=RETRIEVAL_TIMESTAMP,
            )
            output_data = load_json(output_path)
            source_to_encoder_to_results[source_name][encoder] = output_data["results"]
    return source_to_encoder_to_results

def build_query_level_ranked_lists(sample_encoder_results):
    ranked_list_name_to_chunks = {}
    ranked_list_name_to_encoder = {}

    for encoder, encoder_result in sample_encoder_results.items():
        if encoder_result.get("retrieval_failed"):
            continue

        for retrieved_chunk in encoder_result["retrieval_results"]:
            query_type = retrieved_chunk["query_matching_retrieved_chunk"]["query_type"]
            query = retrieved_chunk["query_matching_retrieved_chunk"]["query"]
            ranked_list_name = f"{encoder}::{query_type}::{query}"
            ranked_list_name_to_encoder[ranked_list_name] = encoder
            ranked_list_name_to_chunks.setdefault(ranked_list_name, []).append({
                "payload": serialize_payload(retrieved_chunk["payload"]),
                "category": retrieved_chunk["payload"].get("category"),
                "rank": retrieved_chunk["rank"],
            })

    return ranked_list_name_to_chunks, ranked_list_name_to_encoder

def build_source_ranked_lists_by_sample(source_to_encoder_to_results):
    source_to_sample_ranked_lists = {}
    for source_name in DATA_SOURCES:
        n_samples = len(source_to_encoder_to_results[source_name][ENCODERS[0]])
        source_to_sample_ranked_lists[source_name] = []
        for sample_index in range(n_samples):
            sample_encoder_results = {
                encoder: source_to_encoder_to_results[source_name][encoder][sample_index]
                for encoder in ENCODERS
            }
            source_to_sample_ranked_lists[source_name].append(
                build_query_level_ranked_lists(sample_encoder_results)
            )
    return source_to_sample_ranked_lists

def fuse_ranked_lists_with_weighted_rrf(
        ranked_list_name_to_chunks,
        ranked_list_name_to_encoder,
        encoder_to_weight,
        ):
    payload_to_rrf_candidate = {}
    for ranked_list_name, top_k_chunks in ranked_list_name_to_chunks.items():
        encoder = ranked_list_name_to_encoder[ranked_list_name]
        ranked_list_weight = encoder_to_weight[encoder]
        for retrieved_chunk in top_k_chunks:
            payload_value = retrieved_chunk["payload"]
            if payload_value not in payload_to_rrf_candidate:
                payload_to_rrf_candidate[payload_value] = {
                    "score": 0.0,
                    "payload": payload_value,
                    "category": retrieved_chunk["category"],
                }
            payload_to_rrf_candidate[payload_value]["score"] += (
                ranked_list_weight
                / (RRF_RANK_CONSTANT + retrieved_chunk["rank"])
            )

    return sorted(
        payload_to_rrf_candidate.values(),
        key=lambda retrieved_chunk: retrieved_chunk["score"],
        reverse=True,
    )

def select_source_rrf_chunks(source_ranked_lists, encoder_to_weight):
    ranked_list_name_to_chunks, ranked_list_name_to_encoder = source_ranked_lists
    if not ranked_list_name_to_chunks:
        return []
    if len(ranked_list_name_to_chunks) == 1:
        ranked_chunks = next(iter(ranked_list_name_to_chunks.values()))
    else:
        ranked_chunks = fuse_ranked_lists_with_weighted_rrf(
            ranked_list_name_to_chunks,
            ranked_list_name_to_encoder,
            encoder_to_weight,
        )
    return keep_category_minimums_from_ranked_chunks(
        ranked_chunks,
        top_k=TOP_K_AFTER_SOURCE_RRF,
        category_to_min_final_count=CATEGORY_TO_MIN_FINAL_COUNT_AFTER_RRF,
    )

def get_temperature_weights(single_encoder_scores, temperature):
    scaled_scores = {
        encoder: single_encoder_scores[encoder] / N_EVAL_SAMPLES
        for encoder in ENCODERS
    }
    max_scaled_score = max(scaled_scores.values())
    exp_scores = {
        encoder: math.exp((scaled_scores[encoder] - max_scaled_score) / temperature)
        for encoder in ENCODERS
    }
    exp_score_sum = sum(exp_scores.values())
    return {
        encoder: exp_scores[encoder] / exp_score_sum
        for encoder in ENCODERS
    }

def score_interpretable_weights(
        support_by_sample,
        source_to_sample_ranked_lists,
        encoder_to_weight,
        ):
    full_support_mass = 0.0
    partial_support_mass = 0.0

    for sample_index, support_data in enumerate(support_by_sample):
        selected_payloads = set()
        for source_name in DATA_SOURCES:
            selected_chunks = select_source_rrf_chunks(
                source_to_sample_ranked_lists[source_name][sample_index],
                encoder_to_weight,
            )
            selected_payloads.update(
                selected_chunk["payload"]
                for selected_chunk in selected_chunks
            )

        selected_support_mass = sum(
            support_weight
            for payload, support_weight in support_data["payload_to_weight"].items()
            if payload in selected_payloads
        )
        if support_data["ans_label"] == "1":
            full_support_mass += selected_support_mass
        elif support_data["ans_label"] == "0":
            partial_support_mass += selected_support_mass

    return {
        "full_support_mass": full_support_mass,
        "partial_support_mass": partial_support_mass,
        "aggregate_support_mass": full_support_mass + 0.5 * partial_support_mass,
    }

def run_interpretable_search():
    oracle_path = resolve_oracle_discriminator_path(
        project_root=project_root,
        split_name=SPLIT_NAME,
        variant=DATA_VARIANT,
        timestamp=ORACLE_TIMESTAMP,
    )
    oracle_data = load_json(oracle_path)
    oracle_results = oracle_data["results"]
    source_to_encoder_to_results = load_encoder_outputs()
    source_to_sample_ranked_lists = build_source_ranked_lists_by_sample(
        source_to_encoder_to_results
    )

    single_encoder_scores = build_single_encoder_scores(oracle_results)
    support_by_sample = build_oracle_support_by_sample(oracle_results)

    print("single encoder aggregate scores")
    for encoder in ENCODERS:
        print(f"{encoder}: {single_encoder_scores[encoder]:.3f}")

    print("\ntemperature search")
    best_result = None
    for temperature in TEMPERATURES:
        encoder_to_weight = get_temperature_weights(single_encoder_scores, temperature)
        result = score_interpretable_weights(
            support_by_sample,
            source_to_sample_ranked_lists,
            encoder_to_weight,
        )
        result["temperature"] = temperature
        result["encoder_to_weight"] = encoder_to_weight
        if (
            best_result is None
            or result["aggregate_support_mass"] > best_result["aggregate_support_mass"]
        ):
            best_result = result

        print(
            f"temperature={temperature:.3f} | "
            f"full={result['full_support_mass']:.3f} | "
            f"partial={result['partial_support_mass']:.3f} | "
            f"aggregate={result['aggregate_support_mass']:.3f} | "
            f"{format_weights(encoder_to_weight)}"
        )

    print("\nbest")
    print(
        f"temperature={best_result['temperature']:.3f} | "
        f"full={best_result['full_support_mass']:.3f} | "
        f"partial={best_result['partial_support_mass']:.3f} | "
        f"aggregate={best_result['aggregate_support_mass']:.3f} | "
        f"{format_weights(best_result['encoder_to_weight'])}"
    )

if __name__ == "__main__":
    run_interpretable_search()
