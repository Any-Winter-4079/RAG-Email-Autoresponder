import sys
from os.path import dirname, abspath

# python eval/plot_oracle_query_type_coverage_contribution.py <data_variant> [timestamp]

project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from config.eval import RESULTS_DIR_NAME, DATA_VARIANT_TEST_SPLIT_NAME
from helpers.data import lighten_hex_color

AGGREGATE_COLOR = "#B7DABD"
ENCODER_ORDER = [
    "bm25",
    "splade",
    "bge_m3_sparse",
    "bge_m3_dense",
    "qwen3_embedding_0_6b",
    "jina_v5_text_small",
]
ENCODER_TO_DISPLAY = {
    "bm25": "BM25",
    "splade": "SPLADE",
    "bge_m3_sparse": "BGE-M3 sparse",
    "bge_m3_dense": "BGE-M3 dense",
    "qwen3_embedding_0_6b": "Qwen3 Embedding 0.6B",
    "jina_v5_text_small": "Jina v5 text small",
}
ENCODER_TO_COLOR = {
    "bm25": lighten_hex_color("#D8C454"),
    "splade": lighten_hex_color("#A27D39"),
    "bge_m3_sparse": lighten_hex_color("#FFA600"),
    "bge_m3_dense": lighten_hex_color("#6BAED6"),
    "qwen3_embedding_0_6b": lighten_hex_color("#5FB7D7"),
    "jina_v5_text_small": lighten_hex_color("#0084D1"),
}
QUERY_TYPE_ORDER = ["keyword", "natural", "hyde", "question", "reranker"]
QUERY_TYPE_TO_DISPLAY = {
    "keyword": "Keyword",
    "natural": "Natural",
    "hyde": "HyDE",
    "question": "Question",
    "reranker": "Reranker",
}
LABEL_ORDER = ["1", "0"]
LABEL_TO_DISPLAY = {
    "1": "Fully answerable",
    "0": "Partially answerable",
}
TOTAL_KEY = "__total__"
TOTAL_DISPLAY = "Any encoder"
TOTAL_COLOR = AGGREGATE_COLOR

def format_path(path):
    try:
        return path.relative_to(project_root)
    except ValueError:
        return path

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)

def get_oracle_results_root():
    return (
        Path(project_root)
        / "eval"
        / RESULTS_DIR_NAME
        / "run_oracle_discriminator"
        / DATA_VARIANT_TEST_SPLIT_NAME
    )

def resolve_path(path):
    resolved_path = Path(path)
    if resolved_path.is_absolute():
        return resolved_path
    return Path(project_root) / resolved_path

def get_reranker_retrieval_path(oracle_data):
    metadata = oracle_data.get("oracle_input_metadata") or {}
    retrieval_output_paths = metadata.get("retrieval_output_paths") or []
    for retrieval_output_path in retrieval_output_paths:
        if Path(str(retrieval_output_path)).name == "reranker.json":
            reranker_path = resolve_path(retrieval_output_path)
            if reranker_path.exists():
                return reranker_path
            raise ValueError(
                "plot_oracle_query_type_coverage_contribution: reranker file does not exist:\n"
                f"\t{reranker_path}"
            )
    raise ValueError(
        "plot_oracle_query_type_coverage_contribution: oracle file does not reference "
        "a reranker.json retrieval output"
    )

def is_reranker_retrieval_oracle(oracle_data):
    if oracle_data.get("oracle_input_mode") != "retrieval":
        return False
    metadata = oracle_data.get("oracle_input_metadata") or {}
    retrieval_output_paths = metadata.get("retrieval_output_paths") or []
    return any(
        Path(str(retrieval_output_path)).name == "reranker.json"
        for retrieval_output_path in retrieval_output_paths
    )

def resolve_latest_oracle_path(data_variant):
    candidate_paths = sorted(
        get_oracle_results_root().glob(f"*/{data_variant}/oracle_discriminator.json")
    )
    for oracle_path in reversed(candidate_paths):
        oracle_data = load_json(oracle_path)
        if is_reranker_retrieval_oracle(oracle_data):
            return oracle_path
    raise ValueError(
        "plot_oracle_query_type_coverage_contribution: no retrieval-mode reranker "
        f"oracle file found for data variant:\n\t{data_variant}"
    )

def resolve_oracle_path(data_variant, timestamp):
    if timestamp is None:
        return resolve_latest_oracle_path(data_variant)

    oracle_path = (
        get_oracle_results_root()
        / timestamp
        / data_variant
        / "oracle_discriminator.json"
    )
    if oracle_path.exists():
        return oracle_path
    raise ValueError(
        "plot_oracle_query_type_coverage_contribution: oracle file does not exist:\n"
        f"\t{oracle_path}"
    )

def get_supporting_retrieval_results(oracle_result):
    discriminator_result = oracle_result.get("discriminator_result") or {}
    supporting_retrieval_results = []
    seen_chunk_ids = set()
    for subquery in discriminator_result.get("subqueries") or []:
        for supporting_chunk in subquery.get("supporting_chunks") or []:
            chunk_id = supporting_chunk.get("id")
            if chunk_id in seen_chunk_ids:
                continue
            retrieval_result = supporting_chunk.get("retrieval_result")
            if not retrieval_result:
                continue
            seen_chunk_ids.add(chunk_id)
            supporting_retrieval_results.append(retrieval_result)
    return supporting_retrieval_results

def parse_ranked_list_query_match(ranked_list_name, query_match):
    if ranked_list_name in ENCODER_ORDER:
        query_type = (query_match or {}).get("query_type")
        if query_type in QUERY_TYPE_ORDER:
            return ranked_list_name, query_type
        return None, None

    ranked_list_parts = ranked_list_name.split("::", 2)
    if len(ranked_list_parts) != 3:
        return None, None

    encoder, query_type, _ = ranked_list_parts
    if encoder not in ENCODER_ORDER or query_type not in QUERY_TYPE_ORDER:
        return None, None
    return encoder, query_type

def get_fusion_display_name(reranker_data):
    use_max_similarity_query_fusion_before_rrf = (
        reranker_data.get("use_max_similarity_query_fusion_before_rrf")
    )
    if use_max_similarity_query_fusion_before_rrf is True:
        return "max similarity fusion"
    if use_max_similarity_query_fusion_before_rrf is False:
        return "query-level RRF fusion"

    for result in reranker_data.get("results") or []:
        for retrieval_result in result.get("retrieval_results") or []:
            ranked_list_matches = (
                retrieval_result.get("ranked_list_to_query_matching_retrieved_chunk")
                or {}
            )
            if any("::" in ranked_list_name for ranked_list_name in ranked_list_matches):
                return "query-level RRF fusion"
            if ranked_list_matches:
                return "max similarity fusion"
    return "unknown fusion"

def get_result_label(oracle_result):
    if oracle_result.get("generation_failed"):
        return None
    discriminator_result = oracle_result.get("discriminator_result") or {}
    label = str(discriminator_result.get("answerability"))
    if label in LABEL_ORDER:
        return label
    return None

def initialize_coverage_counts():
    return {
        label: {
            query_type: {
                encoder: 0
                for encoder in [*ENCODER_ORDER, TOTAL_KEY]
            }
            for query_type in QUERY_TYPE_ORDER
        }
        for label in LABEL_ORDER
    }

def collect_query_type_coverage_statistics(oracle_data):
    coverage_counts = initialize_coverage_counts()
    contribution_counts = initialize_coverage_counts()
    label_counts = {
        label: 0
        for label in LABEL_ORDER
    }

    for oracle_result in oracle_data.get("results") or []:
        label = get_result_label(oracle_result)
        if label is None:
            continue
        label_counts[label] += 1

        supporting_retrieval_results = get_supporting_retrieval_results(oracle_result)
        if not supporting_retrieval_results:
            continue
        supporting_chunk_weight = 1 / len(supporting_retrieval_results)

        for retrieval_result in supporting_retrieval_results:
            covered_query_types = set()
            covered_encoder_query_types = set()
            query_matches = (
                retrieval_result.get("ranked_list_to_query_matching_retrieved_chunk")
                or {}
            )
            for ranked_list_name, query_match in query_matches.items():
                encoder, query_type = parse_ranked_list_query_match(
                    ranked_list_name=ranked_list_name,
                    query_match=query_match,
                )
                if encoder is None or query_type is None:
                    continue
                covered_query_types.add(query_type)
                covered_encoder_query_types.add((encoder, query_type))

            for query_type in covered_query_types:
                coverage_counts[label][query_type][TOTAL_KEY] += supporting_chunk_weight
            for encoder, query_type in covered_encoder_query_types:
                coverage_counts[label][query_type][encoder] += supporting_chunk_weight
            if covered_query_types:
                query_type_contribution = supporting_chunk_weight / len(covered_query_types)
                for query_type in covered_query_types:
                    contribution_counts[label][query_type][TOTAL_KEY] += query_type_contribution

    return {
        "coverage_counts": coverage_counts,
        "contribution_counts": contribution_counts,
        "label_counts": label_counts,
        "n_oracle_results": len(oracle_data.get("results") or []),
    }

def get_bar_series():
    return [
        (encoder, ENCODER_TO_DISPLAY[encoder], ENCODER_TO_COLOR[encoder])
        for encoder in ENCODER_ORDER
    ] + [
        (TOTAL_KEY, TOTAL_DISPLAY, TOTAL_COLOR)
    ]

def format_fusion_label(fusion_label):
    if fusion_label == "query-level RRF fusion":
        return "Query-level RRF"
    if fusion_label == "max similarity fusion":
        return "Max similarity"
    return fusion_label

def format_coverage_value(value):
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"

def format_share_value(value):
    return f"{value:.3f}"

def plot_query_type_coverage(data_variant, fusion_label, oracle_path, statistics, output_path=None):
    if output_path is None:
        output_path = oracle_path.with_name("oracle_query_type_coverage_contribution.png")

    coverage_counts = statistics["coverage_counts"]
    contribution_counts = statistics["contribution_counts"]
    label_counts = statistics["label_counts"]
    bar_series = get_bar_series()
    x_values = [
        index * 0.82
        for index in range(len(QUERY_TYPE_ORDER))
    ]
    bar_width = 0.105

    query_type_contribution_shares = {}
    for label in LABEL_ORDER:
        query_type_total = sum(
            contribution_counts[label][query_type][TOTAL_KEY]
            for query_type in QUERY_TYPE_ORDER
        )
        query_type_contribution_shares[label] = {
            query_type: (
                contribution_counts[label][query_type][TOTAL_KEY] / query_type_total
                if query_type_total
                else 0
            )
            for query_type in QUERY_TYPE_ORDER
        }

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(13.6, 16.0), sharey=False)
    fig.suptitle(
        "Oracle supporting-chunk query-type coverage and contribution\n"
        f"({format_fusion_label(fusion_label)}, {data_variant})",
        fontsize=16,
    )

    max_count = max(
        coverage_counts[label][query_type][series_key]
        for label in LABEL_ORDER
        for query_type in QUERY_TYPE_ORDER
        for series_key, _, _ in bar_series
    )
    axis_index = 0
    for label in LABEL_ORDER:
        ax = axes[axis_index]
        for series_index, (series_key, series_label, color) in enumerate(bar_series):
            offset = (series_index - (len(bar_series) - 1) / 2) * bar_width
            bars = ax.bar(
                [x_value + offset for x_value in x_values],
                [
                    coverage_counts[label][query_type][series_key]
                    for query_type in QUERY_TYPE_ORDER
                ],
                width=bar_width,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=series_label,
            )
            for bar in bars:
                height = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height,
                    format_coverage_value(height),
                    ha="center",
                    va="bottom",
                    fontsize=7.4,
                )

        ax.set_title(f"Coverage - {LABEL_TO_DISPLAY[label]} (n={label_counts[label]})")
        ax.set_xticks(x_values)
        ax.set_xticklabels(
            [QUERY_TYPE_TO_DISPLAY[query_type] for query_type in QUERY_TYPE_ORDER],
            fontsize=10,
        )
        ax.set_ylim(0, max_count * 1.16 if max_count else 1)
        ax.set_ylabel("Effective number of covered samples", fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        axis_index += 1

    query_type_max_share = max(
        query_type_contribution_shares[label][query_type]
        for label in LABEL_ORDER
        for query_type in QUERY_TYPE_ORDER
    )
    for label in LABEL_ORDER:
        ax = axes[axis_index]
        bars = ax.bar(
            x_values,
            [
                query_type_contribution_shares[label][query_type]
                for query_type in QUERY_TYPE_ORDER
            ],
            color=TOTAL_COLOR,
            edgecolor="white",
            linewidth=0.8,
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                format_share_value(height),
                ha="center",
                va="bottom",
                fontsize=8.8,
            )

        ax.set_title(f"Query type contribution - {LABEL_TO_DISPLAY[label]} (n={label_counts[label]})")
        ax.set_xticks(x_values)
        ax.set_xticklabels(
            [QUERY_TYPE_TO_DISPLAY[query_type] for query_type in QUERY_TYPE_ORDER],
            fontsize=10,
        )
        ax.set_ylim(0, query_type_max_share * 1.16 if query_type_max_share else 1)
        ax.set_ylabel("Contribution share", fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        axis_index += 1

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, frameon=False, fontsize=9.5, ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.035))
    fig.text(
        0.5,
        0.015,
        "Coverage is non-exclusive; split contribution shares normalize each answerability class to 1.",
        ha="center",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_variant",
        help="Selected data variant, e.g. lm_cleaned_text_chunks.",
    )
    parser.add_argument(
        "timestamp",
        nargs="?",
        help="Optional oracle run timestamp, e.g. 2026-05-18_17-54-42.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output PNG path. Defaults next to the oracle discriminator file.",
    )
    args = parser.parse_args()

    oracle_path = resolve_oracle_path(args.data_variant, args.timestamp)
    oracle_data = load_json(oracle_path)
    if not is_reranker_retrieval_oracle(oracle_data):
        raise ValueError(
            "plot_oracle_query_type_coverage_contribution: expected a retrieval-mode "
            f"reranker oracle file:\n\t{oracle_path}"
        )
    reranker_path = get_reranker_retrieval_path(oracle_data)
    reranker_data = load_json(reranker_path)
    if args.output_path:
        output_path = Path(args.output_path)
        if not output_path.is_absolute():
            output_path = Path(project_root) / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    statistics = collect_query_type_coverage_statistics(oracle_data)
    output_path = plot_query_type_coverage(
        data_variant=args.data_variant,
        fusion_label=get_fusion_display_name(reranker_data),
        oracle_path=oracle_path,
        statistics=statistics,
        output_path=output_path,
    )
    print(
        "plot_oracle_query_type_coverage_contribution: plot saved:\n"
        f"\toracle path: {format_path(oracle_path)}\n"
        f"\treranker path: {format_path(reranker_path)}\n"
        f"\toutput path: {format_path(output_path)}\n"
        f"\tlabel counts: {statistics['label_counts']}"
    )

if __name__ == "__main__":
    main()
