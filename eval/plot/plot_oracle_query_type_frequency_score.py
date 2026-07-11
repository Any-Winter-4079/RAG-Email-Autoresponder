import sys
from os.path import dirname, abspath

# python eval/plot/plot_oracle_query_type_frequency_score.py lm_cleaned_text_chunks 2026-05-17_13-48-14
# python eval/plot/plot_oracle_query_type_frequency_score.py lm_cleaned_text_chunks 2026-05-18_17-54-42

project_root = abspath(dirname(dirname(dirname(__file__))))
sys.path.insert(0, project_root)

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from config.eval import DATA_VARIANT_TEST_SPLIT_NAME
from helpers.data import lighten_hex_color
from helpers.oracle_support import (
    format_path,
    get_reranker_retrieval_path,
    load_json,
    resolve_oracle_path_from_timestamp,
)

QUERY_TYPE_ORDER = ["keyword", "natural", "hyde", "question", "reranker"]
QUERY_TYPE_TO_DISPLAY = {
    "keyword": "Keyword",
    "natural": "Natural",
    "hyde": "HyDE",
    "question": "Question",
    "reranker": "Reranker",
}
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
ANSWERABILITY_ORDER = ["1", "0"]
ANSWERABILITY_TO_DISPLAY = {
    "1": "Fully answerable",
    "0": "Partially answerable",
}
TOTAL_KEY = "__total__"
TOTAL_DISPLAY = "Any encoder"
TOTAL_COLOR = lighten_hex_color("#399B57")

def is_reranker_retrieval_oracle(oracle_data):
    if oracle_data.get("oracle_input_mode") != "retrieval":
        return False
    metadata = oracle_data.get("oracle_input_metadata") or {}
    retrieval_output_paths = metadata.get("retrieval_output_paths") or []
    return any(
        Path(str(retrieval_output_path)).name == "reranker.json"
        for retrieval_output_path in retrieval_output_paths
    )

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
                retrieval_result.get("ranked_list_name_to_query_matching_retrieved_chunk")
                or retrieval_result.get("ranked_list_to_query_matching_retrieved_chunk")
                or {}
            )
            if any("::" in ranked_list_name for ranked_list_name in ranked_list_matches):
                return "query-level RRF fusion"
            if ranked_list_matches:
                return "max similarity fusion"
    return "unknown fusion"

def format_fusion_label(fusion_label):
    if fusion_label == "query-level RRF fusion":
        return "Query-level RRF"
    if fusion_label == "max similarity fusion":
        return "Maximum similarity + WRRF"
    return fusion_label

def get_result_answerability(oracle_result):
    if oracle_result.get("generation_failed"):
        return None
    discriminator_result = oracle_result.get("discriminator_result") or {}
    answerability = str(discriminator_result.get("answerability"))
    if answerability in ANSWERABILITY_ORDER:
        return answerability
    return None

def collect_query_type_frequency_scores(oracle_data):
    query_type_scores = {
        answerability: {
            query_type: 0.0
            for query_type in QUERY_TYPE_ORDER
        }
        for answerability in ANSWERABILITY_ORDER
    }
    encoder_query_type_scores = {
        answerability: {
            query_type: {
                encoder: 0.0
                for encoder in ENCODER_ORDER
            }
            for query_type in QUERY_TYPE_ORDER
        }
        for answerability in ANSWERABILITY_ORDER
    }
    encoder_scores = {
        answerability: {
            encoder: 0.0
            for encoder in ENCODER_ORDER
        }
        for answerability in ANSWERABILITY_ORDER
    }
    answerability_counts = {
        answerability: 0
        for answerability in ANSWERABILITY_ORDER
    }
    supported_sample_counts = {
        answerability: 0
        for answerability in ANSWERABILITY_ORDER
    }

    for oracle_result in oracle_data.get("results") or []:
        answerability = get_result_answerability(oracle_result)
        if answerability is None:
            continue
        answerability_counts[answerability] += 1

        supporting_retrieval_results = []
        seen_chunk_ids = set()
        discriminator_result = oracle_result.get("discriminator_result") or {}
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

        if not supporting_retrieval_results:
            continue
        supported_sample_counts[answerability] += 1
        supporting_chunk_weight = 1 / len(supporting_retrieval_results)

        for retrieval_result in supporting_retrieval_results:
            encoder_query_type_entries = []
            query_matches = (
                retrieval_result.get("ranked_list_name_to_query_matching_retrieved_chunk")
                or retrieval_result.get("ranked_list_to_query_matching_retrieved_chunk")
                or {}
            )
            for ranked_list_name, query_match in query_matches.items():
                if ranked_list_name in ENCODER_ORDER:
                    encoder = ranked_list_name
                    query_type = (query_match or {}).get("query_type")
                else:
                    ranked_list_parts = ranked_list_name.split("::", 2)
                    if len(ranked_list_parts) != 3:
                        continue
                    encoder, query_type, _ = ranked_list_parts

                if (
                    encoder not in ENCODER_ORDER
                    or query_type not in QUERY_TYPE_ORDER
                ):
                    continue
                encoder_query_type_entries.append((encoder, query_type))

            if not encoder_query_type_entries:
                continue
            entry_weight = supporting_chunk_weight / len(encoder_query_type_entries)
            for encoder, query_type in encoder_query_type_entries:
                query_type_scores[answerability][query_type] += entry_weight
                encoder_query_type_scores[answerability][query_type][encoder] += entry_weight
                encoder_scores[answerability][encoder] += entry_weight

    return {
        "query_type_scores": query_type_scores,
        "encoder_query_type_scores": encoder_query_type_scores,
        "encoder_scores": encoder_scores,
        "answerability_counts": answerability_counts,
        "supported_sample_counts": supported_sample_counts,
        "n_oracle_results": len(oracle_data.get("results") or []),
    }

def get_bar_series():
    return [
        (encoder, ENCODER_TO_DISPLAY[encoder], ENCODER_TO_COLOR[encoder])
        for encoder in ENCODER_ORDER
    ] + [
        (TOTAL_KEY, TOTAL_DISPLAY, TOTAL_COLOR)
    ]

def format_score_value(value):
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"

def get_retrieval_timestamp(reranker_path):
    return reranker_path.parent.parent.name

def get_oracle_timestamp(oracle_path):
    return oracle_path.parent.parent.name

def plot_query_type_frequency_score(data_variant, fusion_label, oracle_path, reranker_path, statistics, output_path=None):
    if output_path is None:
        output_path = oracle_path.with_name("oracle_query_type_frequency_score.png")

    query_type_scores = statistics["query_type_scores"]
    encoder_query_type_scores = statistics["encoder_query_type_scores"]
    encoder_scores = statistics["encoder_scores"]
    answerability_counts = statistics["answerability_counts"]
    bar_series = get_bar_series()
    query_type_x_values = [
        index * 0.82
        for index in range(len(QUERY_TYPE_ORDER))
    ]
    query_type_bar_width = 0.105
    query_type_x_axis_min = query_type_x_values[0] - 0.42
    query_type_x_axis_max = query_type_x_values[-1] + 0.42
    encoder_x_values = list(range(len(ENCODER_ORDER)))
    encoder_bar_width = 0.58
    max_score = max(
        max(
            [
                encoder_query_type_scores[answerability][query_type][encoder]
                for encoder in ENCODER_ORDER
            ] + [
                query_type_scores[answerability][query_type]
            ]
        )
        for answerability in ANSWERABILITY_ORDER
        for query_type in QUERY_TYPE_ORDER
    )
    max_encoder_score = max(
        encoder_scores[answerability][encoder]
        for answerability in ANSWERABILITY_ORDER
        for encoder in ENCODER_ORDER
    )
    max_score = max(max_score, max_encoder_score)

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(13.6, 14.8), sharex=False)
    fig.suptitle(
        "Query-type and encoder provenance of oracle-supported chunks\n"
        f"(fusion: {format_fusion_label(fusion_label)}, variant: {data_variant}\n"
        f"retrieval: {get_retrieval_timestamp(reranker_path)}, "
        f"oracle: {get_oracle_timestamp(oracle_path)})",
        fontsize=16,
        y=0.99,
    )

    axis_index = 0
    for answerability in ANSWERABILITY_ORDER:
        ax = axes[axis_index]
        for series_index, (series_key, series_label, color) in enumerate(bar_series):
            offset = (series_index - (len(bar_series) - 1) / 2) * query_type_bar_width
            if series_key == TOTAL_KEY:
                values = [
                    query_type_scores[answerability][query_type]
                    for query_type in QUERY_TYPE_ORDER
                ]
            else:
                values = [
                    encoder_query_type_scores[answerability][query_type][series_key]
                    for query_type in QUERY_TYPE_ORDER
                ]
            bars = ax.bar(
                [x_value + offset for x_value in query_type_x_values],
                values,
                width=query_type_bar_width,
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
                    format_score_value(height),
                    ha="center",
                    va="bottom",
                    fontsize=7.4,
                )
        ax.set_title(
            f"{ANSWERABILITY_TO_DISPLAY[answerability]} - query types "
            f"(n={answerability_counts[answerability]})"
        )
        ax.set_xticks(query_type_x_values)
        ax.set_xticklabels(
            [QUERY_TYPE_TO_DISPLAY[query_type] for query_type in QUERY_TYPE_ORDER],
            fontsize=10,
        )
        ax.set_xlim(query_type_x_axis_min, query_type_x_axis_max)
        ax.set_ylim(0, max_score * 1.16 if max_score else 1)
        ax.set_ylabel("Number of samples", fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="y", labelsize=11)
        axis_index += 1

        ax = axes[axis_index]
        bars = ax.bar(
            encoder_x_values,
            [
                encoder_scores[answerability][encoder]
                for encoder in ENCODER_ORDER
            ],
            width=encoder_bar_width,
            color=[
                ENCODER_TO_COLOR[encoder]
                for encoder in ENCODER_ORDER
            ],
            edgecolor="white",
            linewidth=0.8,
        )
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                format_score_value(height),
                ha="center",
                va="bottom",
                fontsize=8.4,
            )
        ax.set_title(
            f"{ANSWERABILITY_TO_DISPLAY[answerability]} - encoders "
            f"(n={answerability_counts[answerability]})"
        )
        ax.set_xticks(encoder_x_values)
        ax.set_xticklabels(
            [ENCODER_TO_DISPLAY[encoder] for encoder in ENCODER_ORDER],
            fontsize=10,
        )
        ax.set_ylim(0, max_score * 1.16 if max_score else 1)
        ax.set_ylabel("Number of samples", fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        ax.tick_params(axis="y", labelsize=11)
        axis_index += 1

    handles, legend_labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend_labels,
        frameon=False,
        fontsize=11,
        ncol=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
    )
    fig.tight_layout(rect=(0, 0.02, 1, 0.92))
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
        help="Oracle run timestamp, e.g. 2026-05-18_17-54-42.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output PNG path. Defaults next to the oracle discriminator file.",
    )
    args = parser.parse_args()

    oracle_path = resolve_oracle_path_from_timestamp(
        project_root,
        DATA_VARIANT_TEST_SPLIT_NAME,
        args.data_variant,
        args.timestamp,
        "plot_oracle_query_type_frequency_score",
    )
    oracle_data = load_json(oracle_path)
    if not is_reranker_retrieval_oracle(oracle_data):
        raise ValueError(
            "plot_oracle_query_type_frequency_score: expected a retrieval-mode "
            f"reranker oracle file:\n\t{oracle_path}"
        )
    reranker_path = get_reranker_retrieval_path(
        oracle_data,
        project_root,
        "plot_oracle_query_type_frequency_score",
    )
    reranker_data = load_json(reranker_path)
    if args.output_path:
        output_path = Path(args.output_path)
        if not output_path.is_absolute():
            output_path = Path(project_root) / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    statistics = collect_query_type_frequency_scores(oracle_data)
    output_path = plot_query_type_frequency_score(
        data_variant=args.data_variant,
        fusion_label=get_fusion_display_name(reranker_data),
        oracle_path=oracle_path,
        reranker_path=reranker_path,
        statistics=statistics,
        output_path=output_path,
    )
    score_sums = {
        answerability: sum(statistics["query_type_scores"][answerability].values())
        for answerability in ANSWERABILITY_ORDER
    }
    encoder_score_sums = {
        answerability: sum(statistics["encoder_scores"][answerability].values())
        for answerability in ANSWERABILITY_ORDER
    }
    print(
        "plot_oracle_query_type_frequency_score: plot saved:\n"
        f"\toracle path: {format_path(oracle_path, project_root)}\n"
        f"\treranker path: {format_path(reranker_path, project_root)}\n"
        f"\toutput path: {format_path(output_path, project_root)}\n"
        f"\tanswerability counts: {statistics['answerability_counts']}\n"
        f"\tsupported sample counts: {statistics['supported_sample_counts']}\n"
        f"\tquery type score sums: {score_sums}\n"
        f"\tencoder score sums: {encoder_score_sums}"
    )

if __name__ == "__main__":
    main()
