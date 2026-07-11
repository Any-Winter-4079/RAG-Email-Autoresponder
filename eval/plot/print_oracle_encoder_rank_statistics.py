import sys
from os.path import dirname, abspath

# python eval/plot/print_oracle_encoder_rank_statistics.py lm_cleaned_text_chunks 2026-05-17_13-48-14 2026-05-18_17-54-42

project_root = abspath(dirname(dirname(dirname(__file__))))
sys.path.insert(0, project_root)

import argparse

from config.encoder import DEFAULT_RRF_ENCODER_WEIGHT
from config.eval import DATA_VARIANT_TEST_SPLIT_NAME, TOP_K_PER_QUERY
from helpers.general import resolve_oracle_discriminator_path
from helpers.oracle_support import (
    get_answerability_label,
    get_reranker_retrieval_path,
    get_supporting_chunk_data,
    load_json,
)

RRF_RANK_CONSTANT = 60

ENCODER_ORDER = [
    "bm25",
    "splade",
    "bge_m3_muia_sparse",
    "bge_m3_muia_dense",
    "qwen3_embedding_0_6b",
    "jina_v5_text_small",
]
ENCODER_TO_DISPLAY = {
    "bm25": "BM25",
    "splade": "SPLADE",
    "bge_m3_muia_sparse": "BGE-M3 sparse",
    "bge_m3_muia_dense": "BGE-M3 dense",
    "qwen3_embedding_0_6b": "Qwen3 Embedding 0.6B",
    "jina_v5_text_small": "Jina v5 text small",
}
DATA_SOURCES = ["web", "email"]
ORIGIN_ENCODER_ALIASES = {
    "bge_m3_sparse": "bge_m3_muia_sparse",
    "bge_m3_dense": "bge_m3_muia_dense",
}
ANSWERABILITY_ORDER = ["1", "0"]
ANSWERABILITY_TO_DISPLAY = {
    "1": "Fully answerable",
    "0": "Partially answerable",
}

def get_fusion_display_name(reranker_data):
    use_max_similarity_query_fusion_before_rrf = (
        reranker_data.get("use_max_similarity_query_fusion_before_rrf")
    )
    if use_max_similarity_query_fusion_before_rrf is True:
        return "Maximum similarity + WRRF"
    if use_max_similarity_query_fusion_before_rrf is False:
        return "Query-level RRF"

    for result in reranker_data.get("results") or []:
        for retrieval_result in result.get("retrieval_results") or []:
            query_matches = (
                retrieval_result.get("ranked_list_name_to_query_matching_retrieved_chunk")
                or retrieval_result.get("ranked_list_to_query_matching_retrieved_chunk")
                or {}
            )
            if any("::" in ranked_list_name for ranked_list_name in query_matches):
                return "Query-level RRF"
            if query_matches:
                return "Maximum similarity + WRRF"
    return "Unknown fusion"

def collect_encoder_rank_statistics(oracle_data):
    # use the per-query retrieval depth as the bounded point scale
    rank_point_limit = TOP_K_PER_QUERY
    answerability_counts = {
        answerability: 0
        for answerability in ANSWERABILITY_ORDER
    }
    supported_sample_counts = {
        answerability: 0
        for answerability in ANSWERABILITY_ORDER
    }
    sample_best_ranks = {
        answerability: {
            encoder: []
            for encoder in ENCODER_ORDER
        }
        for answerability in ANSWERABILITY_ORDER
    }
    sample_average_ranks = {
        answerability: {
            encoder: []
            for encoder in ENCODER_ORDER
        }
        for answerability in ANSWERABILITY_ORDER
    }
    rank_point_scores = {
        answerability: {
            encoder: 0.0
            for encoder in ENCODER_ORDER
        }
        for answerability in ANSWERABILITY_ORDER
    }
    wrrf_contribution_scores = {
        answerability: {
            encoder: 0.0
            for encoder in ENCODER_ORDER
        }
        for answerability in ANSWERABILITY_ORDER
    }

    for oracle_result in oracle_data.get("results") or []:
        answerability = get_answerability_label(oracle_result, ANSWERABILITY_ORDER)
        if answerability is None:
            continue
        answerability_counts[answerability] += 1

        supporting_chunk_data = get_supporting_chunk_data(
            oracle_result=oracle_result,
            encoder_order=ENCODER_ORDER,
            data_sources=DATA_SOURCES,
            encoder_aliases=ORIGIN_ENCODER_ALIASES,
            include_payload=False,
        )
        if not supporting_chunk_data:
            continue
        supported_sample_counts[answerability] += 1
        # one sample has mass 1, split equally across its supported chunks
        supporting_chunk_weight = 1 / len(supporting_chunk_data)

        sample_encoder_best_chunk_ranks = {
            encoder: []
            for encoder in ENCODER_ORDER
        }
        sample_encoder_average_chunk_ranks = {
            encoder: []
            for encoder in ENCODER_ORDER
        }
        sample_rank_point_scores = {
            encoder: 0.0
            for encoder in ENCODER_ORDER
        }
        sample_wrrf_contribution_scores = {
            encoder: 0.0
            for encoder in ENCODER_ORDER
        }

        for supporting_chunk in supporting_chunk_data:
            # ranks are chunk-local: rank of this chunk in each ranked list that retrieved it
            chunk_encoder_ranks = supporting_chunk["encoder_to_ranks"]
            for encoder, ranks in chunk_encoder_ranks.items():
                for rank in ranks:
                    rank = int(rank)
                    sample_wrrf_contribution_scores[encoder] += (
                        supporting_chunk_weight
                        * DEFAULT_RRF_ENCODER_WEIGHT
                        / (RRF_RANK_CONSTANT + rank)
                    )

            for encoder, ranks in chunk_encoder_ranks.items():
                if not ranks:
                    continue
                ranks = [int(rank) for rank in ranks]
                # query-level RRF can provide multiple ranks for the same encoder/chunk
                sample_encoder_best_chunk_ranks[encoder].append(min(ranks))
                sample_encoder_average_chunk_ranks[encoder].append(
                    sum(ranks) / len(ranks)
                )
                # repeated votes are added; ranks worse than the per-query top-k get 0 points
                sample_rank_point_scores[encoder] += (
                    supporting_chunk_weight
                    * sum(max(rank_point_limit - rank + 1, 0) for rank in ranks)
                )

        for encoder in ENCODER_ORDER:
            if sample_encoder_best_chunk_ranks[encoder]:
                # rank means are computed only over samples where the encoder appears
                sample_best_ranks[answerability][encoder].append(
                    sum(sample_encoder_best_chunk_ranks[encoder])
                    / len(sample_encoder_best_chunk_ranks[encoder])
                )
                sample_average_ranks[answerability][encoder].append(
                    sum(sample_encoder_average_chunk_ranks[encoder])
                    / len(sample_encoder_average_chunk_ranks[encoder])
                )
            # point scores are averaged over all samples, so absence contributes 0
            rank_point_scores[answerability][encoder] += sample_rank_point_scores[encoder]
            wrrf_contribution_scores[answerability][encoder] += (
                sample_wrrf_contribution_scores[encoder]
            )

    table_rows = {
        answerability: []
        for answerability in ANSWERABILITY_ORDER
    }
    for answerability in ANSWERABILITY_ORDER:
        n_samples = answerability_counts[answerability]
        for encoder in ENCODER_ORDER:
            best_ranks = sample_best_ranks[answerability][encoder]
            average_ranks = sample_average_ranks[answerability][encoder]
            table_rows[answerability].append({
                "encoder": encoder,
                "n_samples_with_vote": len(best_ranks),
                "mean_best_rank": (
                    sum(best_ranks) / len(best_ranks)
                    if best_ranks
                    else None
                ),
                "mean_average_rank": (
                    sum(average_ranks) / len(average_ranks)
                    if average_ranks
                    else None
                ),
                "mean_rank_point_score": (
                    rank_point_scores[answerability][encoder] / n_samples
                    if n_samples
                    else 0
                ),
                "mean_wrrf_contribution": (
                    wrrf_contribution_scores[answerability][encoder] / n_samples
                    if n_samples
                    else 0
                ),
            })

    return {
        "rank_point_limit": rank_point_limit,
        "rrf_rank_constant": RRF_RANK_CONSTANT,
        "answerability_counts": answerability_counts,
        "supported_sample_counts": supported_sample_counts,
        "table_rows": table_rows,
        "n_oracle_results": len(oracle_data.get("results") or []),
    }

def format_optional_float(value):
    if value is None:
        return "-"
    return f"{value:.2f}"

def print_terminal_table(rows):
    table_rows = [
        [
            ENCODER_TO_DISPLAY[row["encoder"]],
            str(row["n_samples_with_vote"]),
            format_optional_float(row["mean_best_rank"]),
            format_optional_float(row["mean_average_rank"]),
            format_optional_float(row["mean_rank_point_score"]),
            f"{row['mean_wrrf_contribution']:.5f}",
        ]
        for row in rows
    ]
    headers = [
        "Encoder",
        "n samples",
        "Mean best rank",
        "Mean average rank",
        "Mean rank-point score",
        "Mean WRRF contribution",
    ]
    column_widths = [
        max(len(header), *[len(row[index]) for row in table_rows])
        for index, header in enumerate(headers)
    ]
    separator = "  ".join("-" * width for width in column_widths)

    print(
        f"{headers[0]:<{column_widths[0]}}  "
        f"{headers[1]:>{column_widths[1]}}  "
        f"{headers[2]:>{column_widths[2]}}  "
        f"{headers[3]:>{column_widths[3]}}  "
        f"{headers[4]:>{column_widths[4]}}  "
        f"{headers[5]:>{column_widths[5]}}"
    )
    print(separator)
    for row in table_rows:
        print(
            f"{row[0]:<{column_widths[0]}}  "
            f"{row[1]:>{column_widths[1]}}  "
            f"{row[2]:>{column_widths[2]}}  "
            f"{row[3]:>{column_widths[3]}}  "
            f"{row[4]:>{column_widths[4]}}  "
            f"{row[5]:>{column_widths[5]}}"
        )

def print_run_tables(data_variant, timestamp):
    oracle_path = resolve_oracle_discriminator_path(
        project_root=project_root,
        split_name=DATA_VARIANT_TEST_SPLIT_NAME,
        variant=data_variant,
        timestamp=timestamp,
    )
    oracle_data = load_json(oracle_path)
    reranker_path = get_reranker_retrieval_path(
        oracle_data,
        project_root,
        "print_oracle_encoder_rank_statistics",
    )
    reranker_data = load_json(reranker_path)
    fusion_display_name = get_fusion_display_name(reranker_data)
    statistics = collect_encoder_rank_statistics(oracle_data)

    print(f"# {fusion_display_name}")
    print()
    print(f"- data variant: `{data_variant}`")
    print(f"- retrieval: `{reranker_path.parent.parent.name}`")
    print(f"- oracle: `{oracle_path.parent.parent.name}`")
    print(f"- rank-point limit: `{statistics['rank_point_limit']}`")
    print(f"- RRF rank constant: `{statistics['rrf_rank_constant']}`")
    print(f"- oracle path: `{oracle_path}`")
    print(f"- reranker path: `{reranker_path}`")

    for answerability in ANSWERABILITY_ORDER:
        print()
        print(
            f"## {ANSWERABILITY_TO_DISPLAY[answerability]} "
            f"(n={statistics['answerability_counts'][answerability]}, "
            f"supported={statistics['supported_sample_counts'][answerability]})"
        )
        print()
        print_terminal_table(statistics["table_rows"][answerability])
    print()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "data_variant",
        help="Selected data variant, e.g. lm_cleaned_text_chunks.",
    )
    parser.add_argument(
        "timestamps",
        nargs="+",
        help="Oracle run timestamps to summarize.",
    )
    args = parser.parse_args()

    for timestamp in args.timestamps:
        print_run_tables(args.data_variant, timestamp)

if __name__ == "__main__":
    main()
