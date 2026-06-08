import sys
from os.path import dirname, abspath

# python eval/plot_oracle_encoder_coverage_contribution.py <data_variant> [timestamp]

project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from config.eval import RESULTS_DIR_NAME, DATA_VARIANT_TEST_SPLIT_NAME
from helpers.data import lighten_hex_color

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
    "bge_m3_sparse": "BGE-M3\nsparse",
    "bge_m3_dense": "BGE-M3\ndense",
    "qwen3_embedding_0_6b": "Qwen3 Embedding\n0.6B",
    "jina_v5_text_small": "Jina v5\ntext small",
}
ENCODER_TO_COLOR = {
    "bm25": lighten_hex_color("#D8C454"),
    "splade": lighten_hex_color("#A27D39"),
    "bge_m3_sparse": lighten_hex_color("#FFA600"),
    "bge_m3_dense": lighten_hex_color("#6BAED6"),
    "qwen3_embedding_0_6b": lighten_hex_color("#5FB7D7"),
    "jina_v5_text_small": lighten_hex_color("#0084D1"),
}
LABEL_ORDER = ["1", "0"]
LABEL_TO_DISPLAY = {
    "1": "Fully answerable",
    "0": "Partially answerable",
}

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

def is_encoder_retrieval_oracle(oracle_data):
    if oracle_data.get("oracle_input_mode") != "retrieval":
        return False
    metadata = oracle_data.get("oracle_input_metadata") or {}
    retrieval_output_names = metadata.get("retrieval_output_names") or []
    return set(retrieval_output_names) == set(ENCODER_ORDER)

def resolve_latest_oracle_path(data_variant):
    candidate_paths = sorted(
        get_oracle_results_root().glob(f"*/{data_variant}/oracle_discriminator.json")
    )
    for oracle_path in reversed(candidate_paths):
        oracle_data = load_json(oracle_path)
        if is_encoder_retrieval_oracle(oracle_data):
            return oracle_path
    raise ValueError(
        "plot_oracle_encoder_coverage_contribution: no retrieval-mode encoder "
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
        "plot_oracle_encoder_coverage_contribution: oracle file does not exist:\n"
        f"\t{oracle_path}"
    )

def get_result_label(oracle_result):
    if oracle_result.get("generation_failed"):
        return None
    discriminator_result = oracle_result.get("discriminator_result") or {}
    label = str(discriminator_result.get("answerability"))
    if label in LABEL_ORDER:
        return label
    return None

def get_origin_encoder(origin):
    origin_id = str(origin.get("id", ""))
    source_name = origin.get("source_name")
    if not source_name:
        return None

    for encoder in ENCODER_ORDER:
        if origin_id.startswith(f"{source_name}_{encoder}_"):
            return encoder
    return None

def get_supporting_chunks(oracle_result):
    discriminator_result = oracle_result.get("discriminator_result") or {}
    supporting_chunks = []
    seen_chunk_ids = set()
    for subquery in discriminator_result.get("subqueries") or []:
        for supporting_chunk in subquery.get("supporting_chunks") or []:
            chunk_id = supporting_chunk.get("id")
            if chunk_id in seen_chunk_ids:
                continue
            seen_chunk_ids.add(chunk_id)
            supporting_chunks.append(supporting_chunk)
    return supporting_chunks

def initialize_encoder_counts():
    return {
        label: {
            encoder: 0
            for encoder in ENCODER_ORDER
        }
        for label in LABEL_ORDER
    }

def collect_encoder_statistics(oracle_data):
    coverage_counts = initialize_encoder_counts()
    contribution_counts = initialize_encoder_counts()
    label_counts = {
        label: 0
        for label in LABEL_ORDER
    }

    for oracle_result in oracle_data.get("results") or []:
        label = get_result_label(oracle_result)
        if label is None:
            continue
        label_counts[label] += 1

        supporting_chunks = get_supporting_chunks(oracle_result)
        if not supporting_chunks:
            continue
        supporting_chunk_weight = 1 / len(supporting_chunks)

        for supporting_chunk in supporting_chunks:
            covered_encoders = {
                encoder
                for encoder in [
                    get_origin_encoder(origin)
                    for origin in supporting_chunk.get("retrieval_origins") or []
                ]
                if encoder in ENCODER_ORDER
            }
            if not covered_encoders:
                continue
            for encoder in covered_encoders:
                coverage_counts[label][encoder] += supporting_chunk_weight
            encoder_contribution = supporting_chunk_weight / len(covered_encoders)
            for encoder in covered_encoders:
                contribution_counts[label][encoder] += encoder_contribution

    return {
        "coverage_counts": coverage_counts,
        "contribution_counts": contribution_counts,
        "label_counts": label_counts,
        "n_oracle_results": len(oracle_data.get("results") or []),
    }

def format_coverage_value(value):
    if float(value).is_integer():
        return str(int(value))
    return f"{value:.1f}"

def format_share_value(value):
    return f"{value:.3f}"

def add_bar_labels(ax, bars, value_formatter, fontsize=9):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            value_formatter(height),
            ha="center",
            va="bottom",
            fontsize=fontsize,
        )

def get_retrieval_timestamp(oracle_data, oracle_path):
    metadata = oracle_data.get("oracle_input_metadata") or {}
    retrieval_output_paths = metadata.get("retrieval_output_paths") or []
    timestamps = {
        Path(str(retrieval_output_path)).parent.parent.parent.name
        for retrieval_output_path in retrieval_output_paths
    }
    if len(timestamps) == 1:
        return next(iter(timestamps))
    return oracle_path.parent.parent.name

def plot_encoder_coverage(data_variant, oracle_data, oracle_path, statistics, output_path=None):
    if output_path is None:
        output_path = oracle_path.with_name("oracle_encoder_coverage_contribution.png")

    coverage_counts = statistics["coverage_counts"]
    contribution_counts = statistics["contribution_counts"]
    label_counts = statistics["label_counts"]
    x_values = list(range(len(ENCODER_ORDER)))
    bar_colors = [
        ENCODER_TO_COLOR[encoder]
        for encoder in ENCODER_ORDER
    ]

    contribution_shares = {}
    for label in LABEL_ORDER:
        contribution_total = sum(
            contribution_counts[label][encoder]
            for encoder in ENCODER_ORDER
        )
        contribution_shares[label] = {
            encoder: (
                contribution_counts[label][encoder] / contribution_total
                if contribution_total
                else 0
            )
            for encoder in ENCODER_ORDER
        }

    fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(12.8, 13.8), sharey=False)
    fig.suptitle(
        "Oracle supporting-chunk encoder coverage and contribution\n"
        f"({data_variant}, retrieval {get_retrieval_timestamp(oracle_data, oracle_path)})",
        fontsize=16,
    )

    max_count = max(
        coverage_counts[label][encoder]
        for label in LABEL_ORDER
        for encoder in ENCODER_ORDER
    )
    axis_index = 0
    for label in LABEL_ORDER:
        ax = axes[axis_index]
        bars = ax.bar(
            x_values,
            [
                coverage_counts[label][encoder]
                for encoder in ENCODER_ORDER
            ],
            color=bar_colors,
            edgecolor="white",
            linewidth=0.8,
        )
        add_bar_labels(ax, bars, format_coverage_value, fontsize=8.4)
        ax.set_title(f"Coverage - {LABEL_TO_DISPLAY[label]} (n={label_counts[label]})")
        ax.set_xticks(x_values)
        ax.set_xticklabels(
            [ENCODER_TO_DISPLAY[encoder] for encoder in ENCODER_ORDER],
            fontsize=10,
        )
        ax.set_ylim(0, max_count * 1.16 if max_count else 1)
        ax.set_ylabel("Covered samples", fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        axis_index += 1

    max_share = max(
        contribution_shares[label][encoder]
        for label in LABEL_ORDER
        for encoder in ENCODER_ORDER
    )
    for label in LABEL_ORDER:
        ax = axes[axis_index]
        bars = ax.bar(
            x_values,
            [
                contribution_shares[label][encoder]
                for encoder in ENCODER_ORDER
            ],
            color=bar_colors,
            edgecolor="white",
            linewidth=0.8,
        )
        add_bar_labels(ax, bars, format_share_value, fontsize=8.4)
        ax.set_title(f"Split contribution - {LABEL_TO_DISPLAY[label]} (n={label_counts[label]})")
        ax.set_xticks(x_values)
        ax.set_xticklabels(
            [ENCODER_TO_DISPLAY[encoder] for encoder in ENCODER_ORDER],
            fontsize=10,
        )
        ax.set_ylim(0, max_share * 1.16 if max_share else 1)
        ax.set_ylabel("Contribution share", fontsize=12)
        ax.grid(axis="y", alpha=0.25)
        axis_index += 1

    fig.text(
        0.5,
        0.02,
        "Coverage is non-exclusive; split contribution shares normalize each answerability class to 1.",
        ha="center",
        fontsize=10.5,
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
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
        help="Optional oracle run timestamp, e.g. 2026-06-04_12-00-00.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output PNG path. Defaults next to the oracle discriminator file.",
    )
    args = parser.parse_args()

    oracle_path = resolve_oracle_path(args.data_variant, args.timestamp)
    oracle_data = load_json(oracle_path)
    if not is_encoder_retrieval_oracle(oracle_data):
        raise ValueError(
            "plot_oracle_encoder_coverage_contribution: expected a retrieval-mode "
            f"encoder oracle file:\n\t{oracle_path}"
        )
    if args.output_path:
        output_path = Path(args.output_path)
        if not output_path.is_absolute():
            output_path = Path(project_root) / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    statistics = collect_encoder_statistics(oracle_data)
    output_path = plot_encoder_coverage(
        data_variant=args.data_variant,
        oracle_data=oracle_data,
        oracle_path=oracle_path,
        statistics=statistics,
        output_path=output_path,
    )
    print(
        "plot_oracle_encoder_coverage_contribution: plot saved:\n"
        f"\toracle path: {format_path(oracle_path)}\n"
        f"\toutput path: {format_path(output_path)}\n"
        f"\tlabel counts: {statistics['label_counts']}"
    )

if __name__ == "__main__":
    main()
