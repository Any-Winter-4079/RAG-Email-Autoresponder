import sys
from os.path import dirname, abspath

# python eval/plot/plot_oracle_supporting_chunk_count_distribution.py lm_cleaned_text_chunks dev 2026-06-16_23-50-58

project_root = abspath(dirname(dirname(dirname(__file__))))
sys.path.insert(0, project_root)

import argparse
from collections import Counter
from pathlib import Path
import matplotlib.pyplot as plt

from helpers.data import lighten_hex_color
from helpers.oracle_support import (
    format_path,
    get_answerability_label,
    get_oracle_results_root,
    get_supporting_chunk_data,
    load_json,
    resolve_oracle_path_from_timestamp,
)

ANSWERABILITY_ORDER = ["1", "0"]
ANSWERABILITY_TO_DISPLAY = {
    "1": "Fully answerable",
    "0": "Partially answerable",
}
ANSWERABILITY_TO_COLOR = {
    "1": lighten_hex_color("#9DCA1C"),
    "0": lighten_hex_color("#FFA600"),
}
ENCODER_ORDER = [
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

def collect_supporting_chunk_count_distribution(oracle_data):
    label_to_counts = {
        label: Counter()
        for label in ANSWERABILITY_ORDER
    }
    for oracle_result in oracle_data.get("results") or []:
        label = get_answerability_label(oracle_result, ANSWERABILITY_ORDER)
        if label is None:
            continue
        n_supporting_chunks = len(
            get_supporting_chunk_data(
                oracle_result=oracle_result,
                encoder_order=ENCODER_ORDER,
                data_sources=DATA_SOURCES,
                encoder_aliases=ENCODER_ALIASES,
            )
        )
        label_to_counts[label][n_supporting_chunks] += 1
    return label_to_counts

def add_bar_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        if not height:
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9.5,
        )

def get_lte_five_count(counts):
    return sum(
        sample_count
        for n_supporting_chunks, sample_count in counts.items()
        if n_supporting_chunks <= 5
    )

def plot_supporting_chunk_count_distribution(
        data_variant,
        split_name,
        oracle_timestamp,
        label_to_counts,
        output_path=None,
        ):
    if output_path is None:
        output_path = (
            get_oracle_results_root(project_root, split_name)
            / f"oracle_supporting_chunk_count_distribution_{data_variant}.png"
        )

    max_n_supporting_chunks = max(
        max(counts) if counts else 0
        for counts in label_to_counts.values()
    )
    x_values = list(range(1, max_n_supporting_chunks + 1))
    bar_width = 0.38
    fig, ax = plt.subplots(figsize=(12.6, 5.8))
    fig.suptitle(
        "Oracle-supported chunk count distribution\n"
        f"(variant: {data_variant}, split: {split_name}, oracle: {oracle_timestamp})",
        fontsize=16,
    )

    for label_index, label in enumerate(ANSWERABILITY_ORDER):
        offset = (label_index - 0.5) * bar_width
        bars = ax.bar(
            [x_value + offset for x_value in x_values],
            [
                label_to_counts[label].get(x_value, 0)
                for x_value in x_values
            ],
            width=bar_width,
            color=ANSWERABILITY_TO_COLOR[label],
            edgecolor="white",
            linewidth=0.8,
            label=ANSWERABILITY_TO_DISPLAY[label],
        )
        add_bar_labels(ax, bars)

    ax.axvline(5.5, color="#3F3F3F", linewidth=1.2, linestyle="--")
    full_lte_five_count = get_lte_five_count(label_to_counts["1"])
    full_count = sum(label_to_counts["1"].values())
    ax.text(
        0.99,
        0.895,
        f"Fully answerable samples with <=5 supported chunks: {full_lte_five_count}/{full_count}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
    )

    ax.set_title("Unique oracle-supported chunks per (fully/partially) answerable sample")
    ax.set_xlabel("Number of unique oracle-supported chunks", fontsize=12)
    ax.set_ylabel("Number of samples", fontsize=12)
    ax.set_xticks(x_values)
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=11, ncol=2, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_variant")
    parser.add_argument("split_name")
    parser.add_argument("timestamp")
    parser.add_argument("--output-path", default=None)
    args = parser.parse_args()

    output_path = Path(args.output_path) if args.output_path else None
    if output_path is not None and not output_path.is_absolute():
        output_path = Path(project_root) / output_path

    oracle_path = resolve_oracle_path_from_timestamp(
        project_root,
        args.split_name,
        args.data_variant,
        args.timestamp,
        "plot_oracle_supporting_chunk_count_distribution",
    )
    oracle_data = load_json(oracle_path)
    label_to_counts = collect_supporting_chunk_count_distribution(oracle_data)
    output_path = plot_supporting_chunk_count_distribution(
        data_variant=args.data_variant,
        split_name=args.split_name,
        oracle_timestamp=args.timestamp,
        label_to_counts=label_to_counts,
        output_path=output_path,
    )
    print(
        "plot_oracle_supporting_chunk_count_distribution: plot saved:\n"
        f"\toracle path: {format_path(oracle_path, project_root)}\n"
        f"\toutput path: {format_path(output_path, project_root)}\n"
        f"\tcounts: {dict(label_to_counts)}"
    )

if __name__ == "__main__":
    main()
