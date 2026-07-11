import sys
from os.path import dirname, abspath

# python eval/plot/plot_oracle_test_answerability.py lm_cleaned_text_chunks test 2026-06-30_15-01-07

project_root = abspath(dirname(dirname(dirname(__file__))))
sys.path.insert(0, project_root)

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from helpers.data import lighten_hex_color
from helpers.oracle_support import (
    format_path,
    get_oracle_results_root,
    load_json,
    resolve_oracle_path_from_timestamp,
)

LABEL_ORDER = ["1", "0", "-1"]
LABEL_TO_DISPLAY = {
    "1": "Fully",
    "0": "Partially",
    "-1": "Not answerable",
}
LABEL_TO_COLOR = {
    "1": lighten_hex_color("#9DCA1C"),
    "0": lighten_hex_color("#FFA600"),
    "-1": lighten_hex_color("#F53255"),
}
CONFIG_DISPLAY = "Post-SFT\nmax sim. + WRRF\nbest-temp. weights"

def get_label_counts(oracle_data):
    counts = oracle_data.get("discriminator_label_counts") or {}
    return {
        label: int(counts.get(label, 0))
        for label in LABEL_ORDER
    }

def get_label_percentages(oracle_data):
    percentages = oracle_data.get("discriminator_label_percentages") or {}
    return {
        label: float(percentages.get(label, 0.0))
        for label in LABEL_ORDER
    }

def collect_test_statistics(data_variant, split_name, timestamp):
    oracle_path = resolve_oracle_path_from_timestamp(
        project_root,
        split_name,
        data_variant,
        timestamp,
        "plot_oracle_test_answerability",
    )
    oracle_data = load_json(oracle_path)
    return {
        "oracle_path": oracle_path,
        "timestamp": timestamp,
        "counts": get_label_counts(oracle_data),
        "percentages": get_label_percentages(oracle_data),
    }

def add_bar_labels(ax, bars):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

def get_plot_metadata(data_variant, split_name, statistics):
    return (
        f"(variant: {data_variant}\n"
        f"split: {split_name}, oracle: {statistics['timestamp']})"
    )

def plot_test_answerability(data_variant, split_name, statistics, output_path=None):
    if output_path is None:
        output_path = (
            get_oracle_results_root(project_root, split_name)
            / f"oracle_test_answerability_{data_variant}.png"
        )

    x_values = [0]
    bar_width = 0.24
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    fig.suptitle(
        "Oracle test answerability results\n"
        f"{get_plot_metadata(data_variant, split_name, statistics)}",
        fontsize=16,
    )
    for label_index, label in enumerate(LABEL_ORDER):
        offset = (label_index - 1) * bar_width
        bars = ax.bar(
            [x_value + offset for x_value in x_values],
            [statistics["counts"][label]],
            width=bar_width,
            color=LABEL_TO_COLOR[label],
            edgecolor="white",
            linewidth=0.8,
            label=LABEL_TO_DISPLAY[label],
        )
        add_bar_labels(ax, bars)

    ax.set_title("Query-level answerability distribution")
    ax.set_ylabel("Number of samples", fontsize=12)
    ax.set_xticks(x_values)
    ax.set_xticklabels([CONFIG_DISPLAY], fontsize=10.5)
    max_count = max(statistics["counts"].values())
    ax.set_ylim(0, max_count * 1.14 if max_count else 1)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=11, ncol=3, loc="upper center")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
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
        "split_name",
        help="Oracle split name, e.g. test.",
    )
    parser.add_argument(
        "timestamp",
        help="Oracle run timestamp, e.g. 2026-06-30_15-01-07.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output PNG path. Defaults to the oracle discriminator split results root.",
    )
    args = parser.parse_args()

    if args.output_path:
        output_path = Path(args.output_path)
        if not output_path.is_absolute():
            output_path = Path(project_root) / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    statistics = collect_test_statistics(
        args.data_variant,
        args.split_name,
        args.timestamp,
    )
    output_path = plot_test_answerability(
        data_variant=args.data_variant,
        split_name=args.split_name,
        statistics=statistics,
        output_path=output_path,
    )
    print(
        "plot_oracle_test_answerability: plot saved:\n"
        f"\toutput path: {format_path(output_path, project_root)}\n"
        f"\toracle path: {format_path(statistics['oracle_path'], project_root)}\n"
        f"\tcounts: {statistics['counts']}\n"
        f"\tpercentages: {statistics['percentages']}"
    )

if __name__ == "__main__":
    main()
