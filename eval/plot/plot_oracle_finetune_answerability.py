import sys
from os.path import dirname, abspath

# python eval/plot/plot_oracle_finetune_answerability.py lm_cleaned_text_chunks --split-name dev --pre-sft-timestamp 2026-05-17_13-48-14 --post-sft-max-sim-timestamp 2026-06-16_20-38-47 --post-sft-query-level-timestamp 2026-06-28_19-46-17 --post-sft-single-encoder-timestamp 2026-06-17_10-26-13 --post-sft-best-temperature-timestamp 2026-06-29_21-40-27

project_root = abspath(dirname(dirname(dirname(__file__))))
sys.path.insert(0, project_root)

import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from config.eval import DATA_VARIANT_TEST_SPLIT_NAME
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
CONFIG_ORDER = [
    "pre_sft",
    "post_sft_max_sim",
    "post_sft_query_level",
    "post_sft_single_encoder",
    "post_sft_best_temperature",
]
CONFIG_TO_DISPLAY = {
    "pre_sft": "Pre-SFT\nmax sim. + WRRF\nequal weights",
    "post_sft_max_sim": "Post-SFT\nmax sim. + WRRF\nequal weights",
    "post_sft_query_level": "Post-SFT\nquery-level WRRF\nequal weights",
    "post_sft_single_encoder": "Post-SFT\nmax sim. + WRRF\nsingle-encoder-perf.-based weights",
    "post_sft_best_temperature": "Post-SFT\nmax sim. + WRRF\nbest-temp. weights",
}
CONFIG_TO_TIMESTAMP_ARGUMENT = {
    "pre_sft": "pre_sft_timestamp",
    "post_sft_max_sim": "post_sft_max_sim_timestamp",
    "post_sft_query_level": "post_sft_query_level_timestamp",
    "post_sft_single_encoder": "post_sft_single_encoder_timestamp",
    "post_sft_best_temperature": "post_sft_best_temperature_timestamp",
}
def is_cross_source_reranker_oracle(oracle_data):
    metadata = oracle_data.get("oracle_input_metadata") or {}
    retrieval_output_names = metadata.get("retrieval_output_names") or []
    data_sources = metadata.get("data_sources") or []
    return (
        oracle_data.get("oracle_input_mode") == "retrieval"
        and (not retrieval_output_names or retrieval_output_names == ["reranker"])
        and data_sources == ["web", "email"]
    )

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

def collect_config_statistics(data_variant, split_name, config_to_timestamp):
    config_statistics = {}
    for config_name in CONFIG_ORDER:
        timestamp = config_to_timestamp[config_name]
        oracle_path = resolve_oracle_path_from_timestamp(
            project_root,
            split_name,
            data_variant,
            timestamp,
            "plot_oracle_finetune_answerability",
        )
        oracle_data = load_json(oracle_path)
        if not is_cross_source_reranker_oracle(oracle_data):
            raise ValueError(
                "plot_oracle_finetune_answerability: expected a cross-source reranker oracle file:\n"
                f"\t{oracle_path}"
            )
        config_statistics[config_name] = {
            "timestamp": timestamp,
            "oracle_path": oracle_path,
            "counts": get_label_counts(oracle_data),
            "percentages": get_label_percentages(oracle_data),
        }
    return config_statistics

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

def get_inline_label(label):
    return " ".join(label.splitlines())

def get_plot_metadata(data_variant, config_statistics):
    timestamps = "\n".join(
        f"{get_inline_label(CONFIG_TO_DISPLAY[config_name])} oracle: "
        f"{config_statistics[config_name]['timestamp']}"
        for config_name in CONFIG_ORDER
    )
    return f"(variant: {data_variant}\n{timestamps})"

def plot_finetune_answerability(data_variant, split_name, config_statistics, output_path=None):
    if output_path is None:
        output_path = (
            get_oracle_results_root(project_root, split_name)
            / f"oracle_finetune_answerability_{data_variant}.png"
        )

    x_values = list(range(len(CONFIG_ORDER)))
    bar_width = 0.24
    fig, ax = plt.subplots(figsize=(15.2, 6.2))
    fig.suptitle(
        "Oracle post-SFT final answerability results",
        fontsize=16,
        y=0.98,
    )
    fig.text(
        0.5,
        0.925,
        get_plot_metadata(data_variant, config_statistics),
        ha="center",
        va="top",
        fontsize=16,
    )
    for label_index, label in enumerate(LABEL_ORDER):
        offset = (label_index - 1) * bar_width
        bars = ax.bar(
            [x_value + offset for x_value in x_values],
            [
                config_statistics[config_name]["counts"][label]
                for config_name in CONFIG_ORDER
            ],
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
    ax.set_xticklabels(
        [
            CONFIG_TO_DISPLAY[config_name]
            for config_name in CONFIG_ORDER
        ],
        fontsize=9.6,
    )
    max_count = max(
        config_statistics[config_name]["counts"][label]
        for config_name in CONFIG_ORDER
        for label in LABEL_ORDER
    )
    ax.set_ylim(0, max_count * 1.14 if max_count else 1)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        frameon=False,
        fontsize=11,
        ncol=3,
        loc="upper center",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.70))
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
        "--split-name",
        default=DATA_VARIANT_TEST_SPLIT_NAME,
        help="Oracle split name, e.g. dev, train, or test.",
    )
    parser.add_argument(
        "--pre-sft-timestamp",
        required=True,
        help="Timestamp for the pre-SFT cross-source reranker oracle run.",
    )
    parser.add_argument(
        "--post-sft-max-sim-timestamp",
        required=True,
        help="Timestamp for the post-SFT max-sim cross-source reranker oracle run.",
    )
    parser.add_argument(
        "--post-sft-query-level-timestamp",
        required=True,
        help="Timestamp for the post-SFT query-level cross-source reranker oracle run.",
    )
    parser.add_argument(
        "--post-sft-single-encoder-timestamp",
        required=True,
        help="Timestamp for the post-SFT single-encoder-weight cross-source reranker oracle run.",
    )
    parser.add_argument(
        "--post-sft-best-temperature-timestamp",
        required=True,
        help="Timestamp for the post-SFT best-temperature-weight cross-source reranker oracle run.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output PNG path. Defaults to the oracle discriminator dev results root.",
    )
    args = parser.parse_args()

    config_to_timestamp = {
        "pre_sft": args.pre_sft_timestamp,
        "post_sft_max_sim": args.post_sft_max_sim_timestamp,
        "post_sft_query_level": args.post_sft_query_level_timestamp,
        "post_sft_single_encoder": args.post_sft_single_encoder_timestamp,
        "post_sft_best_temperature": args.post_sft_best_temperature_timestamp,
    }

    if args.output_path:
        output_path = Path(args.output_path)
        if not output_path.is_absolute():
            output_path = Path(project_root) / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    config_statistics = collect_config_statistics(
        args.data_variant,
        args.split_name,
        config_to_timestamp,
    )
    output_path = plot_finetune_answerability(
        data_variant=args.data_variant,
        split_name=args.split_name,
        config_statistics=config_statistics,
        output_path=output_path,
    )

    summary_lines = []
    for config_name in CONFIG_ORDER:
        statistics = config_statistics[config_name]
        summary_lines.append(
            f"\t{get_inline_label(CONFIG_TO_DISPLAY[config_name])} ({statistics['timestamp']}): "
            f"path={format_path(statistics['oracle_path'], project_root)}, "
            f"counts={statistics['counts']}, "
            f"percentages={statistics['percentages']}"
        )
    print(
        "plot_oracle_finetune_answerability: plot saved:\n"
        f"\toutput path: {format_path(output_path, project_root)}\n"
        + "\n".join(summary_lines)
    )

if __name__ == "__main__":
    main()
