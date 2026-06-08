import sys
from os.path import dirname, abspath

# python eval/plot_oracle_data_variant_answerability.py
# python eval/plot_oracle_data_variant_answerability.py --raw-chunks <timestamp> --manually-cleaned-chunks <timestamp> --lm-cleaned-text-chunks <timestamp> --lm-summary-chunks <timestamp> --lm-q-and-a-chunks <timestamp> --lm-q-and-a-for-q-only-chunks <timestamp>

project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from config.eval import RESULTS_DIR_NAME, DATA_VARIANT_TEST_SPLIT_NAME
from helpers.data import lighten_hex_color

VARIANT_ORDER = [
    "raw_chunks",
    "manually_cleaned_chunks",
    "lm_cleaned_text_chunks",
    "lm_summary_chunks",
    "lm_q_and_a_chunks",
    "lm_q_and_a_for_q_only_chunks",
]
VARIANT_TO_DISPLAY = {
    "raw_chunks": "Raw",
    "manually_cleaned_chunks": "Manually\ncleaned",
    "lm_cleaned_text_chunks": "LM cleaned\ntext",
    "lm_summary_chunks": "LM\nsummaries",
    "lm_q_and_a_chunks": "LM Q&A",
    "lm_q_and_a_for_q_only_chunks": "LM Q&A\n(question-only)",
}
VARIANT_TO_ARGUMENT = {
    "raw_chunks": "raw_chunks",
    "manually_cleaned_chunks": "manually_cleaned_chunks",
    "lm_cleaned_text_chunks": "lm_cleaned_text_chunks",
    "lm_summary_chunks": "lm_summary_chunks",
    "lm_q_and_a_chunks": "lm_q_and_a_chunks",
    "lm_q_and_a_for_q_only_chunks": "lm_q_and_a_for_q_only_chunks",
}
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
SCORE_COLOR = lighten_hex_color("#249646")

def format_path(path):
    try:
        return path.relative_to(project_root)
    except ValueError:
        return path

def load_oracle_data(oracle_path):
    with open(oracle_path, "r", encoding="utf-8") as oracle_file:
        return json.load(oracle_file)

def get_oracle_results_root():
    return (
        Path(project_root)
        / "eval"
        / RESULTS_DIR_NAME
        / "run_oracle_discriminator"
        / DATA_VARIANT_TEST_SPLIT_NAME
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

def validate_reranker_oracle_path(oracle_path):
    oracle_data = load_oracle_data(oracle_path)
    if not is_reranker_retrieval_oracle(oracle_data):
        raise ValueError(
            "plot_oracle_data_variant_answerability: expected a retrieval-mode "
            f"reranker oracle file:\n\t{oracle_path}"
        )
    return oracle_data

def resolve_oracle_path_from_timestamp(variant, timestamp):
    oracle_path = (
        get_oracle_results_root()
        / timestamp
        / variant
        / "oracle_discriminator.json"
    )
    if not oracle_path.exists():
        raise ValueError(
            "plot_oracle_data_variant_answerability: oracle file does not exist:\n"
            f"\t{oracle_path}"
        )
    return oracle_path

def resolve_latest_oracle_path(variant):
    candidate_paths = sorted(
        get_oracle_results_root().glob(f"*/{variant}/oracle_discriminator.json")
    )
    for oracle_path in reversed(candidate_paths):
        oracle_data = load_oracle_data(oracle_path)
        if is_reranker_retrieval_oracle(oracle_data):
            return oracle_path
    raise ValueError(
        "plot_oracle_data_variant_answerability: no retrieval-mode reranker "
        f"oracle file found for variant:\n\t{variant}"
    )

def resolve_oracle_paths(args):
    configured_timestamps = {
        variant: getattr(args, VARIANT_TO_ARGUMENT[variant])
        for variant in VARIANT_ORDER
    }
    has_configured_timestamps = any(
        timestamp is not None
        for timestamp in configured_timestamps.values()
    )
    if has_configured_timestamps and any(
            timestamp is None
            for timestamp in configured_timestamps.values()):
        raise ValueError(
            "plot_oracle_data_variant_answerability: either provide no timestamps "
            "or provide one timestamp for every data variant"
        )

    if has_configured_timestamps:
        return {
            variant: resolve_oracle_path_from_timestamp(variant, timestamp)
            for variant, timestamp in configured_timestamps.items()
        }

    return {
        variant: resolve_latest_oracle_path(variant)
        for variant in VARIANT_ORDER
    }

def get_result_label(result):
    if result.get("generation_failed"):
        return None
    discriminator_result = result.get("discriminator_result") or {}
    label = str(discriminator_result.get("answerability"))
    if label not in LABEL_ORDER:
        return None
    return label

def collect_variant_statistics(oracle_paths):
    variant_statistics = {}
    for variant, oracle_path in oracle_paths.items():
        oracle_data = validate_reranker_oracle_path(oracle_path)
        counts = {
            label: 0
            for label in LABEL_ORDER
        }
        for result in oracle_data.get("results") or []:
            label = get_result_label(result)
            if label is None:
                continue
            counts[label] += 1
        total_count = sum(counts.values())
        score = counts["1"] + 0.5 * counts["0"]
        variant_statistics[variant] = {
            "oracle_path": oracle_path,
            "counts": counts,
            "total_count": total_count,
            "score": score,
        }
    return variant_statistics

def add_bar_labels(ax, bars, label_formatter):
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            label_formatter(height),
            ha="center",
            va="bottom",
            fontsize=10,
        )

def plot_answerability_distribution(ax, variant_statistics):
    x_values = list(range(len(VARIANT_ORDER)))
    bar_width = 0.24
    for label_index, label in enumerate(LABEL_ORDER):
        offset = (label_index - 1) * bar_width
        bars = ax.bar(
            [x_value + offset for x_value in x_values],
            [
                variant_statistics[variant]["counts"][label]
                for variant in VARIANT_ORDER
            ],
            width=bar_width,
            color=LABEL_TO_COLOR[label],
            edgecolor="white",
            linewidth=0.8,
            label=LABEL_TO_DISPLAY[label],
        )
        add_bar_labels(ax, bars, lambda height: f"{int(height)}")

    ax.set_title("Query-level answerability distribution")
    ax.set_ylabel("Number of queries", fontsize=12)
    max_count = max(
        variant_statistics[variant]["counts"][label]
        for variant in VARIANT_ORDER
        for label in LABEL_ORDER
    )
    ax.set_ylim(0, max_count * 1.14 if max_count else 1)
    ax.set_xticks(x_values)
    ax.set_xticklabels(
        [VARIANT_TO_DISPLAY[variant] for variant in VARIANT_ORDER],
        fontsize=10.5,
    )
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=11, ncol=3)

def plot_answerability_score(ax, variant_statistics):
    x_values = list(range(len(VARIANT_ORDER)))
    bars = ax.bar(
        x_values,
        [
            variant_statistics[variant]["score"]
            for variant in VARIANT_ORDER
        ],
        color=SCORE_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )
    add_bar_labels(ax, bars, lambda height: f"{height:.1f}")
    ax.set_title("Data variant answerability score")
    ax.set_ylabel("Score", fontsize=12)
    ax.set_xticks(x_values)
    ax.set_xticklabels(
        [VARIANT_TO_DISPLAY[variant] for variant in VARIANT_ORDER],
        fontsize=10.5,
    )
    max_score = max(
        variant_statistics[variant]["score"]
        for variant in VARIANT_ORDER
    )
    ax.set_ylim(0, max_score * 1.14 if max_score else 1)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.98,
        0.95,
        "Score = 1 * fully + 0.5 * partially",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#D8DEE9"},
    )

def save_variant_plot(variant_statistics, output_path=None):
    if output_path is None:
        output_path = (
            get_oracle_results_root()
            / "oracle_reranker_data_variant_answerability.png"
        )

    fig, axes = plt.subplots(
        nrows=2,
        ncols=1,
        figsize=(13.6, 8.8),
        gridspec_kw={"height_ratios": [2.0, 1.35]},
    )
    fig.suptitle(
        "Oracle data variant answerability results",
        fontsize=16,
    )
    plot_answerability_distribution(
        ax=axes[0],
        variant_statistics=variant_statistics,
    )
    plot_answerability_score(
        ax=axes[1],
        variant_statistics=variant_statistics,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--raw-chunks",
        default=None,
        help="Timestamp for raw_chunks oracle_discriminator.json.",
    )
    parser.add_argument(
        "--manually-cleaned-chunks",
        default=None,
        help="Timestamp for manually_cleaned_chunks oracle_discriminator.json.",
    )
    parser.add_argument(
        "--lm-cleaned-text-chunks",
        default=None,
        help="Timestamp for lm_cleaned_text_chunks oracle_discriminator.json.",
    )
    parser.add_argument(
        "--lm-summary-chunks",
        default=None,
        help="Timestamp for lm_summary_chunks oracle_discriminator.json.",
    )
    parser.add_argument(
        "--lm-q-and-a-chunks",
        default=None,
        help="Timestamp for lm_q_and_a_chunks oracle_discriminator.json.",
    )
    parser.add_argument(
        "--lm-q-and-a-for-q-only-chunks",
        default=None,
        help="Timestamp for lm_q_and_a_for_q_only_chunks oracle_discriminator.json.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output PNG path. Defaults to the oracle discriminator dev results root.",
    )
    args = parser.parse_args()

    if args.output_path:
        output_path = Path(args.output_path)
        if not output_path.is_absolute():
            output_path = Path(project_root) / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    oracle_paths = resolve_oracle_paths(args)
    variant_statistics = collect_variant_statistics(oracle_paths)
    output_path = save_variant_plot(
        variant_statistics=variant_statistics,
        output_path=output_path,
    )

    summary_lines = []
    for variant in VARIANT_ORDER:
        statistics = variant_statistics[variant]
        counts = statistics["counts"]
        summary_lines.append(
            f"\t{variant}: path={format_path(statistics['oracle_path'])}, "
            f"counts={counts}, score={statistics['score']:.1f}"
        )
    print(
        "plot_oracle_data_variant_answerability: plot saved:\n"
        f"\toutput path: {format_path(output_path)}\n"
        + "\n".join(summary_lines)
    )

if __name__ == "__main__":
    main()
