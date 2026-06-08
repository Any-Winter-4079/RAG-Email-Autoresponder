import sys
from os.path import dirname, abspath

# python eval/plot_oracle_final_answerability.py <data_variant>
# python eval/plot_oracle_final_answerability.py <data_variant> --email-rrf-timestamp <timestamp> --reranker-timestamp <timestamp> --both-rrf-timestamp <timestamp>

project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from config.eval import RESULTS_DIR_NAME, DATA_VARIANT_TEST_SPLIT_NAME
from helpers.data import lighten_hex_color

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
STRATEGY_ORDER = ["email_rrf", "reranker", "both_rrf"]
STRATEGY_TO_DISPLAY = {
    "email_rrf": "Email-only RRF",
    "reranker": "Cross-source reranker",
    "both_rrf": "Cross-source RRF",
}
STRATEGY_TO_TIMESTAMP_ARGUMENT = {
    "email_rrf": "email_rrf_timestamp",
    "reranker": "reranker_timestamp",
    "both_rrf": "both_rrf_timestamp",
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

def resolve_path(path):
    resolved_path = Path(path)
    if resolved_path.is_absolute():
        return resolved_path
    return Path(project_root) / resolved_path

def resolve_oracle_path_from_timestamp(data_variant, timestamp):
    oracle_path = (
        get_oracle_results_root()
        / timestamp
        / data_variant
        / "oracle_discriminator.json"
    )
    if oracle_path.exists():
        return oracle_path
    raise ValueError(
        "plot_oracle_final_answerability: oracle file does not exist:\n"
        f"\t{oracle_path}"
    )

def get_retrieval_output_paths(oracle_data):
    metadata = oracle_data.get("oracle_input_metadata") or {}
    return [
        resolve_path(retrieval_output_path)
        for retrieval_output_path in metadata.get("retrieval_output_paths") or []
    ]

def get_data_sources(oracle_data):
    metadata = oracle_data.get("oracle_input_metadata") or {}
    return metadata.get("data_sources") or []

def is_retrieval_oracle(oracle_data):
    return oracle_data.get("oracle_input_mode") == "retrieval"

def has_retrieval_file_name(oracle_data, file_name):
    return any(
        retrieval_output_path.name == file_name
        for retrieval_output_path in get_retrieval_output_paths(oracle_data)
    )

def get_retrieval_run_dir_from_reranker_path(reranker_path):
    return reranker_path.parent

def get_retrieval_max_similarity_value(oracle_data):
    retrieval_output_paths = get_retrieval_output_paths(oracle_data)
    if has_retrieval_file_name(oracle_data, "reranker.json"):
        reranker_path = next(
            path
            for path in retrieval_output_paths
            if path.name == "reranker.json"
        )
        retrieval_run_dir = get_retrieval_run_dir_from_reranker_path(reranker_path)
        rrf_paths = [
            retrieval_run_dir / data_source / "rrf.json"
            for data_source in get_data_sources(oracle_data)
        ]
    else:
        rrf_paths = [
            path
            for path in retrieval_output_paths
            if path.name == "rrf.json"
        ]

    values = []
    for rrf_path in rrf_paths:
        if not rrf_path.exists():
            return None
        rrf_data = load_json(rrf_path)
        values.append(rrf_data.get("use_max_similarity_query_fusion_before_rrf"))

    if not values:
        return None
    if all(value is True for value in values):
        return True
    if all(value is False for value in values):
        return False
    return None

def is_strategy_oracle(oracle_data, strategy):
    if not is_retrieval_oracle(oracle_data):
        return False
    if strategy == "email_rrf":
        return (
            get_data_sources(oracle_data) == ["email"]
            and all(
                path.name == "rrf.json"
                for path in get_retrieval_output_paths(oracle_data)
            )
            and get_retrieval_max_similarity_value(oracle_data) is True
        )
    if strategy == "reranker":
        return (
            get_data_sources(oracle_data) == ["web", "email"]
            and has_retrieval_file_name(oracle_data, "reranker.json")
            and get_retrieval_max_similarity_value(oracle_data) is True
        )
    if strategy == "both_rrf":
        return (
            get_data_sources(oracle_data) == ["web", "email"]
            and all(
                path.name == "rrf.json"
                for path in get_retrieval_output_paths(oracle_data)
            )
            and get_retrieval_max_similarity_value(oracle_data) is True
        )
    raise ValueError(
        "plot_oracle_final_answerability: unknown strategy "
        f"{strategy}"
    )

def resolve_latest_strategy_oracle_path(data_variant, strategy):
    candidate_paths = sorted(
        get_oracle_results_root().glob(f"*/{data_variant}/oracle_discriminator.json")
    )
    for oracle_path in reversed(candidate_paths):
        oracle_data = load_json(oracle_path)
        if is_strategy_oracle(oracle_data, strategy):
            return oracle_path
    raise ValueError(
        "plot_oracle_final_answerability: no matching oracle file "
        f"found for strategy {strategy} and data variant:\n\t{data_variant}"
    )

def resolve_strategy_oracle_paths(data_variant, args):
    strategy_to_oracle_path = {}
    for strategy in STRATEGY_ORDER:
        timestamp = getattr(args, STRATEGY_TO_TIMESTAMP_ARGUMENT[strategy])
        if timestamp is not None:
            strategy_to_oracle_path[strategy] = resolve_oracle_path_from_timestamp(
                data_variant=data_variant,
                timestamp=timestamp,
            )
        else:
            strategy_to_oracle_path[strategy] = resolve_latest_strategy_oracle_path(
                data_variant=data_variant,
                strategy=strategy,
            )
    return strategy_to_oracle_path

def get_result_label(result):
    if result.get("generation_failed"):
        return None
    discriminator_result = result.get("discriminator_result") or {}
    label = str(discriminator_result.get("answerability"))
    if label not in LABEL_ORDER:
        return None
    return label

def collect_strategy_statistics(strategy_to_oracle_path):
    strategy_statistics = {}
    for strategy, oracle_path in strategy_to_oracle_path.items():
        oracle_data = load_json(oracle_path)
        if not is_strategy_oracle(oracle_data, strategy):
            raise ValueError(
                "plot_oracle_final_answerability: oracle file does "
                f"not match strategy {strategy}:\n\t{oracle_path}"
            )
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
        percentages = {
            label: (100 * counts[label] / total_count if total_count else 0)
            for label in LABEL_ORDER
        }
        strategy_statistics[strategy] = {
            "oracle_path": oracle_path,
            "timestamp": oracle_path.parent.parent.name,
            "counts": counts,
            "percentages": percentages,
            "total_count": total_count,
        }
    return strategy_statistics

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

def get_strategy_xtick_label(strategy, statistics):
    return f"{STRATEGY_TO_DISPLAY[strategy]}\n{statistics['timestamp']}"

def plot_final_answerability(data_variant, strategy_statistics, output_path=None):
    if output_path is None:
        output_path = (
            get_oracle_results_root()
            / f"oracle_final_answerability_{data_variant}.png"
        )

    x_values = list(range(len(STRATEGY_ORDER)))
    bar_width = 0.24
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    fig.suptitle(
        "Oracle final answerability results\n"
        f"({data_variant})",
        fontsize=16,
    )
    for label_index, label in enumerate(LABEL_ORDER):
        offset = (label_index - 1) * bar_width
        bars = ax.bar(
            [x_value + offset for x_value in x_values],
            [
                strategy_statistics[strategy]["counts"][label]
                for strategy in STRATEGY_ORDER
            ],
            width=bar_width,
            color=LABEL_TO_COLOR[label],
            edgecolor="white",
            linewidth=0.8,
            label=LABEL_TO_DISPLAY[label],
        )
        add_bar_labels(ax, bars)

    ax.set_title("Query-level answerability distribution")
    ax.set_ylabel("Number of judged samples", fontsize=12)
    ax.set_xticks(x_values)
    ax.set_xticklabels(
        [
            get_strategy_xtick_label(strategy, strategy_statistics[strategy])
            for strategy in STRATEGY_ORDER
        ],
        fontsize=10.5,
    )
    max_count = max(
        strategy_statistics[strategy]["counts"][label]
        for strategy in STRATEGY_ORDER
        for label in LABEL_ORDER
    )
    ax.set_ylim(0, max_count * 1.14 if max_count else 1)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=11, ncol=3)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
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
        "--email-rrf-timestamp",
        default=None,
        help="Timestamp for the email-only RRF oracle run.",
    )
    parser.add_argument(
        "--reranker-timestamp",
        default=None,
        help="Timestamp for the cross-source reranker oracle run.",
    )
    parser.add_argument(
        "--both-rrf-timestamp",
        default=None,
        help="Timestamp for the cross-source RRF oracle run.",
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

    strategy_to_oracle_path = resolve_strategy_oracle_paths(args.data_variant, args)
    strategy_statistics = collect_strategy_statistics(strategy_to_oracle_path)
    output_path = plot_final_answerability(
        data_variant=args.data_variant,
        strategy_statistics=strategy_statistics,
        output_path=output_path,
    )

    summary_lines = []
    for strategy in STRATEGY_ORDER:
        statistics = strategy_statistics[strategy]
        summary_lines.append(
            f"\t{STRATEGY_TO_DISPLAY[strategy]} ({statistics['timestamp']}): "
            f"path={format_path(statistics['oracle_path'])}, "
            f"counts={statistics['counts']}, "
            f"percentages={statistics['percentages']}"
        )
    print(
        "plot_oracle_final_answerability: plot saved:\n"
        f"\toutput path: {format_path(output_path)}\n"
        + "\n".join(summary_lines)
    )

if __name__ == "__main__":
    main()
