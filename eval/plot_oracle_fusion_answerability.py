import sys
from os.path import dirname, abspath

# python eval/plot_oracle_fusion_answerability.py <data_variant> [timestamps...]

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
                "plot_oracle_fusion_answerability: reranker file does not exist:\n"
                f"\t{reranker_path}"
            )
    raise ValueError(
        "plot_oracle_fusion_answerability: oracle file does not reference "
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
        "plot_oracle_fusion_answerability: oracle file does not exist:\n"
        f"\t{oracle_path}"
    )

def resolve_latest_oracle_paths(data_variant, n_paths=2):
    candidate_paths = sorted(
        get_oracle_results_root().glob(f"*/{data_variant}/oracle_discriminator.json")
    )
    selected_paths = []
    for oracle_path in reversed(candidate_paths):
        oracle_data = load_json(oracle_path)
        if is_reranker_retrieval_oracle(oracle_data):
            selected_paths.append(oracle_path)
        if len(selected_paths) == n_paths:
            return list(reversed(selected_paths))
    raise ValueError(
        "plot_oracle_fusion_answerability: fewer than "
        f"{n_paths} retrieval-mode reranker oracle files found for data variant:\n"
        f"\t{data_variant}"
    )

def resolve_oracle_paths(data_variant, timestamps):
    if timestamps:
        return [
            resolve_oracle_path_from_timestamp(data_variant, timestamp)
            for timestamp in timestamps
        ]
    return resolve_latest_oracle_paths(data_variant)

def get_fusion_display_name(reranker_data):
    use_max_similarity_query_fusion_before_rrf = (
        reranker_data.get("use_max_similarity_query_fusion_before_rrf")
    )
    if use_max_similarity_query_fusion_before_rrf is True:
        return "Max similarity"
    if use_max_similarity_query_fusion_before_rrf is False:
        return "Query-level WRRF"

    for result in reranker_data.get("results") or []:
        for retrieval_result in result.get("retrieval_results") or []:
            ranked_list_matches = (
                retrieval_result.get("ranked_list_to_query_matching_retrieved_chunk")
                or {}
            )
            if any("::" in ranked_list_name for ranked_list_name in ranked_list_matches):
                return "Query-level WRRF"
            if ranked_list_matches:
                return "Max similarity"
    return "Unknown fusion"

def get_result_label(result):
    if result.get("generation_failed"):
        return None
    discriminator_result = result.get("discriminator_result") or {}
    label = str(discriminator_result.get("answerability"))
    if label not in LABEL_ORDER:
        return None
    return label

def collect_fusion_statistics(oracle_paths):
    fusion_statistics = []
    for oracle_path in oracle_paths:
        oracle_data = load_json(oracle_path)
        if not is_reranker_retrieval_oracle(oracle_data):
            raise ValueError(
                "plot_oracle_fusion_answerability: expected a retrieval-mode "
                f"reranker oracle file:\n\t{oracle_path}"
            )
        reranker_path = get_reranker_retrieval_path(oracle_data)
        reranker_data = load_json(reranker_path)
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
        fusion_statistics.append({
            "oracle_path": oracle_path,
            "reranker_path": reranker_path,
            "fusion_label": get_fusion_display_name(reranker_data),
            "timestamp": oracle_path.parent.parent.name,
            "counts": counts,
            "percentages": percentages,
            "total_count": total_count,
        })
    return fusion_statistics

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

def get_fusion_xtick_label(statistics):
    return f"{statistics['fusion_label']}\n{statistics['timestamp']}"

def plot_fusion_answerability(data_variant, fusion_statistics, output_path=None):
    if output_path is None:
        output_path = (
            get_oracle_results_root()
            / f"oracle_fusion_answerability_{data_variant}.png"
        )

    x_values = list(range(len(fusion_statistics)))
    bar_width = 0.24
    fig, ax = plt.subplots(figsize=(11.8, 5.8))
    fig.suptitle(
        "Oracle fusion answerability results\n"
        f"({data_variant})",
        fontsize=16,
    )
    for label_index, label in enumerate(LABEL_ORDER):
        offset = (label_index - 1) * bar_width
        bars = ax.bar(
            [x_value + offset for x_value in x_values],
            [
                statistics["counts"][label]
                for statistics in fusion_statistics
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
            get_fusion_xtick_label(statistics)
            for statistics in fusion_statistics
        ],
        fontsize=10.5,
    )
    max_count = max(
        statistics["counts"][label]
        for statistics in fusion_statistics
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
        "timestamps",
        nargs="*",
        help="Optional oracle run timestamps. Defaults to the latest two.",
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

    oracle_paths = resolve_oracle_paths(args.data_variant, args.timestamps)
    fusion_statistics = collect_fusion_statistics(oracle_paths)
    output_path = plot_fusion_answerability(
        data_variant=args.data_variant,
        fusion_statistics=fusion_statistics,
        output_path=output_path,
    )

    summary_lines = []
    for statistics in fusion_statistics:
        summary_lines.append(
            f"\t{statistics['fusion_label']} ({statistics['timestamp']}): "
            f"path={format_path(statistics['oracle_path'])}, "
            f"counts={statistics['counts']}, "
            f"percentages={statistics['percentages']}"
        )
    print(
        "plot_oracle_fusion_answerability: plot saved:\n"
        f"\toutput path: {format_path(output_path)}\n"
        + "\n".join(summary_lines)
    )

if __name__ == "__main__":
    main()
