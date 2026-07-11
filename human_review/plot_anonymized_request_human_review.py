import sys
from os.path import dirname, abspath

# python human_review/plot_anonymized_request_human_review.py
# python human_review/plot_anonymized_request_human_review.py human_review/results/dev/<timestamp>/anonymized_request_human_review.json

project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from config.eval import DATA_VARIANT_TEST_SPLIT_NAME
from config.human_review import HUMAN_REVIEW_RESULTS_DIR
from helpers.data import lighten_hex_color

LABEL_ORDER = ["yes", "no"]
LABEL_TO_DISPLAY = {
    "yes": "Human agrees",
    "no": "Human disagrees",
}
LABEL_TO_COLOR = {
    "yes": lighten_hex_color("#9DCA1C"),
    "no": lighten_hex_color("#F53255"),
}

def normalize_binary_label(raw_value):
    if raw_value is None:
        return None
    if raw_value is True:
        return "yes"
    if raw_value is False:
        return "no"
    raise ValueError(
        "plot_anonymized_request_human_review: invalid human review value, expected true or false:\n"
        f"\t{raw_value}"
    )

def resolve_review_path(candidate_path=None):
    if candidate_path:
        review_path = Path(candidate_path)
        if not review_path.is_absolute():
            review_path = Path(project_root) / review_path
        if not review_path.exists():
            raise ValueError(
                "plot_anonymized_request_human_review: review file does not exist:\n"
                f"\t{review_path}"
            )
        return review_path

    review_paths = sorted(
        (
            Path(project_root)
            / HUMAN_REVIEW_RESULTS_DIR
            / DATA_VARIANT_TEST_SPLIT_NAME
        ).rglob("anonymized_request_human_review.json")
    )
    if not review_paths:
        raise ValueError(
            "plot_anonymized_request_human_review: no review files found under:\n"
            f"\t{Path(project_root) / HUMAN_REVIEW_RESULTS_DIR / DATA_VARIANT_TEST_SPLIT_NAME}"
        )
    return review_paths[-1]

def load_review_data(review_path):
    with open(review_path, "r", encoding="utf-8") as review_file:
        return json.load(review_file)

def format_path(path):
    try:
        return path.relative_to(project_root)
    except ValueError:
        return path

def count_review_labels(review_data):
    label_counts = {
        label: 0
        for label in LABEL_ORDER
    }
    for review_entry in review_data:
        label = normalize_binary_label(
            review_entry.get("human_review_anonymized_request_is_good")
        )
        if label is not None:
            label_counts[label] += 1
    return label_counts

def save_review_plot(review_path, label_counts):
    total_count = sum(label_counts.values())
    fig, ax = plt.subplots(figsize=(10.8, 3.4))

    left = 0
    for label in LABEL_ORDER:
        count = label_counts[label]
        if count == 0:
            continue
        percentage = 100.0 * count / total_count if total_count else 0.0
        ax.barh(
            [0],
            [percentage],
            left=left,
            color=LABEL_TO_COLOR[label],
            edgecolor="white",
            linewidth=0.8,
        )
        ax.text(
            left + percentage / 2,
            0,
            f"{LABEL_TO_DISPLAY[label]}\n{count} ({percentage:.1f}%)",
            ha="center",
            va="center",
            color="white",
            fontsize=11,
            fontweight="bold",
        )
        left += percentage

    ax.set_xlim(0, 100)
    ax.set_yticks([0])
    ax.set_yticklabels([""], fontsize=11)
    ax.set_xticks(range(0, 101, 10))
    ax.set_xlabel("Percentage of reviewed anonymized-request samples", fontsize=12, labelpad=12)
    ax.set_title(
        f"Human-query rewriter anonymized-request alignment results\n"
        f"({review_path.parent.name})",
        fontsize=15,
        pad=16,
    )
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="x", labelsize=11, pad=6)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=LABEL_TO_COLOR[label])
        for label in LABEL_ORDER
    ]
    legend_labels = [
        LABEL_TO_DISPLAY[label]
        for label in LABEL_ORDER
    ]
    ax.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -1.0),
        ncol=2,
        fontsize=11,
        frameon=False,
    )

    output_path = review_path.with_name("anonymized_request_human_review_alignment.png")
    fig.subplots_adjust(left=0.15, right=0.95, bottom=0.42, top=0.72)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "review_path",
        nargs="?",
        default=None,
        help="Path to anonymized_request_human_review.json (defaults to latest review file).",
    )
    args = parser.parse_args()

    review_path = resolve_review_path(args.review_path)
    review_data = load_review_data(review_path)
    label_counts = count_review_labels(review_data)
    output_path = save_review_plot(review_path, label_counts)

    print(
        "plot_anonymized_request_human_review: plot saved:\n"
        f"\treview path: {format_path(review_path)}\n"
        f"\toutput path: {format_path(output_path)}\n"
        f"\treviewed items: {sum(label_counts.values())}\n"
        f"\thuman agrees: {label_counts['yes']}\n"
        f"\thuman disagrees: {label_counts['no']}"
    )

if __name__ == "__main__":
    main()
