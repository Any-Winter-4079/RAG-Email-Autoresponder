import sys
from os.path import dirname, abspath

# python human_review/plot_no_request_human_review.py
# python human_review/plot_no_request_human_review.py human_review/results/dev/<timestamp>/no_request_human_review.json

project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from config.eval import DATA_VARIANT_TEST_SPLIT_NAME
from config.human_review import (
    HUMAN_REVIEW_NO_REQUEST_CATEGORY_PLACEHOLDER,
    HUMAN_REVIEW_RESULTS_DIR,
)
from helpers.data import lighten_hex_color

LABEL_ORDER = ["yes", "no"]
LABEL_TO_DISPLAY = {
    "yes": "Human agreement",
    "no": "Human disagreement",
}
LABEL_TO_COLOR = {
    "yes": lighten_hex_color("#9DCA1C"),
    "no": lighten_hex_color("#F53255"),
}
CATEGORY_COLORS = [
    lighten_hex_color("#29BDFD"),
    lighten_hex_color("#FFA600"),
    lighten_hex_color("#B66DFF"),
    lighten_hex_color("#00A676"),
    lighten_hex_color("#FF6B6B"),
    lighten_hex_color("#4E79A7"),
    lighten_hex_color("#F28E2B"),
    lighten_hex_color("#59A14F"),
]

def normalize_binary_label(raw_value):
    if raw_value is None:
        return None
    if raw_value is True:
        return "yes"
    if raw_value is False:
        return "no"
    raise ValueError(
        "plot_no_request_human_review: invalid human review value, expected true or false:\n"
        f"\t{raw_value}"
    )

def resolve_review_path(candidate_path=None):
    if candidate_path:
        review_path = Path(candidate_path)
        if not review_path.is_absolute():
            review_path = Path(project_root) / review_path
        if not review_path.exists():
            raise ValueError(
                "plot_no_request_human_review: review file does not exist:\n"
                f"\t{review_path}"
            )
        return review_path

    review_paths = sorted(
        (
            Path(project_root)
            / HUMAN_REVIEW_RESULTS_DIR
            / DATA_VARIANT_TEST_SPLIT_NAME
        ).rglob("no_request_human_review.json")
    )
    if not review_paths:
        raise ValueError(
            "plot_no_request_human_review: no review files found under:\n"
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
            review_entry.get("human_review_no_request_is_correct")
        )
        if label is not None:
            label_counts[label] += 1
    return label_counts

def count_review_categories(review_data):
    category_counts = {}
    for review_entry in review_data:
        category = (review_entry.get("human_review_no_request_category") or "").strip()
        if not category or category == HUMAN_REVIEW_NO_REQUEST_CATEGORY_PLACEHOLDER:
            continue
        category_counts[category] = category_counts.get(category, 0) + 1
    return dict(
        sorted(
            category_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    )

def save_review_plot(review_path, label_counts, category_counts):
    total_count = sum(label_counts.values())
    fig, (agreement_ax, category_ax) = plt.subplots(
        2,
        1,
        figsize=(10.8, 6.7),
        gridspec_kw={"height_ratios": [1.0, 1.35]},
    )

    left = 0
    for label in LABEL_ORDER:
        count = label_counts[label]
        if count == 0:
            continue
        percentage = 100.0 * count / total_count if total_count else 0.0
        agreement_ax.barh(
            [0],
            [percentage],
            left=left,
            color=LABEL_TO_COLOR[label],
            edgecolor="white",
            linewidth=0.8,
        )
        agreement_ax.text(
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

    agreement_ax.set_xlim(0, 100)
    agreement_ax.set_yticks([0])
    agreement_ax.set_yticklabels([""], fontsize=11)
    agreement_ax.set_xticks(range(0, 101, 10))
    agreement_ax.set_xlabel("Percentage of reviewed no-request samples", fontsize=12, labelpad=12)
    agreement_ax.set_title(
        "No-request agreement",
        fontsize=13,
        pad=16,
    )
    agreement_ax.grid(axis="x", alpha=0.25)
    agreement_ax.tick_params(axis="x", labelsize=11, pad=6)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=LABEL_TO_COLOR[label])
        for label in LABEL_ORDER
    ]
    legend_labels = [
        LABEL_TO_DISPLAY[label]
        for label in LABEL_ORDER
    ]
    agreement_ax.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.98),
        ncol=2,
        fontsize=11,
        frameon=False,
    )

    if category_counts:
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        category_total_count = sum(counts)
        y_positions = list(range(len(categories)))
        colors = [
            CATEGORY_COLORS[index % len(CATEGORY_COLORS)]
            for index in y_positions
        ]
        percentages = [
            100.0 * count / category_total_count
            for count in counts
        ]
        category_ax.barh(
            y_positions,
            percentages,
            color=colors,
            edgecolor="white",
            linewidth=0.8,
        )
        for y_position, count, percentage in zip(y_positions, counts, percentages):
            category_ax.text(
                percentage + 0.8,
                y_position,
                f"{count} ({percentage:.1f}%)",
                va="center",
                fontsize=11,
            )
        category_ax.set_yticks(y_positions)
        category_ax.set_yticklabels(categories, fontsize=10)
        category_ax.invert_yaxis()
        category_ax.set_xlim(0, 100)
        category_ax.set_xticks(range(0, 101, 10))
        category_ax.set_xlabel("Percentage of categorized no-request samples", fontsize=12, labelpad=12)
        category_ax.grid(axis="x", alpha=0.25)
        category_ax.tick_params(axis="x", labelsize=11, pad=6)
    else:
        category_ax.text(
            0.5,
            0.5,
            "No categories reviewed",
            ha="center",
            va="center",
            fontsize=12,
        )
        category_ax.set_xticks([])
        category_ax.set_yticks([])
    category_ax.set_title("No-request category", fontsize=13, pad=16)

    title = fig.suptitle(
        f"Human-query rewriter no-request alignment results\n"
        f"(human review: {review_path.parent.name})",
        fontsize=15,
        y=0.98,
    )
    output_path = review_path.with_name("no_request_human_review_alignment.png")
    fig.subplots_adjust(bottom=0.10, top=0.86, hspace=1.20)
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    title.set_x(
        (agreement_ax.get_position().x0 + agreement_ax.get_position().x1) / 2
    )
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "review_path",
        nargs="?",
        default=None,
        help="Path to no_request_human_review.json (defaults to latest review file).",
    )
    args = parser.parse_args()

    review_path = resolve_review_path(args.review_path)
    review_data = load_review_data(review_path)
    label_counts = count_review_labels(review_data)
    category_counts = count_review_categories(review_data)
    output_path = save_review_plot(review_path, label_counts, category_counts)

    print(
        "plot_no_request_human_review: plot saved:\n"
        f"\treview path: {format_path(review_path)}\n"
        f"\toutput path: {format_path(output_path)}\n"
        f"\treviewed items: {sum(label_counts.values())}\n"
        f"\thuman agrees: {label_counts['yes']}\n"
        f"\thuman disagrees: {label_counts['no']}\n"
        f"\tcategories reviewed: {sum(category_counts.values())}"
    )

if __name__ == "__main__":
    main()
