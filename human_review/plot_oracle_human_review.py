import sys
from os.path import dirname, abspath

# python human_review/plot_oracle_human_review.py
# python human_review/plot_oracle_human_review.py human_review/results/dev/<timestamp>/oracle_human_review.json

project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from config.eval import DATA_VARIANT_TEST_SPLIT_NAME
from config.human_review import (
    HUMAN_REVIEW_ORACLE_ANSWERABILITY_CATEGORY_PLACEHOLDER,
    HUMAN_REVIEW_RESULTS_DIR,
)
from helpers.data import lighten_hex_color

LABEL_ORDER = ["1", "0", "-1"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABEL_ORDER)}
LABEL_TO_INT = {"1": 1, "0": 0, "-1": -1}
EXISTS_ORDER = ["yes", "no"]
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

def normalize_human_exists_flag(raw_value):
    if raw_value is None:
        return None
    if raw_value is True:
        return "yes"
    if raw_value is False:
        return "no"
    raise ValueError(
        "plot_oracle_human_review: invalid human_review_subquery_exists value, expected true or false:\n"
        f"\t{raw_value}"
    )

def resolve_review_path(candidate_path=None):
    if candidate_path:
        review_path = Path(candidate_path)
        if not review_path.is_absolute():
            review_path = Path(project_root) / review_path
        if not review_path.exists():
            raise ValueError(
                "plot_oracle_human_review: review file does not exist:\n"
                f"\t{review_path}"
            )
        return review_path

    review_paths = sorted(
        (
            Path(project_root)
            / HUMAN_REVIEW_RESULTS_DIR
            / DATA_VARIANT_TEST_SPLIT_NAME
        ).rglob("oracle_human_review.json")
    )
    if not review_paths:
        raise ValueError(
            "plot_oracle_human_review: no review files found under:\n"
            f"\t{Path(project_root) / HUMAN_REVIEW_RESULTS_DIR / DATA_VARIANT_TEST_SPLIT_NAME}"
        )
    return review_paths[-1]

def load_review_data(review_path):
    with open(review_path, "r", encoding="utf-8") as review_file:
        review_data = json.load(review_file)
    return review_data

def format_path(path):
    try:
        return path.relative_to(project_root)
    except ValueError:
        return path

def collect_alignment_statistics(review_data):
    confusion_matrix = [
        [0 for _ in LABEL_ORDER]
        for _ in LABEL_ORDER
    ]
    human_exists_counts = {
        exists_label: 0
        for exists_label in EXISTS_ORDER
    }
    email_level_mean_absolute_disagreements = []
    reviewed_existing_subquery_count = 0
    exact_agreement_count = 0
    adjacent_disagreement_count = 0
    severe_disagreement_count = 0
    category_counts = {}

    for result in review_data.get("results", []):
        discriminator_result = result.get("discriminator_result") or {}
        subqueries = discriminator_result.get("subqueries") or []
        per_email_absolute_differences = []

        for subquery in subqueries:
            human_exists_label = normalize_human_exists_flag(
                subquery.get("human_review_subquery_exists")
            )
            if human_exists_label is None:
                continue
            human_exists_counts[human_exists_label] += 1
            if human_exists_label == "no":
                continue

            oracle_label = str(subquery.get("answerability"))
            human_label = subquery.get("human_review_answerability")
            if human_label is None:
                continue
            if not isinstance(human_label, str):
                raise ValueError(
                    "plot_oracle_human_review: invalid human_review_answerability value, expected string:\n"
                    f"\t{human_label}"
                )
            human_label = human_label.strip()
            if human_label == "1|0|-1":
                continue
            if human_label not in LABEL_TO_INDEX:
                raise ValueError(
                    "plot_oracle_human_review: invalid human_review_answerability value, expected '1', '0', or '-1':\n"
                    f"\t{human_label}"
                )
            if oracle_label not in LABEL_TO_INDEX:
                continue

            category = (subquery.get("human_review_answerability_category") or "").strip()
            if (
                    human_label != "1"
                    and
                    category
                    and
                    category != HUMAN_REVIEW_ORACLE_ANSWERABILITY_CATEGORY_PLACEHOLDER):
                category_counts[category] = category_counts.get(category, 0) + 1

            confusion_matrix[LABEL_TO_INDEX[human_label]][LABEL_TO_INDEX[oracle_label]] += 1
            reviewed_existing_subquery_count += 1

            absolute_difference = abs(LABEL_TO_INT[oracle_label] - LABEL_TO_INT[human_label])
            per_email_absolute_differences.append(absolute_difference)

            if absolute_difference == 0:
                exact_agreement_count += 1
            elif absolute_difference == 1:
                adjacent_disagreement_count += 1
            elif absolute_difference == 2:
                severe_disagreement_count += 1

        if per_email_absolute_differences:
            email_level_mean_absolute_disagreements.append(
                sum(per_email_absolute_differences) / len(per_email_absolute_differences)
            )

    return {
        "confusion_matrix": confusion_matrix,
        "human_exists_counts": human_exists_counts,
        "email_level_mean_absolute_disagreements": email_level_mean_absolute_disagreements,
        "reviewed_existing_subquery_count": reviewed_existing_subquery_count,
        "exact_agreement_count": exact_agreement_count,
        "adjacent_disagreement_count": adjacent_disagreement_count,
        "severe_disagreement_count": severe_disagreement_count,
        "category_counts": dict(
            sorted(
                category_counts.items(),
                key=lambda item: (-item[1], item[0]),
            )
        ),
    }

def save_alignment_plot(review_path, alignment_statistics):
    confusion_matrix = alignment_statistics["confusion_matrix"]
    human_exists_counts = alignment_statistics["human_exists_counts"]
    email_level_mean_absolute_disagreements = alignment_statistics["email_level_mean_absolute_disagreements"]
    reviewed_existing_subquery_count = alignment_statistics["reviewed_existing_subquery_count"]
    category_counts = alignment_statistics["category_counts"]
    exists_yes_color = lighten_hex_color("#9DCA1C")
    exists_no_color = lighten_hex_color("#F53255")
    confusion_low_color = "#FFF7E6"
    confusion_mid_color = lighten_hex_color("#FFA600")
    confusion_high_color = "#FFA600"
    histogram_color = lighten_hex_color("#F3473E")
    confusion_cmap = LinearSegmentedColormap.from_list(
        "oracle_alignment_palette",
        [confusion_low_color, confusion_mid_color, confusion_high_color],
    )

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13.6, 10.0))
    fig.suptitle(
        f"Human-LM judge alignment results\n(human review: {review_path.parent.name})",
        fontsize=16,
    )

    exists_ax = axes[0][0]
    exists_x_values = EXISTS_ORDER
    exists_y_values = [human_exists_counts[label] for label in exists_x_values]
    exists_bars = exists_ax.bar(
        exists_x_values,
        exists_y_values,
        color=[exists_yes_color, exists_no_color],
        edgecolor="white",
        linewidth=0.8,
    )
    exists_ax.set_title("Human-reviewed subquery existence")
    exists_ax.set_ylabel("Number of reviewed subqueries", fontsize=12)
    exists_ax.tick_params(axis="both", labelsize=11)
    exists_ax.grid(axis="y", alpha=0.25)
    for bar in exists_bars:
        exists_ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    confusion_ax = axes[0][1]
    confusion_image = confusion_ax.imshow(confusion_matrix, cmap=confusion_cmap)
    divider = make_axes_locatable(confusion_ax)
    colorbar_ax = divider.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(confusion_image, cax=colorbar_ax)
    colorbar_ax.tick_params(labelsize=10)
    confusion_ax.set_title("Subquery-level confusion matrix")
    confusion_ax.set_xlabel("Oracle answerability", fontsize=12)
    confusion_ax.set_ylabel("Human review answerability", fontsize=12)
    confusion_ax.set_aspect("auto")
    confusion_ax.set_xticks(range(len(LABEL_ORDER)))
    confusion_ax.set_yticks(range(len(LABEL_ORDER)))
    confusion_ax.set_xticklabels(LABEL_ORDER, fontsize=11)
    confusion_ax.set_yticklabels(LABEL_ORDER, fontsize=11)
    max_confusion_value = max(
        (cell_value for row_values in confusion_matrix for cell_value in row_values),
        default=0,
    )
    for row_index, row_values in enumerate(confusion_matrix):
        for column_index, cell_value in enumerate(row_values):
            confusion_ax.text(
                column_index,
                row_index,
                str(cell_value),
                ha="center",
                va="center",
                color="white" if max_confusion_value and cell_value > 0.55 * max_confusion_value else "black",
                fontsize=11,
            )

    disagreement_ax = axes[1][0]
    if email_level_mean_absolute_disagreements:
        max_disagreement_value = max(email_level_mean_absolute_disagreements)
        bin_width = 0.25
        max_bin_edge = max(2.0, ((int(max_disagreement_value / bin_width) + 1) * bin_width))
        histogram_bins = [
            round(bin_width * bin_index, 2)
            for bin_index in range(int(max_bin_edge / bin_width) + 1)
        ]
        disagreement_ax.hist(
            email_level_mean_absolute_disagreements,
            bins=histogram_bins,
            color=histogram_color,
            edgecolor="white",
            linewidth=0.8,
        )
        disagreement_ax.set_xlim(0, histogram_bins[-1])
        disagreement_ax.set_xticks(histogram_bins)
    disagreement_ax.set_title("Distribution of email-level average L1 disagreement\n(left is better)")
    disagreement_ax.set_xlabel("Email-level average L1 disagreement", fontsize=12)
    disagreement_ax.set_ylabel("Number of reviewed emails", fontsize=12)
    disagreement_ax.tick_params(axis="both", labelsize=11)
    disagreement_ax.grid(axis="y", alpha=0.25)

    exact_agreement_rate = (
        100.0 * alignment_statistics["exact_agreement_count"] / reviewed_existing_subquery_count
        if reviewed_existing_subquery_count else 0.0
    )
    severe_disagreement_rate = (
        100.0 * alignment_statistics["severe_disagreement_count"] / reviewed_existing_subquery_count
        if reviewed_existing_subquery_count else 0.0
    )
    disagreement_ax.text(
        0.98,
        0.98,
        (
            f"Existing subqueries reviewed: {reviewed_existing_subquery_count}\n"
            f"Exact agreement: {exact_agreement_rate:.1f}%\n"
            f"Severe disagreement: {severe_disagreement_rate:.1f}%"
        ),
        transform=disagreement_ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#D8DEE9"},
    )

    category_ax = axes[1][1]
    if category_counts:
        categories = list(category_counts.keys())
        counts = list(category_counts.values())
        category_total_count = sum(counts)
        percentages = [
            100.0 * count / category_total_count
            for count in counts
        ]
        y_positions = list(range(len(categories)))
        colors = [
            CATEGORY_COLORS[index % len(CATEGORY_COLORS)]
            for index in y_positions
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
                fontsize=10.5,
            )
        category_ax.set_yticks(y_positions)
        category_ax.set_yticklabels(categories, fontsize=10)
        category_ax.invert_yaxis()
        category_ax.set_xlim(0, 100)
        category_ax.set_xticks(range(0, 101, 10))
        category_ax.set_xlabel("Percentage of categorized non-answerable subqueries", fontsize=12)
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
    category_ax.set_title("Non-answerability category")

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    output_path = review_path.with_name("oracle_human_review_alignment.png")
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "review_path",
        nargs="?",
        default=None,
        help="Path to oracle_human_review.json (defaults to latest review file).",
    )
    args = parser.parse_args()

    review_path = resolve_review_path(args.review_path)
    review_data = load_review_data(review_path)
    alignment_statistics = collect_alignment_statistics(review_data)
    output_path = save_alignment_plot(review_path, alignment_statistics)

    reviewed_existing_subquery_count = alignment_statistics["reviewed_existing_subquery_count"]
    exact_agreement_rate = (
        100.0 * alignment_statistics["exact_agreement_count"] / reviewed_existing_subquery_count
        if reviewed_existing_subquery_count else 0.0
    )
    adjacent_disagreement_rate = (
        100.0 * alignment_statistics["adjacent_disagreement_count"] / reviewed_existing_subquery_count
        if reviewed_existing_subquery_count else 0.0
    )
    severe_disagreement_rate = (
        100.0 * alignment_statistics["severe_disagreement_count"] / reviewed_existing_subquery_count
        if reviewed_existing_subquery_count else 0.0
    )
    print(
        "plot_oracle_human_review: plot saved:\n"
        f"\treview path: {format_path(review_path)}\n"
        f"\toutput path: {format_path(output_path)}\n"
        f"\treviewed existing subqueries: {reviewed_existing_subquery_count}\n"
        f"\texisting subqueries: {alignment_statistics['human_exists_counts']['yes']}\n"
        f"\tnon-existing subqueries: {alignment_statistics['human_exists_counts']['no']}\n"
        f"\texact agreement: {exact_agreement_rate:.1f}%\n"
        f"\tadjacent disagreement: {adjacent_disagreement_rate:.1f}%\n"
        f"\tsevere disagreement: {severe_disagreement_rate:.1f}%\n"
        f"\tcategories reviewed: {sum(alignment_statistics['category_counts'].values())}"
    )

if __name__ == "__main__":
    main()
