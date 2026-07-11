import sys
from os.path import dirname, abspath

# python eval/plot/plot_oracle_source_complementarity.py --web-oracle-path eval/results/run_oracle_discriminator/dev/2026-05-14_19-31-38/lm_summary_chunks/oracle_discriminator.json --email-oracle-path eval/results/run_oracle_discriminator/dev/2026-05-15_01-52-23/lm_summary_chunks/oracle_discriminator.json

project_root = abspath(dirname(dirname(dirname(__file__))))
sys.path.insert(0, project_root)

import json
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable

from helpers.data import lighten_hex_color
from helpers.oracle_support import (
    format_path,
    load_json,
    resolve_existing_path,
)

SOURCE_ORDER = ["web", "email"]
LABEL_ORDER = ["1", "0", "-1"]
LABEL_TO_INDEX = {label: index for index, label in enumerate(LABEL_ORDER)}
LABEL_TO_INT = {"1": 1, "0": 0, "-1": -1}
LABEL_TO_DISPLAY = {
    "1": "1\nFully",
    "0": "0\nPartially",
    "-1": "-1\nNot answerable",
}
SOURCE_TO_DISPLAY = {
    "web": "Web",
    "email": "Email",
    "union": "Both",
}
SOURCE_TO_COLOR = {
    "web": lighten_hex_color("#00CBBF"),
    "email": lighten_hex_color("#9DCA1C"),
    "union": lighten_hex_color("#FFA600"),
}
DISAGREEMENT_COLOR = lighten_hex_color("#F3473E")

def get_timestamp_after_path_part(path_text, path_part):
    path_parts = Path(path_text).parts
    for index, current_path_part in enumerate(path_parts[:-1]):
        if current_path_part == path_part:
            return path_parts[index + 1]
    return None

def get_metadata_paths(metadata, field_name):
    paths = []
    if metadata.get(field_name):
        paths.append(str(metadata[field_name]))
    paths.extend(
        str(path)
        for path in metadata.get(f"{field_name}s") or []
    )
    return paths

def get_input_timestamp(oracle_data):
    metadata = oracle_data.get("oracle_input_metadata") or {}
    for collection_dump_path in get_metadata_paths(metadata, "collection_dump_path"):
        timestamp = get_timestamp_after_path_part(
            collection_dump_path,
            "run_dump_collection_payloads",
        )
        if timestamp:
            return timestamp
    return "unknown"

def get_oracle_timestamp(oracle_path):
    return oracle_path.parent.parent.name

def get_data_variant(web_oracle_path, email_oracle_path):
    if web_oracle_path.parent.name == email_oracle_path.parent.name:
        return web_oracle_path.parent.name
    return f"{web_oracle_path.parent.name} / {email_oracle_path.parent.name}"

def get_result_label(result):
    if result.get("generation_failed"):
        return None
    discriminator_result = result.get("discriminator_result") or {}
    label = str(discriminator_result.get("answerability"))
    if label not in LABEL_TO_INDEX:
        return None
    return label

def get_subquery_label(subquery):
    label = str(subquery.get("answerability"))
    if label not in LABEL_TO_INDEX:
        return None
    return label

def count_query_labels(results):
    counts = {label: 0 for label in LABEL_ORDER}
    for result in results:
        label = get_result_label(result)
        if label is None:
            continue
        counts[label] += 1
    return counts

def count_subquery_labels(results):
    counts = {label: 0 for label in LABEL_ORDER}
    for result in results:
        if result.get("generation_failed"):
            continue
        discriminator_result = result.get("discriminator_result") or {}
        for subquery in discriminator_result.get("subqueries") or []:
            label = get_subquery_label(subquery)
            if label is None:
                continue
            counts[label] += 1
    return counts

def get_sample_key(result):
    sample = result.get("sample") or {}
    email = sample.get("email") or {}
    return json.dumps(
        [
            sample.get("folder_uri"),
            sample.get("thread_id"),
            email.get("subject"),
            email.get("body"),
        ],
        ensure_ascii=False,
        sort_keys=True,
    )

def pair_results(web_results, email_results):
    if (
            len(web_results) == len(email_results)
            and all(
                get_sample_key(web_result) == get_sample_key(email_result)
                for web_result, email_result in zip(web_results, email_results)
            )):
        return list(zip(web_results, email_results))

    email_result_by_sample_key = {
        get_sample_key(email_result): email_result
        for email_result in email_results
    }
    return [
        (web_result, email_result_by_sample_key[sample_key])
        for web_result in web_results
        for sample_key in [get_sample_key(web_result)]
        if sample_key in email_result_by_sample_key
    ]

def collect_comparison_statistics(web_data, email_data):
    web_results = web_data.get("results") or []
    email_results = email_data.get("results") or []
    paired_results = pair_results(web_results, email_results)

    confusion_matrix = [
        [0 for _ in LABEL_ORDER]
        for _ in LABEL_ORDER
    ]
    l1_disagreements = []
    n_web_higher = 0
    n_email_higher = 0
    n_same = 0
    union_query_counts = {label: 0 for label in LABEL_ORDER}
    for web_result, email_result in paired_results:
        web_label = get_result_label(web_result)
        email_label = get_result_label(email_result)
        if web_label is None or email_label is None:
            continue
        confusion_matrix[LABEL_TO_INDEX[web_label]][LABEL_TO_INDEX[email_label]] += 1
        difference = LABEL_TO_INT[email_label] - LABEL_TO_INT[web_label]
        l1_disagreements.append(abs(difference))
        if difference > 0:
            n_email_higher += 1
        elif difference < 0:
            n_web_higher += 1
        else:
            n_same += 1
        union_label = max([web_label, email_label], key=lambda label: LABEL_TO_INT[label])
        union_query_counts[union_label] += 1

    return {
        "web_query_counts": count_query_labels(web_results),
        "email_query_counts": count_query_labels(email_results),
        "union_query_counts": union_query_counts,
        "web_subquery_counts": count_subquery_labels(web_results),
        "email_subquery_counts": count_subquery_labels(email_results),
        "confusion_matrix": confusion_matrix,
        "l1_disagreements": l1_disagreements,
        "n_paired_results": len(paired_results),
        "n_same": n_same,
        "n_web_higher": n_web_higher,
        "n_email_higher": n_email_higher,
    }

def add_bar_labels(ax, bars):
    for bar in bars:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=10.5,
        )

def plot_answerability_counts(ax, web_counts, email_counts, title, ylabel, union_counts=None):
    x_values = list(range(len(LABEL_ORDER)))
    bar_width = 0.24 if union_counts is not None else 0.36
    web_bars = ax.bar(
        [
            x_value - bar_width
            if union_counts is not None
            else x_value - bar_width / 2
            for x_value in x_values
        ],
        [web_counts[label] for label in LABEL_ORDER],
        width=bar_width,
        color=SOURCE_TO_COLOR["web"],
        edgecolor="white",
        linewidth=0.8,
        label=SOURCE_TO_DISPLAY["web"],
    )
    email_bars = ax.bar(
        [
            x_value
            if union_counts is not None
            else x_value + bar_width / 2
            for x_value in x_values
        ],
        [email_counts[label] for label in LABEL_ORDER],
        width=bar_width,
        color=SOURCE_TO_COLOR["email"],
        edgecolor="white",
        linewidth=0.8,
        label=SOURCE_TO_DISPLAY["email"],
    )
    if union_counts is not None:
        union_bars = ax.bar(
            [x_value + bar_width for x_value in x_values],
            [union_counts[label] for label in LABEL_ORDER],
            width=bar_width,
            color=SOURCE_TO_COLOR["union"],
            edgecolor="white",
            linewidth=0.8,
            label=SOURCE_TO_DISPLAY["union"],
        )
        add_bar_labels(ax, union_bars)
    add_bar_labels(ax, web_bars)
    add_bar_labels(ax, email_bars)
    ax.set_title(title)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_xticks(x_values)
    ax.set_xticklabels([LABEL_TO_DISPLAY[label] for label in LABEL_ORDER], fontsize=11)
    ax.tick_params(axis="y", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=11)

def plot_confusion_matrix(ax, fig, confusion_matrix):
    confusion_cmap = LinearSegmentedColormap.from_list(
        "oracle_source_complementarity_palette",
        ["#FFF7E6", lighten_hex_color("#FFA600"), "#FFA600"],
    )
    confusion_image = ax.imshow(confusion_matrix, cmap=confusion_cmap)
    divider = make_axes_locatable(ax)
    colorbar_ax = divider.append_axes("right", size="4%", pad=0.08)
    fig.colorbar(confusion_image, cax=colorbar_ax)
    colorbar_ax.tick_params(labelsize=10)
    ax.set_title("Query-level answerability confusion matrix")
    ax.set_xlabel("Email answerability", fontsize=12)
    ax.set_ylabel("Web answerability", fontsize=12)
    ax.set_aspect("auto")
    ax.set_xticks(range(len(LABEL_ORDER)))
    ax.set_yticks(range(len(LABEL_ORDER)))
    ax.set_xticklabels(LABEL_ORDER, fontsize=11)
    ax.set_yticklabels(LABEL_ORDER, fontsize=11)

    max_value = max(
        (cell_value for row_values in confusion_matrix for cell_value in row_values),
        default=0,
    )
    for row_index, row_values in enumerate(confusion_matrix):
        for column_index, cell_value in enumerate(row_values):
            ax.text(
                column_index,
                row_index,
                str(cell_value),
                ha="center",
                va="center",
                color="white" if max_value and cell_value > 0.55 * max_value else "black",
                fontsize=11,
            )

def plot_l1_disagreement(ax, comparison_statistics):
    l1_disagreements = comparison_statistics["l1_disagreements"]
    l1_counts = {
        disagreement_value: l1_disagreements.count(disagreement_value)
        for disagreement_value in [0, 1, 2]
    }
    bars = ax.bar(
        list(l1_counts.keys()),
        list(l1_counts.values()),
        color=DISAGREEMENT_COLOR,
        edgecolor="white",
        linewidth=0.8,
    )
    add_bar_labels(ax, bars)
    mean_l1 = (
        sum(l1_disagreements) / len(l1_disagreements)
        if l1_disagreements
        else 0.0
    )
    ax.set_title("Query-level L1 disagreement")
    ax.set_xlabel("Absolute label difference", fontsize=12)
    ax.set_ylabel("Number of paired queries", fontsize=12)
    ax.set_xticks([0, 1, 2])
    ax.tick_params(axis="both", labelsize=11)
    ax.grid(axis="y", alpha=0.25)
    ax.text(
        0.98,
        0.98,
        (
            f"Paired queries: {len(l1_disagreements)}\n"
            f"Mean L1: {mean_l1:.2f}\n"
            f"Same: {comparison_statistics['n_same']}\n"
            f"Email higher: {comparison_statistics['n_email_higher']}\n"
            f"Web higher: {comparison_statistics['n_web_higher']}"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#D8DEE9"},
    )

def save_comparison_plot(
        web_oracle_path,
        email_oracle_path,
        output_path,
        web_data,
        email_data,
        comparison_statistics):
    if output_path is None:
        output_path = email_oracle_path.with_name("oracle_source_complementarity.png")

    fig, axes = plt.subplots(nrows=2, ncols=2, figsize=(13.6, 10.0))
    fig.suptitle(
        "Oracle data source answerability results\n"
        f"(variant: {get_data_variant(web_oracle_path, email_oracle_path)}\n"
        f"web dump: {get_input_timestamp(web_data)}, "
        f"web oracle: {get_oracle_timestamp(web_oracle_path)}\n"
        f"email dump: {get_input_timestamp(email_data)}, "
        f"email oracle: {get_oracle_timestamp(email_oracle_path)})",
        fontsize=16,
    )

    plot_answerability_counts(
        ax=axes[0][0],
        web_counts=comparison_statistics["web_query_counts"],
        email_counts=comparison_statistics["email_query_counts"],
        union_counts=comparison_statistics["union_query_counts"],
        title="Query-level answerability distribution",
        ylabel="Number of queries",
    )
    plot_confusion_matrix(
        ax=axes[0][1],
        fig=fig,
        confusion_matrix=comparison_statistics["confusion_matrix"],
    )
    plot_answerability_counts(
        ax=axes[1][0],
        web_counts=comparison_statistics["web_subquery_counts"],
        email_counts=comparison_statistics["email_subquery_counts"],
        title="Subquery-level answerability distribution",
        ylabel="Number of subqueries",
    )
    plot_l1_disagreement(
        ax=axes[1][1],
        comparison_statistics=comparison_statistics,
    )

    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "oracle_paths",
        nargs="*",
        help="Positional paths: web oracle path followed by email oracle path.",
    )
    parser.add_argument(
        "--web-oracle-path",
        default=None,
        help="Path to the web-source oracle_discriminator.json.",
    )
    parser.add_argument(
        "--email-oracle-path",
        default=None,
        help="Path to the email-source oracle_discriminator.json.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output PNG path. Defaults to oracle_source_complementarity.png next to the email oracle file.",
    )
    args = parser.parse_args()

    if args.oracle_paths and len(args.oracle_paths) != 2:
        raise ValueError(
            "plot_oracle_source_complementarity: expected either no positional paths "
            "or two positional paths: web oracle path and email oracle path"
        )

    web_candidate_path = args.web_oracle_path
    email_candidate_path = args.email_oracle_path
    if args.oracle_paths:
        web_candidate_path, email_candidate_path = args.oracle_paths
    elif web_candidate_path is None or email_candidate_path is None:
        raise ValueError(
            "plot_oracle_source_complementarity: provide two positional oracle paths "
            "or both --web-oracle-path and --email-oracle-path"
        )

    web_oracle_path = resolve_existing_path(
        web_candidate_path,
        project_root,
        "web oracle",
        "plot_oracle_source_complementarity",
    )
    email_oracle_path = resolve_existing_path(
        email_candidate_path,
        project_root,
        "email oracle",
        "plot_oracle_source_complementarity",
    )
    if args.output_path:
        output_path = Path(args.output_path)
        if not output_path.is_absolute():
            output_path = Path(project_root) / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    web_data = load_json(web_oracle_path)
    email_data = load_json(email_oracle_path)
    comparison_statistics = collect_comparison_statistics(
        web_data=web_data,
        email_data=email_data,
    )
    output_path = save_comparison_plot(
        web_oracle_path=web_oracle_path,
        email_oracle_path=email_oracle_path,
        output_path=output_path,
        web_data=web_data,
        email_data=email_data,
        comparison_statistics=comparison_statistics,
    )

    print(
        "plot_oracle_source_complementarity: plot saved:\n"
        f"\tweb oracle path: {format_path(web_oracle_path, project_root)}\n"
        f"\temail oracle path: {format_path(email_oracle_path, project_root)}\n"
        f"\toutput path: {format_path(output_path, project_root)}\n"
        f"\tpaired queries: {comparison_statistics['n_paired_results']}\n"
        f"\tweb query counts: {comparison_statistics['web_query_counts']}\n"
        f"\temail query counts: {comparison_statistics['email_query_counts']}\n"
        f"\tunion query counts: {comparison_statistics['union_query_counts']}\n"
        f"\tweb subquery counts: {comparison_statistics['web_subquery_counts']}\n"
        f"\temail subquery counts: {comparison_statistics['email_subquery_counts']}"
    )

if __name__ == "__main__":
    main()
