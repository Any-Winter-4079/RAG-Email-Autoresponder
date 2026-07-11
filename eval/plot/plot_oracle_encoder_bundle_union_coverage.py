import sys
from os.path import dirname, abspath

# python eval/plot/plot_oracle_encoder_bundle_union_coverage.py lm_cleaned_text_chunks dev 2026-06-04_14-51-58
# python eval/plot/plot_oracle_encoder_bundle_union_coverage.py lm_cleaned_text_chunks dev 2026-06-16_23-50-58

project_root = abspath(dirname(dirname(dirname(__file__))))
sys.path.insert(0, project_root)

import argparse
import itertools
import textwrap
from pathlib import Path
import matplotlib.pyplot as plt

from helpers.oracle_support import (
    get_answerability_label,
    get_supporting_chunk_data,
    load_json,
)
from helpers.general import resolve_oracle_discriminator_path

ENCODER_ORDER = [
    "bm25",
    "splade",
    "bge_m3_muia_sparse",
    "bge_m3_muia_dense",
    "qwen3_embedding_0_6b",
    "jina_v5_text_small",
]
ENCODER_TO_DISPLAY = {
    "bm25": "BM25",
    "splade": "SPLADE",
    "bge_m3_muia_sparse": "M3 sparse",
    "bge_m3_muia_dense": "M3 dense",
    "qwen3_embedding_0_6b": "Qwen3",
    "jina_v5_text_small": "Jina",
}
DATA_SOURCES = ["web", "email"]
LABEL_ORDER = ["1", "0"]
LABEL_TO_DISPLAY = {
    "1": "Fully answerable",
    "0": "Partially answerable",
}
LABEL_TO_COLOR = {
    "1": "#C9E37B",
    "0": "#FFD071",
}
TOP_N_BUNDLES_PER_SIZE = 2
ORIGIN_ENCODER_ALIASES = {
    "bge_m3_sparse": "bge_m3_muia_sparse",
    "bge_m3_dense": "bge_m3_muia_dense",
    "qwen3_embedding_0": "qwen3_embedding_0_6b",
}

def is_encoder_retrieval_oracle(oracle_data):
    if oracle_data.get("oracle_input_mode") != "retrieval":
        return False
    metadata = oracle_data.get("oracle_input_metadata") or {}
    retrieval_output_names = {
        ORIGIN_ENCODER_ALIASES.get(retrieval_output_name, retrieval_output_name)
        for retrieval_output_name in metadata.get("retrieval_output_names") or []
    }
    return set(retrieval_output_names) == set(ENCODER_ORDER)

def get_bundle_combinations_by_size():
    return {
        bundle_size: list(itertools.combinations(ENCODER_ORDER, bundle_size))
        for bundle_size in range(1, len(ENCODER_ORDER) + 1)
    }

def collect_bundle_statistics(oracle_data):
    bundle_combinations_by_size = get_bundle_combinations_by_size()
    coverage_counts = {
        label: {
            bundle: 0
            for bundles in bundle_combinations_by_size.values()
            for bundle in bundles
        }
        for label in LABEL_ORDER
    }
    skipped_weight = {
        label: 0
        for label in LABEL_ORDER
    }
    label_counts = {
        label: 0
        for label in LABEL_ORDER
    }

    for oracle_result in oracle_data.get("results") or []:
        label = get_answerability_label(oracle_result, LABEL_ORDER)
        if label is None:
            continue
        label_counts[label] += 1

        supporting_chunk_data = get_supporting_chunk_data(
            oracle_result=oracle_result,
            encoder_order=ENCODER_ORDER,
            data_sources=DATA_SOURCES,
            encoder_aliases=ORIGIN_ENCODER_ALIASES,
            include_payload=False,
        )
        if not supporting_chunk_data:
            continue
        supporting_chunk_weight = 1 / len(supporting_chunk_data)

        for supporting_chunk in supporting_chunk_data:
            chunk_encoders = supporting_chunk["encoders"]
            if not chunk_encoders:
                skipped_weight[label] += supporting_chunk_weight
                continue
            for bundles in bundle_combinations_by_size.values():
                for bundle in bundles:
                    if chunk_encoders.intersection(bundle):
                        coverage_counts[label][bundle] += supporting_chunk_weight

    return {
        "bundle_combinations_by_size": bundle_combinations_by_size,
        "coverage_counts": coverage_counts,
        "label_counts": label_counts,
        "skipped_weight": skipped_weight,
        "n_oracle_results": len(oracle_data.get("results") or []),
    }

def get_selected_bundles(statistics):
    coverage_counts = statistics["coverage_counts"]
    bundle_combinations_by_size = statistics["bundle_combinations_by_size"]
    selected_bundles = []

    for bundle_size in range(1, len(ENCODER_ORDER)):
        ranked_bundles = sorted(
            bundle_combinations_by_size[bundle_size],
            key=lambda bundle: (
                coverage_counts["1"][bundle] + 0.5 * coverage_counts["0"][bundle],
                coverage_counts["1"][bundle],
                coverage_counts["0"][bundle],
            ),
            reverse=True,
        )
        selected_bundles.extend(ranked_bundles[:TOP_N_BUNDLES_PER_SIZE])
    selected_bundles.append(tuple(ENCODER_ORDER))
    return selected_bundles

def get_previous_size_best_counts(statistics):
    coverage_counts = statistics["coverage_counts"]
    bundle_combinations_by_size = statistics["bundle_combinations_by_size"]
    previous_size_best_counts = {
        label: {}
        for label in LABEL_ORDER
    }
    for label in LABEL_ORDER:
        for bundle_size in range(1, len(ENCODER_ORDER) + 1):
            if bundle_size == 1:
                previous_size_best_counts[label][bundle_size] = None
                continue
            previous_size_best_counts[label][bundle_size] = max(
                coverage_counts[label][bundle]
                for bundle in bundle_combinations_by_size[bundle_size - 1]
            )
    return previous_size_best_counts

def get_bundle_label(bundle):
    label = f"{len(bundle)}: " + " + ".join(
        ENCODER_TO_DISPLAY[encoder]
        for encoder in bundle
    )
    if len(bundle) >= 4:
        label = textwrap.fill(
            label,
            width=30,
            break_long_words=False,
            break_on_hyphens=False,
        )
    return label

def format_coverage_value(value):
    return f"{value:.1f}"

def format_delta_suffix(value, previous_value):
    if not previous_value:
        return ""
    delta = (value - previous_value) / previous_value * 100
    return f" ({delta:+.0f}%)"

def print_single_encoder_performance_table(statistics):
    coverage_counts = statistics["coverage_counts"]
    rows = []
    for encoder in ENCODER_ORDER:
        bundle = (encoder,)
        full_count = coverage_counts["1"][bundle]
        partial_count = coverage_counts["0"][bundle]
        aggregate_count = full_count + 0.5 * partial_count
        rows.append(
            (
                ENCODER_TO_DISPLAY[encoder],
                full_count,
                partial_count,
                aggregate_count,
            )
        )

    print("\nsingle encoder performance")
    print(f"{'Encoder':<10} {'Full':>8} {'Partial':>8} {'Aggregate':>10}")
    for encoder_display, full_count, partial_count, aggregate_count in rows:
        print(
            f"{encoder_display:<10} "
            f"{full_count:>8.3f} "
            f"{partial_count:>8.3f} "
            f"{aggregate_count:>10.3f}"
        )

def plot_encoder_bundle_coverage(data_variant, oracle_data, oracle_path, statistics, output_path=None):
    if output_path is None:
        output_path = oracle_path.with_name("oracle_encoder_bundle_union_coverage.png")

    coverage_counts = statistics["coverage_counts"]
    label_counts = statistics["label_counts"]
    selected_bundles = get_selected_bundles(statistics)
    previous_size_best_counts = get_previous_size_best_counts(statistics)

    bar_height = 0.90
    within_gap = 0.10
    group_gap = 0.62
    positions = []
    y_position = 0
    for _ in selected_bundles:
        positions.append(
            (
                y_position + bar_height / 2 + within_gap / 2,
                y_position - bar_height / 2 - within_gap / 2,
            )
        )
        y_position -= 2 * bar_height + within_gap + group_gap

    fig_height = max(8.6, len(selected_bundles) * 1.08)
    fig, ax = plt.subplots(figsize=(13.8, fig_height))

    for bundle, (full_y_position, partial_y_position) in zip(selected_bundles, positions):
        bundle_size = len(bundle)
        for label, y_position in [("1", full_y_position), ("0", partial_y_position)]:
            value = coverage_counts[label][bundle]
            ax.barh(
                y_position,
                value,
                height=bar_height,
                color=LABEL_TO_COLOR[label],
                edgecolor="white",
                linewidth=0.9,
                label=LABEL_TO_DISPLAY[label] if bundle == selected_bundles[0] else None,
            )
            delta_suffix = format_delta_suffix(
                value=value,
                previous_value=previous_size_best_counts[label][bundle_size],
            )
            ax.text(
                value + 1.2,
                y_position,
                f"{format_coverage_value(value)}{delta_suffix}",
                va="center",
                ha="left",
                fontsize=9.7,
            )

    mid_positions = [
        (full_y_position + partial_y_position) / 2
        for full_y_position, partial_y_position in positions
    ]
    ax.set_yticks(mid_positions)
    ax.set_yticklabels(
        [
            get_bundle_label(bundle)
            for bundle in selected_bundles
        ],
        fontsize=10.3,
    )
    ax.set_xlabel("Sample-based score", fontsize=12)
    ax.set_xlim(0, max(label_counts.values()) * 1.2)
    ax.grid(axis="x", alpha=0.22)
    ax.legend(loc="lower right", ncol=2, frameon=False, fontsize=10.5)
    ax.set_title(
        "Best encoder bundles of sizes 1 through 6\n"
        f"(variant: {data_variant}, oracle: {oracle_path.parent.parent.name})",
        fontsize=15,
        pad=32,
    )
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
        help="Oracle split name, e.g. dev, train, or test.",
    )
    parser.add_argument(
        "timestamp",
        help="Oracle run timestamp, e.g. 2026-06-04_12-00-00.",
    )
    parser.add_argument(
        "--output-path",
        default=None,
        help="Output PNG path. Defaults next to the oracle discriminator file.",
    )
    args = parser.parse_args()

    oracle_path = resolve_oracle_discriminator_path(
        project_root=project_root,
        split_name=args.split_name,
        variant=args.data_variant,
        timestamp=args.timestamp,
    )
    oracle_data = load_json(oracle_path)
    if not is_encoder_retrieval_oracle(oracle_data):
        raise ValueError(
            "plot_oracle_encoder_bundle_union_coverage: expected a retrieval-mode "
            f"encoder oracle file:\n\t{oracle_path}"
        )
    if args.output_path:
        output_path = Path(args.output_path)
        if not output_path.is_absolute():
            output_path = Path(project_root) / output_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_path = None

    statistics = collect_bundle_statistics(oracle_data)
    print_single_encoder_performance_table(statistics)
    output_path = plot_encoder_bundle_coverage(
        data_variant=args.data_variant,
        oracle_data=oracle_data,
        oracle_path=oracle_path,
        statistics=statistics,
        output_path=output_path,
    )
    print(
        "plot_oracle_encoder_bundle_union_coverage: plot saved:\n"
        f"\toracle path: {oracle_path}\n"
        f"\toutput path: {output_path}\n"
        f"\tlabel counts: {statistics['label_counts']}\n"
        f"\tskipped weight: {statistics['skipped_weight']}"
    )

if __name__ == "__main__":
    main()
