import sys
from os.path import dirname, abspath

# modal run eval/run_dump_collection_payloads.py --collection-names '["lm_cleaned_text_chunks","lm_summary_chunks","lm_q_and_a_chunks"]'
# modal run eval/run_dump_collection_payloads.py --collection-names '*'
# modal run eval/run_dump_collection_payloads.py --collection-names '["lm_cleaned_text_chunks","lm_summary_chunks","lm_q_and_a_chunks"]' --include-vectors True

project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

import json
import modal
from datetime import datetime
from pathlib import Path

from config.crawler_agent import ENCODE_VARIANTS
from config.data import EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX
from config.eval import (
    DUMP_COLLECTION_PAYLOADS_DECODER_TOKEN_DISTRIBUTION_PLOT_FILENAME,
    DUMP_COLLECTION_PAYLOADS_ENCODER_TOKEN_DISTRIBUTION_PLOT_FILENAME,
    DUMP_COLLECTION_PAYLOADS_SIZE_COMPARISON_PLOT_FILENAME,
    RESULTS_DIR_NAME,
)
from config.modal_apps import (
    COLLECTION_HANDLER_APP_NAME,
    DUMP_COLLECTION_PAYLOADS_EVAL_APP_NAME,
)
from config.modal_functions import DUMP_COLLECTION_PAYLOADS_FUNCTION_NAME
from helpers.eval import (
    get_collection_point_token_count,
    save_collection_dump_size_comparison_plot,
    save_collection_dump_token_distribution_plot,
    write_eval_output_to_file,
)

app = modal.App(DUMP_COLLECTION_PAYLOADS_EVAL_APP_NAME)

@app.local_entrypoint()
def run_dump_collection_payloads(
        collection_names=None,
        include_vectors=False,
        page_size=1024,
        ):
    # parse collection names
    if isinstance(collection_names, str):
        stripped_collection_names = collection_names.strip()
        if stripped_collection_names.startswith("["):
            collection_names = json.loads(stripped_collection_names)
        else:
            collection_names = [stripped_collection_names]
    if collection_names is None or "*" in collection_names:
        collection_names = list(ENCODE_VARIANTS)
    expanded_collection_names = []
    for collection_name in collection_names:
        expanded_collection_names.append(collection_name)
        if collection_name.startswith("lm_"):
            expanded_collection_names.append(
                f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}{collection_name}"
            )
    collection_names = expanded_collection_names

    dump_collection_payloads = modal.Function.from_name(
        COLLECTION_HANDLER_APP_NAME,
        DUMP_COLLECTION_PAYLOADS_FUNCTION_NAME,
    )

    # create results directory
    script_name = Path(__file__).stem
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_dir = (
        Path(project_root)
        / "eval"
        / RESULTS_DIR_NAME
        / script_name
        / timestamp
    )
    results_dir.mkdir(parents=True, exist_ok=True)

    output_name = "dump_with_vectors" if include_vectors else "dump"
    collection_name_to_n_points = {}
    collection_name_to_decoder_token_counts = {}
    collection_name_to_encoder_token_counts = {}

    # dump collection payloads
    for collection_name in collection_names:
        dumped_points = dump_collection_payloads.remote(
            collection_name=collection_name,
            include_vectors=include_vectors,
            page_size=page_size,
        )

        data_variant_results_dir = results_dir / collection_name
        data_variant_results_dir.mkdir(parents=True, exist_ok=True)

        write_eval_output_to_file(
            data_variant_results_dir=data_variant_results_dir,
            output_name=output_name,
            eval_output=dumped_points,
            data_variant=collection_name,
        )
        collection_name_to_n_points[collection_name] = len(dumped_points)
        collection_name_to_decoder_token_counts[collection_name] = [
            get_collection_point_token_count(
                payload=dumped_point["payload"] or {},
                token_type="decoder",
            )
            for dumped_point in dumped_points
        ]
        collection_name_to_encoder_token_counts[collection_name] = [
            get_collection_point_token_count(
                payload=dumped_point["payload"] or {},
                token_type="encoder",
            )
            for dumped_point in dumped_points
        ]

        print(
            "run_dump_collection_payloads: dumped collection payloads:\n"
            f"\tcollection: {collection_name}\n"
            f"\tinclude_vectors: {include_vectors}\n"
            f"\tpage_size: {page_size}\n"
            f"\tn points: {len(dumped_points)}\n"
            f"\tdecoder tokens: min={min(collection_name_to_decoder_token_counts[collection_name]):,} | "
            f"mean={sum(collection_name_to_decoder_token_counts[collection_name]) / len(collection_name_to_decoder_token_counts[collection_name]):.1f} | "
            f"max={max(collection_name_to_decoder_token_counts[collection_name]):,}\n"
            f"\tencoder tokens: min={min(collection_name_to_encoder_token_counts[collection_name]):,} | "
            f"mean={sum(collection_name_to_encoder_token_counts[collection_name]) / len(collection_name_to_encoder_token_counts[collection_name]):.1f} | "
            f"max={max(collection_name_to_encoder_token_counts[collection_name]):,}"
        )

    # store collection dump plots
    plot_path = results_dir / DUMP_COLLECTION_PAYLOADS_SIZE_COMPARISON_PLOT_FILENAME
    save_collection_dump_size_comparison_plot(
        collection_name_to_n_points=collection_name_to_n_points,
        output_path=plot_path,
    )
    token_distribution_x_axis_max = max([
        token_count
        for collection_name_to_token_counts in [
            collection_name_to_decoder_token_counts,
            collection_name_to_encoder_token_counts,
        ]
        for token_counts in collection_name_to_token_counts.values()
        for token_count in token_counts
    ])
    decoder_plot_path = (
        results_dir
        / DUMP_COLLECTION_PAYLOADS_DECODER_TOKEN_DISTRIBUTION_PLOT_FILENAME
    )
    save_collection_dump_token_distribution_plot(
        collection_name_to_token_counts=collection_name_to_decoder_token_counts,
        output_path=decoder_plot_path,
        token_type="decoder",
        x_axis_max=token_distribution_x_axis_max,
    )
    encoder_plot_path = (
        results_dir
        / DUMP_COLLECTION_PAYLOADS_ENCODER_TOKEN_DISTRIBUTION_PLOT_FILENAME
    )
    save_collection_dump_token_distribution_plot(
        collection_name_to_token_counts=collection_name_to_encoder_token_counts,
        output_path=encoder_plot_path,
        token_type="encoder",
        x_axis_max=token_distribution_x_axis_max,
    )
    print(
        "run_dump_collection_payloads: saved collection dump plots:\n"
        f"\tsize comparison: {plot_path.relative_to(project_root)}\n"
        f"\tdecoder token distribution: {decoder_plot_path.relative_to(project_root)}\n"
        f"\tencoder token distribution: {encoder_plot_path.relative_to(project_root)}"
    )
