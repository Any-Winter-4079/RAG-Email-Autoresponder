import sys
from os.path import dirname, abspath

project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

# python data/prepare_m3_finetune_dataset.py

import json
from datetime import datetime
from pathlib import Path

from config.fine_tune import (
    M3_FINETUNE_DATA_VARIANTS,
    M3_FINETUNE_N_INSUFFICIENT_NEGATIVES_PER_SAMPLE,
    M3_FINETUNE_N_NEGATIVES_PER_SAMPLE,
    M3_FINETUNE_N_QUERIES_PER_SAMPLE,
    M3_FINETUNE_ORACLE_DISCRIMINATOR_SOURCE_TO_TIMESTAMP,
    M3_FINETUNE_QUERY_TYPE_TO_WEIGHT,
    M3_FINETUNE_QUERY_TYPES,
    M3_FINETUNE_QUERY_QUERY_AUGMENTATION_RATIO,
    M3_FINETUNE_RANDOM_SEED,
    M3_FINETUNE_RETRIEVAL_ENCODERS,
    M3_FINETUNE_RETRIEVAL_SOURCES,
    M3_FINETUNE_RETRIEVAL_TIMESTAMP,
    M3_FINETUNE_ORACLE_DISCRIMINATOR_SPLIT,
    M3_FINETUNE_TOP_K_RETRIEVAL_MINED_NEGATIVES_PER_FILE,
)
from config.eval import (
    DATA_VARIANT_CONTEXT_EMAILS_MODE,
    QUERY_REWRITE_CACHE_DIR,
)
from helpers.data import build_finetune_rows
from helpers.general import (
    resolve_query_rewrite_cache_path,
    resolve_oracle_discriminator_path,
    resolve_data_variant_eval_output_path,
)

def main():
    dataset_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(project_root) / "data" / "finetune" / dataset_timestamp
    output_path = output_dir / "train.jsonl"
    intermediate_output_path = output_dir / "train_intermediate.json"

    query_rewrite_cache_path = resolve_query_rewrite_cache_path(
        project_root=project_root,
        query_rewrite_cache_dir=QUERY_REWRITE_CACHE_DIR,
        split_name=M3_FINETUNE_ORACLE_DISCRIMINATOR_SPLIT,
        context_emails_mode=DATA_VARIANT_CONTEXT_EMAILS_MODE,
        n_eval_samples_per_folder_uri=None,
    )
    with open(query_rewrite_cache_path, "r", encoding="utf-8") as query_rewrite_cache_file:
        query_rewrite_entries = json.load(query_rewrite_cache_file)

    data_variant_to_source_to_oracle_results = {}
    data_variant_to_source_to_encoder_results = {}

    for data_variant in M3_FINETUNE_DATA_VARIANTS:
        data_variant_to_source_to_oracle_results[data_variant] = {}
        for data_source, oracle_timestamp in M3_FINETUNE_ORACLE_DISCRIMINATOR_SOURCE_TO_TIMESTAMP.items():
            oracle_results_path = resolve_oracle_discriminator_path(
                project_root=project_root,
                split_name=M3_FINETUNE_ORACLE_DISCRIMINATOR_SPLIT,
                variant=data_variant,
                timestamp=oracle_timestamp,
                data_sources=[data_source],
                input_mode="retrieval",
            )
            with open(oracle_results_path, "r", encoding="utf-8") as oracle_results_file:
                oracle_output = json.load(oracle_results_file)
            data_variant_to_source_to_oracle_results[data_variant][data_source] = oracle_output["results"]

        data_variant_to_source_to_encoder_results[data_variant] = {}
        for data_source in M3_FINETUNE_RETRIEVAL_SOURCES:
            data_variant_to_source_to_encoder_results[data_variant][data_source] = {}
            for encoder_name in M3_FINETUNE_RETRIEVAL_ENCODERS:
                encoder_results_path = resolve_data_variant_eval_output_path(
                    project_root=project_root,
                    split_name=M3_FINETUNE_ORACLE_DISCRIMINATOR_SPLIT,
                    variant=data_variant,
                    source_name=data_source,
                    output_name=encoder_name,
                    timestamp=M3_FINETUNE_RETRIEVAL_TIMESTAMP,
                )
                with open(encoder_results_path, "r", encoding="utf-8") as encoder_results_file:
                    encoder_output = json.load(encoder_results_file)
                data_variant_to_source_to_encoder_results[data_variant][data_source][encoder_name] = encoder_output["results"]

    intermediate_rows, finetune_rows = build_finetune_rows(
        query_rewrite_entries=query_rewrite_entries,
        data_variant_to_source_to_oracle_results=data_variant_to_source_to_oracle_results,
        data_variant_to_source_to_encoder_results=data_variant_to_source_to_encoder_results,
        query_types=M3_FINETUNE_QUERY_TYPES,
        query_type_to_weight=M3_FINETUNE_QUERY_TYPE_TO_WEIGHT,
        query_query_augmentation_ratio=M3_FINETUNE_QUERY_QUERY_AUGMENTATION_RATIO,
        n_queries_per_sample=M3_FINETUNE_N_QUERIES_PER_SAMPLE,
        n_negatives_per_sample=M3_FINETUNE_N_NEGATIVES_PER_SAMPLE,
        n_insufficient_negatives_per_sample=M3_FINETUNE_N_INSUFFICIENT_NEGATIVES_PER_SAMPLE,
        random_seed=M3_FINETUNE_RANDOM_SEED,
        top_k_retrieval_mined_negatives_per_file=M3_FINETUNE_TOP_K_RETRIEVAL_MINED_NEGATIVES_PER_FILE,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with open(intermediate_output_path, "w", encoding="utf-8") as intermediate_output_file:
        json.dump(intermediate_rows, intermediate_output_file, ensure_ascii=False, indent=2)
    with open(output_path, "w", encoding="utf-8") as output_file:
        for finetune_row in finetune_rows:
            output_file.write(json.dumps(finetune_row, ensure_ascii=False) + "\n")

    print(
        "prepare_m3_finetune_dataset: done\n"
        f"\toracle split: {M3_FINETUNE_ORACLE_DISCRIMINATOR_SPLIT}\n"
        f"\tquery types: {M3_FINETUNE_QUERY_TYPES}\n"
        f"\tquery type weights: {M3_FINETUNE_QUERY_TYPE_TO_WEIGHT}\n"
        f"\tquery-query augmentation ratio: {M3_FINETUNE_QUERY_QUERY_AUGMENTATION_RATIO}\n"
        f"\tqueries per sample: {M3_FINETUNE_N_QUERIES_PER_SAMPLE}\n"
        f"\tnegatives per sample: {M3_FINETUNE_N_NEGATIVES_PER_SAMPLE}\n"
        f"\tinsufficient negatives per sample: {M3_FINETUNE_N_INSUFFICIENT_NEGATIVES_PER_SAMPLE}\n"
        f"\tretrieval-mined negatives per file: {M3_FINETUNE_TOP_K_RETRIEVAL_MINED_NEGATIVES_PER_FILE}\n"
        f"\tdata variants: {M3_FINETUNE_DATA_VARIANTS}\n"
        f"\tquery rewrite cache path: {query_rewrite_cache_path.relative_to(project_root)}\n"
        f"\tintermediate output path: {intermediate_output_path.relative_to(project_root)}\n"
        f"\tfinal output path: {output_path.relative_to(project_root)}"
    )

if __name__ == "__main__":
    main()
