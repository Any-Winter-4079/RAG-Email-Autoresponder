import sys
from os.path import dirname, abspath

project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

# python data/prepare_m3_finetune_dataset.py

import json
from datetime import datetime
from pathlib import Path

from config.data import (
    M3_FINETUNE_DATA_VARIANTS,
    M3_FINETUNE_QUERY_TYPES,
    M3_FINETUNE_ORACLE_DISCRIMINATOR_SPLIT,
    M3_FINETUNE_VARIANT_TO_ORACLE_DISCRIMINATOR_TIMESTAMP,
    M3_FINETUNE_VARIANT_TO_RRF_TIMESTAMP,
)
from helpers.data import build_finetune_rows
from helpers.general import (
    resolve_oracle_discriminator_path,
    resolve_data_variant_eval_output_path,
)

def main():
    dataset_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = Path(project_root) / "data" / "finetune" / dataset_timestamp
    output_path = output_dir / "train.jsonl"
    intermediate_output_path = output_dir / "train_intermediate.json"

    data_variant_to_oracle_results = {}
    data_variant_to_rrf_results = {}

    for data_variant in M3_FINETUNE_DATA_VARIANTS:
        oracle_results_path = resolve_oracle_discriminator_path(
            project_root=project_root,
            split_name=M3_FINETUNE_ORACLE_DISCRIMINATOR_SPLIT,
            variant=data_variant,
            timestamp=M3_FINETUNE_VARIANT_TO_ORACLE_DISCRIMINATOR_TIMESTAMP.get(data_variant),
        )
        with open(oracle_results_path, "r", encoding="utf-8") as oracle_results_file:
            oracle_output = json.load(oracle_results_file)
        data_variant_to_oracle_results[data_variant] = oracle_output["results"]

        rrf_results_path = resolve_data_variant_eval_output_path(
            project_root=project_root,
            split_name=M3_FINETUNE_ORACLE_DISCRIMINATOR_SPLIT,
            variant=data_variant,
            output_name="rrf",
            timestamp=M3_FINETUNE_VARIANT_TO_RRF_TIMESTAMP.get(data_variant),
        )
        with open(rrf_results_path, "r", encoding="utf-8") as rrf_results_file:
            rrf_output = json.load(rrf_results_file)
        data_variant_to_rrf_results[data_variant] = rrf_output["results"]

    intermediate_rows, finetune_rows = build_finetune_rows(
        data_variant_to_oracle_results=data_variant_to_oracle_results,
        data_variant_to_rrf_results=data_variant_to_rrf_results,
        query_types=M3_FINETUNE_QUERY_TYPES,
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
        f"\tdata variants: {M3_FINETUNE_DATA_VARIANTS}\n"
        f"\tintermediate output path: {intermediate_output_path.relative_to(project_root)}\n"
        f"\tfinal output path: {output_path.relative_to(project_root)}"
    )

if __name__ == "__main__":
    main()
