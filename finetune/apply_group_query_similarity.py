import json
import shutil
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.fine_tune import M3_FINETUNE_SIMILAR_GROUP_MIN_SCORE
from helpers.finetune import get_condensed_index

# default_train_data_dir = project_root / "data" / "finetune" / "2026-06-13_00-00-30"
default_train_data_dir = project_root / "data" / "finetune" / "2026-06-13_16-36-15"
default_scores_path = project_root / "finetune" / "query_query_scores_upper.npy"

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train-data-dir", default=str(default_train_data_dir))
    parser.add_argument("--scores-path", default=str(default_scores_path))
    parser.add_argument("--similar-group-min-score", type=float, default=M3_FINETUNE_SIMILAR_GROUP_MIN_SCORE)
    return parser.parse_args()

def load_intermediate_rows(train_data_dir):
    intermediate_data_path = train_data_dir / "train_intermediate.json"
    with open(intermediate_data_path, "r", encoding="utf-8") as intermediate_data_file:
        return json.load(intermediate_data_file)

def load_scores(scores_path, n_groups):
    scores = np.load(scores_path)
    n_pairs = n_groups * (n_groups - 1) // 2
    if scores.shape[0] != n_pairs:
        raise ValueError(
            "apply_group_query_similarity: query similarity scores must match the intermediate row order; "
            f"expected {n_pairs} scores, got {scores.shape[0]}"
        )
    return scores

def get_group_id_to_similar_group_ids(scores, intermediate_rows, min_score):
    n_groups = len(intermediate_rows)
    group_id_to_similar_group_ids = {
        intermediate_row["group_id"]: []
        for intermediate_row in intermediate_rows
    }
    for source_index, source_row in enumerate(intermediate_rows[:-1]):
        source_group_id = source_row["group_id"]
        for target_index in range(source_index + 1, n_groups):
            target_group_id = intermediate_rows[target_index]["group_id"]
            score = scores[get_condensed_index(source_index, target_index, n_groups)]
            if score >= min_score:
                group_id_to_similar_group_ids[source_group_id].append(target_group_id)
                group_id_to_similar_group_ids[target_group_id].append(source_group_id)

    return {
        group_id: sorted(similar_group_ids)
        for group_id, similar_group_ids in group_id_to_similar_group_ids.items()
    }

def write_train_jsonl_with_similar_group_ids(train_data_dir, output_dir, group_id_to_similar_group_ids):
    train_data_path = train_data_dir / "train.jsonl"
    output_path = output_dir / "train.jsonl"
    n_finetune_rows = 0
    with open(train_data_path, "r", encoding="utf-8") as input_file, open(output_path, "w", encoding="utf-8") as output_file:
        for line in input_file:
            finetune_row = json.loads(line)
            finetune_row["similar_group_ids"] = group_id_to_similar_group_ids.get(
                finetune_row["group_id"],
                [],
            )
            output_file.write(json.dumps(finetune_row, ensure_ascii=False) + "\n")
            n_finetune_rows += 1

    return output_path, n_finetune_rows

def main():
    args = parse_args()
    train_data_dir = Path(args.train_data_dir)
    if not train_data_dir.is_absolute():
        train_data_dir = project_root / train_data_dir
    scores_path = Path(args.scores_path)
    if not scores_path.is_absolute():
        scores_path = project_root / scores_path
    output_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = project_root / "data" / "finetune" / output_timestamp
    output_dir.mkdir(parents=True, exist_ok=True)

    intermediate_rows = load_intermediate_rows(train_data_dir)
    print(
        "apply_group_query_similarity: assuming score matrix matches train_intermediate.json row order",
        flush=True,
    )
    scores = load_scores(scores_path, len(intermediate_rows))
    group_id_to_similar_group_ids = get_group_id_to_similar_group_ids(
        scores=scores,
        intermediate_rows=intermediate_rows,
        min_score=args.similar_group_min_score,
    )
    shutil.copyfile(train_data_dir / "train_intermediate.json", output_dir / "train_intermediate.json")
    output_path, n_finetune_rows = write_train_jsonl_with_similar_group_ids(
        train_data_dir=train_data_dir,
        output_dir=output_dir,
        group_id_to_similar_group_ids=group_id_to_similar_group_ids,
    )

    print(
        "apply_group_query_similarity: done\n"
        f"\ttrain data dir: {train_data_dir.relative_to(project_root)}\n"
        f"\tscores path: {scores_path.relative_to(project_root)}\n"
        f"\tsimilar group min score: {args.similar_group_min_score}\n"
        f"\tn fine-tune rows: {n_finetune_rows}\n"
        f"\tfinal output path: {output_path.relative_to(project_root)}",
        flush=True,
    )

if __name__ == "__main__":
    main()
