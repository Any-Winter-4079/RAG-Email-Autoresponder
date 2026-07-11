import json
import sys
from argparse import ArgumentParser
from pathlib import Path

import modal
import numpy as np

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from config.eval import RERANKER_NAME
from config.modal_apps import ENCODER_GPU_APP_NAME
from config.modal_functions import RUN_ENCODER_GPU_RERANKER_FUNCTION_NAME
from helpers.finetune import get_condensed_index

# default_train_data_dir = project_root / "data" / "finetune" / "2026-06-13_00-00-30"
default_train_data_dir = project_root / "data" / "finetune" / "2026-06-13_16-36-15"
default_top_k_edges_per_group = 20
default_log_every = 25

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--train-data-dir", default=str(default_train_data_dir))
    parser.add_argument("--reranker-name", default=RERANKER_NAME)
    parser.add_argument("--top-k-edges-per-group", type=int, default=default_top_k_edges_per_group)
    parser.add_argument("--log-every", type=int, default=default_log_every)
    return parser.parse_args()

def load_intermediate_rows(train_data_dir):
    intermediate_data_path = train_data_dir / "train_intermediate.json"
    with open(intermediate_data_path, "r", encoding="utf-8") as intermediate_data_file:
        return json.load(intermediate_data_file)

def load_groups(intermediate_rows):
    groups = []
    for intermediate_row in intermediate_rows:
        reranker_queries = intermediate_row["queries"].get("reranker") or []
        if not reranker_queries:
            continue
        groups.append({
            "group_id": intermediate_row["group_id"],
            "reranker_query": reranker_queries[0],
        })
    return groups

def write_top_edges(scores, groups, output_path, top_k_edges_per_group):
    n_groups = len(groups)
    with open(output_path, "w", encoding="utf-8") as output_file:
        for source_index, source_group in enumerate(groups):
            group_edges = []
            for target_index, target_group in enumerate(groups):
                if target_index == source_index:
                    continue
                score = float(scores[get_condensed_index(source_index, target_index, n_groups)])
                group_edges.append((score, target_group))
            group_edges.sort(reverse=True, key=lambda edge: edge[0])
            for rank, (score, target_group) in enumerate(group_edges[:top_k_edges_per_group], start=1):
                output_file.write(json.dumps({
                    "source_group_id": source_group["group_id"],
                    "target_group_id": target_group["group_id"],
                    "rank": rank,
                    "score": score,
                    "source_query": source_group["reranker_query"],
                    "target_query": target_group["reranker_query"],
                }, ensure_ascii=False) + "\n")

def main():
    args = parse_args()
    train_data_dir = Path(args.train_data_dir)
    if not train_data_dir.is_absolute():
        train_data_dir = project_root / train_data_dir
    scores_path = project_root / "finetune" / "query_query_scores_upper.npy"
    top_edges_path = project_root / "finetune" / "query_query_top_edges.jsonl"

    intermediate_rows = load_intermediate_rows(train_data_dir)
    groups = load_groups(intermediate_rows)
    n_groups = len(groups)
    n_pairs = n_groups * (n_groups - 1) // 2

    print(
        "calculate_group_query_similarity: scoring query-query pairs\n"
        f"\ttrain data dir: {train_data_dir.relative_to(project_root)}\n"
        f"\treranker name: {args.reranker_name}\n"
        f"\tn groups: {n_groups}\n"
        f"\tn undirected pairs: {n_pairs}",
        flush=True,
    )

    run_encoder_gpu_reranker = modal.Function.from_name(
        ENCODER_GPU_APP_NAME,
        RUN_ENCODER_GPU_RERANKER_FUNCTION_NAME,
    )
    scores = np.empty(n_pairs, dtype=np.float32)
    write_index = 0
    for source_index, source_group in enumerate(groups[:-1]):
        target_groups = groups[source_index + 1:]
        target_queries = [
            target_group["reranker_query"]
            for target_group in target_groups
        ]
        source_scores = run_encoder_gpu_reranker.remote(
            args.reranker_name,
            source_group["reranker_query"],
            target_queries,
        )
        if len(source_scores) != len(target_queries):
            raise ValueError(
                "calculate_group_query_similarity: reranker returned wrong number of scores; "
                f"expected {len(target_queries)}, got {len(source_scores)}"
            )
        scores[write_index:write_index + len(source_scores)] = source_scores
        write_index += len(source_scores)
        if (
                source_index == 0 or
                (source_index + 1) % args.log_every == 0 or
                source_index == n_groups - 2
                ):
            print(
                f"\tsource group {source_index + 1}/{n_groups - 1}: "
                f"{write_index}/{n_pairs} pairs scored",
                flush=True,
            )

    np.save(scores_path, scores)
    write_top_edges(
        scores=scores,
        groups=groups,
        output_path=top_edges_path,
        top_k_edges_per_group=args.top_k_edges_per_group,
    )

    print(
        "calculate_group_query_similarity: done\n"
        f"\tscores path: {scores_path.relative_to(project_root)}\n"
        f"\ttop edges path: {top_edges_path.relative_to(project_root)}",
        flush=True,
    )

if __name__ == "__main__":
    main()
