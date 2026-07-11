import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, stdev

project_root = Path(__file__).resolve().parent.parent
train_data_path = project_root / "data" / "finetune" / "2026-06-13_17-03-11" / "train.jsonl"
batch_sizes = [1, 2, 4, 8, 16, 32]
n_simulations = 100
random_seed = 10_000

passage_text_to_id = {}

def get_passage_id(passage_text):
    if passage_text not in passage_text_to_id:
        passage_text_to_id[passage_text] = len(passage_text_to_id)
    return passage_text_to_id[passage_text]

samples = []
with open(train_data_path, "r", encoding="utf-8") as train_data_file:
    for line in train_data_file:
        sample = json.loads(line)
        samples.append({
            "group_id": sample["group_id"],
            "similar_group_ids": set(sample.get("similar_group_ids", [])),
            "positives": [get_passage_id(passage) for passage in sample["pos"]],
            "negatives": [get_passage_id(passage) for passage in sample["neg"]],
        })

def get_n_collisions_by_kind_per_sample(batch_size, simulation_index):
    # with kinds:
    # duplicate_positive
    # group_positive
    # similar_group_positive
    rng = random.Random(random_seed + simulation_index)
    sample_indices = list(range(len(samples)))
    rng.shuffle(sample_indices)
    sample_indices = sample_indices[:len(sample_indices) - len(sample_indices) % batch_size] # given --dataloader_drop_last True
    n_duplicate_positive_collisions = 0
    n_group_positive_collisions = 0
    n_similar_group_positive_collisions = 0

    for batch_start in range(0, len(sample_indices), batch_size):
        # assuming the full in-batch denominator, without max-effective-size capping
        batch_samples = []
        batch_passages = []
        group_id_to_positive_ids = defaultdict(set)
        for sample_index in sample_indices[batch_start:batch_start + batch_size]:
            sample = samples[sample_index]
            selected_positive = rng.choice(sample["positives"])
            batch_samples.append({
                **sample,
                "selected_positive": selected_positive,
            })
            batch_passages.extend([selected_positive, *sample["negatives"]])
            group_id_to_positive_ids[sample["group_id"]].update(sample["positives"])

        batch_passage_counts = Counter(batch_passages)
        group_id_to_collision_count = {
            group_id: sum(
                batch_passage_counts[positive_id]
                for positive_id in positive_ids
            ) - 1
            for group_id, positive_ids in group_id_to_positive_ids.items()
        }

        similar_group_ids_to_collision_count = {}
        batch_group_ids = set(group_id_to_positive_ids.keys())
        for sample in batch_samples:
            # frozenset is used only so this group-id set can be cached as a dictionary key
            similar_group_ids_to_mask_out = frozenset({
                sample["group_id"],
                *(sample["similar_group_ids"] & batch_group_ids),
            })
            if similar_group_ids_to_mask_out not in similar_group_ids_to_collision_count:
                similar_positive_ids_to_mask_out = set()
                for group_id in similar_group_ids_to_mask_out:
                    similar_positive_ids_to_mask_out.update(group_id_to_positive_ids[group_id])
                similar_group_ids_to_collision_count[similar_group_ids_to_mask_out] = sum(
                    batch_passage_counts[positive_id]
                    for positive_id in similar_positive_ids_to_mask_out
                ) - 1

            n_duplicate_positive_collisions += batch_passage_counts[sample["selected_positive"]] - 1
            n_group_positive_collisions += group_id_to_collision_count[sample["group_id"]]
            n_similar_group_positive_collisions += similar_group_ids_to_collision_count[similar_group_ids_to_mask_out]

    return {
        "duplicate_positive": n_duplicate_positive_collisions / len(sample_indices),
        "group_positive": n_group_positive_collisions / len(sample_indices),
        "similar_group_positive": n_similar_group_positive_collisions / len(sample_indices),
    }

for batch_size in batch_sizes:
    n_collisions_by_kind_per_sample = [
        get_n_collisions_by_kind_per_sample(batch_size, simulation_index)
        for simulation_index in range(n_simulations)
    ]
    duplicate_positive_collision_values = [
        result["duplicate_positive"]
        for result in n_collisions_by_kind_per_sample
    ]
    group_positive_collision_values = [
        result["group_positive"]
        for result in n_collisions_by_kind_per_sample
    ]
    similar_group_positive_collision_values = [
        result["similar_group_positive"]
        for result in n_collisions_by_kind_per_sample
    ]
    print(
        f"batch size {batch_size}: "
        f"{mean(duplicate_positive_collision_values):.4f} ± {stdev(duplicate_positive_collision_values):.4f} positive, "
        f"{mean(group_positive_collision_values):.4f} ± {stdev(group_positive_collision_values):.4f} positive + group, "
        f"{mean(similar_group_positive_collision_values):.4f} ± {stdev(similar_group_positive_collision_values):.4f} positive + group + similar collisions per sample"
    )
