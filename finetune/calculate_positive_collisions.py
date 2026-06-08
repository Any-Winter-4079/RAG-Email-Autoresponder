import json
import random
from pathlib import Path
from statistics import mean

project_root = Path(__file__).resolve().parent.parent
train_data_path = project_root / "data" / "finetune" / "2026-05-31_13-04-46" / "train.jsonl"
batch_sizes = [1, 2, 4, 8, 16, 32]
n_simulations = 100
random_seed = 10_000

samples = []
with open(train_data_path, "r", encoding="utf-8") as train_data_file:
    for line in train_data_file:
        sample = json.loads(line)
        samples.append({"positives": sample["pos"], "negatives": sample["neg"]})

def get_n_collisions_per_sample(batch_size, simulation_index):
    rng = random.Random(random_seed + simulation_index)
    sample_indices = list(range(len(samples)))
    rng.shuffle(sample_indices)
    sample_indices = sample_indices[:len(sample_indices) - len(sample_indices) % batch_size] # given --dataloader_drop_last True
    n_collisions = 0

    for batch_start in range(0, len(sample_indices), batch_size):
        batch_positive_and_negatives = []
        for sample_index in sample_indices[batch_start:batch_start + batch_size]:
            sample = samples[sample_index]
            selected_positive = rng.choice(sample["positives"])
            batch_positive_and_negatives.append((selected_positive, sample["negatives"]))

        for sample_index, (selected_positive, sample_negatives) in enumerate(batch_positive_and_negatives):
            negatives_for_selected_positive = sample_negatives.copy()
            for other_sample_index, (other_positive, other_negatives) in enumerate(batch_positive_and_negatives):
                if other_sample_index != sample_index:
                    negatives_for_selected_positive.append(other_positive)
                    negatives_for_selected_positive.extend(other_negatives)
            n_collisions += negatives_for_selected_positive.count(selected_positive)

    return n_collisions / len(sample_indices)

for batch_size in batch_sizes:
    n_collisions_per_sample = mean(
        get_n_collisions_per_sample(batch_size, simulation_index)
        for simulation_index in range(n_simulations)
    )
    print(f"batch size {batch_size}: {n_collisions_per_sample:.4f} collisions per sample")
