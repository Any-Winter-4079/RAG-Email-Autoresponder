import json
from pathlib import Path
from statistics import mean

project_root = Path(__file__).resolve().parent.parent
train_data_dir = project_root / "data" / "finetune" / "2026-06-04_18-35-44"
intermediate_data_path = train_data_dir / "train_intermediate.json"
train_data_path = train_data_dir / "train.jsonl"

with open(intermediate_data_path, "r", encoding="utf-8") as intermediate_data_file:
    intermediate_rows = json.load(intermediate_data_file)

finetune_rows = []
with open(train_data_path, "r", encoding="utf-8") as train_data_file:
    for line in train_data_file:
        finetune_rows.append(json.loads(line))

oracle_negative_counts = []
retrieval_mined_negative_counts = []
for intermediate_row in intermediate_rows:
    selected_negatives = set(intermediate_row["negatives"])
    oracle_negatives = set()
    for source, negative_texts in intermediate_row["negatives_by_source"].items():
        if source.startswith("oracle:"):
            oracle_negatives.update(negative_texts)
    selected_oracle_negatives = selected_negatives.intersection(oracle_negatives)
    oracle_negative_counts.append(len(selected_oracle_negatives))
    retrieval_mined_negative_counts.append(len(selected_negatives) - len(selected_oracle_negatives))

query_texts = set()
positive_texts = set()
negative_texts = set()
for finetune_row in finetune_rows:
    query_texts.add(finetune_row["query"])
    positive_texts.update(finetune_row["pos"])
    negative_texts.update(finetune_row["neg"])

positive_and_negative_texts = positive_texts.intersection(negative_texts)
positive_reuse_percentage = len(positive_and_negative_texts) / len(positive_texts) * 100
negative_reuse_percentage = len(positive_and_negative_texts) / len(negative_texts) * 100

print("Negative source statistics")
print(
    "Oracle insufficient chunks: "
    f"min {min(oracle_negative_counts)}, "
    f"max {max(oracle_negative_counts)}, "
    f"mean {mean(oracle_negative_counts):.2f}"
)
print(
    "Retrieval-mined chunks: "
    f"min {min(retrieval_mined_negative_counts)}, "
    f"max {max(retrieval_mined_negative_counts)}, "
    f"mean {mean(retrieval_mined_negative_counts):.2f}"
)

print("\nUnique text and cross-role reuse statistics")
print(f"Unique query texts: {len(query_texts):,}")
print(f"Unique positive chunk texts: {len(positive_texts):,}")
print(f"Unique negative chunk texts: {len(negative_texts):,}")
print(f"Positive chunk texts also used as negatives: {positive_reuse_percentage:.2f}%")
print(f"Negative chunk texts also used as positives: {negative_reuse_percentage:.2f}%")
