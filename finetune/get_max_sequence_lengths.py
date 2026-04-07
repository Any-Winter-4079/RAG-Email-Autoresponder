#!/usr/bin/env python3

import json
from pathlib import Path

from transformers import AutoTokenizer

train_data_path = sorted((Path(__file__).resolve().parent.parent / "data" / "finetune").glob("*/train.jsonl"))[-1]
sft_path = Path(__file__).resolve().parent / "sft.sh"
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", trust_remote_code=True)
max_query_length = 0
max_positive_length = 0
max_negative_length = 0

with open(train_data_path, "r", encoding="utf-8") as train_data_file:
    for line in train_data_file:
        row = json.loads(line)
        max_query_length = max(max_query_length, len(tokenizer.encode(row["query"], add_special_tokens=False)))
        max_positive_length = max(max_positive_length, max(len(tokenizer.encode(text, add_special_tokens=False)) for text in row["pos"]))
        max_negative_length = max(max_negative_length, max(len(tokenizer.encode(text, add_special_tokens=False)) for text in row["neg"]))

max_passage_length = max(max_positive_length, max_negative_length)

sft_lines = sft_path.read_text().splitlines()
for index, line in enumerate(sft_lines):
    if "--query_max_len " in line:
        sft_lines[index] = f"    --query_max_len {max_query_length} \\"
    if "--passage_max_len " in line:
        sft_lines[index] = f"    --passage_max_len {max_passage_length} \\"
sft_path.write_text("\n".join(sft_lines) + "\n")

print(f"train_data_path: {train_data_path}")
print(f"max_query_length: {max_query_length}")
print(f"max_positive_length: {max_positive_length}")
print(f"max_negative_length: {max_negative_length}")
print(f"max_passage_length: {max_passage_length}")
