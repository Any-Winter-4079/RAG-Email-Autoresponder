#!/usr/bin/env python3

import json
import time
from pathlib import Path
from statistics import mean

from transformers import AutoTokenizer

train_data_path = sorted((Path(__file__).resolve().parent.parent / "data" / "finetune").glob("*/train.jsonl"))[-1]
sft_path = Path(__file__).resolve().parent / "sft.sh"
tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", trust_remote_code=True)
progress_interval = 1_000

query_texts = set()
passage_texts = set()

with open(train_data_path, "r", encoding="utf-8") as train_data_file:
    for line in train_data_file:
        row = json.loads(line)
        query_texts.add(row["query"])
        passage_texts.update(row["pos"])
        passage_texts.update(row["neg"])

def get_token_lengths(texts, label):
    lengths = []
    texts = sorted(texts)
    n_texts = len(texts)
    print(f"tokenizing {label}: {n_texts} unique texts")
    start_time = time.monotonic()
    last_print_time = start_time
    for index, text in enumerate(texts, start=1):
        lengths.append(len(tokenizer.encode(text, add_special_tokens=False)))
        if index % progress_interval == 0 or index == n_texts:
            current_time = time.monotonic()
            print(
                f"tokenizing {label}: {index}/{n_texts} "
                f"(interval {current_time - last_print_time:.2f}s, "
                f"total {current_time - start_time:.2f}s)"
            )
            last_print_time = current_time
    return lengths

query_lengths = get_token_lengths(query_texts, "queries")
passage_lengths = get_token_lengths(passage_texts, "passages")

max_query_length = max(query_lengths)
max_passage_length = max(passage_lengths)

sft_lines = sft_path.read_text().splitlines()
for index, line in enumerate(sft_lines):
    if "--query_max_len " in line:
        sft_lines[index] = f"    --query_max_len {max_query_length} \\"
    if "--passage_max_len " in line:
        sft_lines[index] = f"    --passage_max_len {max_passage_length} \\"
sft_path.write_text("\n".join(sft_lines) + "\n")

print(f"train_data_path: {train_data_path}")
print(f"n_unique_queries: {len(query_texts)}")
print(f"min_query_length: {min(query_lengths)}")
print(f"max_query_length: {max_query_length}")
print(f"mean_query_length: {mean(query_lengths):.3f}")
print(f"n_unique_passages: {len(passage_texts)}")
print(f"min_passage_length: {min(passage_lengths)}")
print(f"max_passage_length: {max_passage_length}")
print(f"mean_passage_length: {mean(passage_lengths):.3f}")
