import sys
from os.path import dirname, abspath
project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

import os
import csv
import json
from dotenv import load_dotenv
from helpers.email_agent import transform_env_csv_into_list
from helpers.data import (
    build_samples_by_folder_uri_and_thread_id,
    get_sample_counts_by_folder_uri,
    split_samples_by_split_name
)
from config.data import (
    MESSAGES_WITH_THREADS_DATASET_PATH,
    TRAIN_SPLIT_PCT,
    DEV_SPLIT_PCT,
    VAL_SPLIT_PCT,
    SPLIT_DATASETS_DIR
)

load_dotenv()
my_email_addresses = transform_env_csv_into_list(os.getenv("MY_EMAIL_ADDRESSES", ""))

def main():
    print("\n"+"="*50)
    print("Step 0")
    print("="*50)
    with open(MESSAGES_WITH_THREADS_DATASET_PATH, newline="") as csv_file:
        messages_with_threads_rows = list(csv.reader(csv_file, delimiter=";"))
    messages_with_threads_header = messages_with_threads_rows[0]
    messages_with_threads_data = messages_with_threads_rows[1:]
    print(f"\nTotal rows in {MESSAGES_WITH_THREADS_DATASET_PATH}: {len(messages_with_threads_data)}")

    print("\n"+"="*50)
    print("Step 1: building samples by folderURI and threadID")
    print("="*50)
    folder_uri_to_thread_id_to_samples = build_samples_by_folder_uri_and_thread_id(
        messages_with_threads_header,
        messages_with_threads_data,
        my_email_addresses
    )
    folder_uri_to_sample_count = get_sample_counts_by_folder_uri(
        folder_uri_to_thread_id_to_samples
    )
    samples_total_from_messages_with_threads = sum(folder_uri_to_sample_count.values())
    print(f"\nSamples from {MESSAGES_WITH_THREADS_DATASET_PATH}: {samples_total_from_messages_with_threads}")

    print("\n"+"="*50)
    print("Step 2: splitting samples into train/dev/val/test")
    print("="*50)
    split_name_to_samples_from_messages_with_threads = split_samples_by_split_name(
        folder_uri_to_thread_id_to_samples,
        TRAIN_SPLIT_PCT,
        DEV_SPLIT_PCT,
        VAL_SPLIT_PCT
    )

    split_name_to_split_count = {
        split_name: len(samples)
        for split_name, samples in split_name_to_samples_from_messages_with_threads.items()
    }
    print(
        "\nsplit_dataset: done\n"
        f"\ttrain: {split_name_to_split_count['train']}\n"
        f"\tdev: {split_name_to_split_count['dev']}\n"
        f"\tval: {split_name_to_split_count['val']}\n"
        f"\ttest: {split_name_to_split_count['test']}"
    )

    print("\n"+"="*50)
    print("Step 3: storing split samples as JSONL and JSON")
    print("="*50)
    os.makedirs(SPLIT_DATASETS_DIR, exist_ok=True)
    for split_name, samples in split_name_to_samples_from_messages_with_threads.items():
        jsonl_path = f"{SPLIT_DATASETS_DIR}/{split_name}.jsonl"
        json_path = f"{SPLIT_DATASETS_DIR}/{split_name}.json"

        with open(jsonl_path, mode="w", encoding="utf-8") as jsonl_file:
            for sample in samples:
                jsonl_file.write(json.dumps(sample, ensure_ascii=False) + "\n")

        with open(json_path, mode="w", encoding="utf-8") as json_file:
            json.dump(samples, json_file, ensure_ascii=False, indent=2)

        print(
            f"\nSaved {split_name} split:\n"
            f"\tsamples: {len(samples)}\n"
            f"\tjsonl: {jsonl_path}\n"
            f"\tjson: {json_path}"
        )

if __name__ == "__main__":
    main()
