import sys
from os.path import dirname, abspath
project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

import os
import json
from pathlib import Path
from dotenv import load_dotenv
from helpers.email_agent import transform_env_csv_into_list
from helpers.data import (
    build_samples_grouped_by_thread_and_folderURI,
    split_samples_by_split_name,
    save_split_summary_plot,
)
from config.data import (
    THREADS_PATH,
    TRAIN_SPLIT_PCT,
    DEV_SPLIT_PCT,
    SPLIT_DATASETS_DIR,
    TRAIN_THREADS_DATASET_PATH,
    DEV_THREADS_DATASET_PATH,
    TEST_THREADS_DATASET_PATH,
    SHUFFLE_SEED,
    THREAD_GROUPING_STRATEGY,
)

load_dotenv()
my_email_addresses = transform_env_csv_into_list(os.getenv("MY_EMAIL_ADDRESSES", ""))

def main():
    print("\n"+"="*50)
    print("Step 0")
    print("="*50)
    with open(THREADS_PATH, mode="r", encoding="utf-8") as json_file:
        threads = json.load(json_file)
    print(f"\nTotal threads in {THREADS_PATH}: {len(threads)}")

    print("\n"+"="*50)
    print("Step 1: building samples grouped by thread and folderURI")
    print("="*50)
    folder_uri_to_thread_samples_groups = build_samples_grouped_by_thread_and_folderURI(
        threads,
        my_email_addresses,
    )
    folder_uri_to_sample_count = {
        folder_uri: sum(len(thread_samples) for thread_samples in thread_samples_groups)
        for folder_uri, thread_samples_groups in folder_uri_to_thread_samples_groups.items()
    }
    sample_count = sum(folder_uri_to_sample_count.values())
    print(f"\nSamples from {THREADS_PATH}: {sample_count}")

    print("\n"+"="*50)
    print("Step 2: splitting samples into train/dev/test")
    print("="*50)
    split_name_to_samples_from_messages_with_threads = split_samples_by_split_name(
        folder_uri_to_thread_samples_groups,
        TRAIN_SPLIT_PCT,
        DEV_SPLIT_PCT,
        SHUFFLE_SEED
    )

    split_name_to_split_count = {
        split_name: len(samples)
        for split_name, samples in split_name_to_samples_from_messages_with_threads.items()
    }
    split_name_to_thread_count = {}
    print(
        "\nsplit_dataset: done\n"
        f"\ttrain: {split_name_to_split_count['train']}\n"
        f"\tdev: {split_name_to_split_count['dev']}\n"
        f"\ttest: {split_name_to_split_count['test']}"
    )

    print("\n"+"="*50)
    print("Step 3: storing split samples as JSON")
    print("="*50)
    split_datasets_dir = Path(SPLIT_DATASETS_DIR)
    split_datasets_dir.mkdir(parents=True, exist_ok=True)

    removed_paths = []
    for pattern in ("*.json", "*.jsonl"):
        for existing_path in split_datasets_dir.glob(pattern):
            if existing_path.is_file():
                # delete the file from disk
                existing_path.unlink()
                removed_paths.append(existing_path.name)

    if removed_paths:
        print(
            "\nRemoved previous split artifacts:\n"
            + "\n".join(f"\t{path_name}" for path_name in sorted(removed_paths))
        )

    for split_name, samples in split_name_to_samples_from_messages_with_threads.items():
        split_samples_path = f"{SPLIT_DATASETS_DIR}/{split_name}.json"
        split_threads_path = {
            "train": TRAIN_THREADS_DATASET_PATH,
            "dev": DEV_THREADS_DATASET_PATH,
            "test": TEST_THREADS_DATASET_PATH,
        }[split_name]

        split_thread_keys = {
            (sample["folder_uri"], sample["thread_id"])
            for sample in samples
        }
        split_threads = [
            thread
            for thread in threads
            if (thread["folder_uri"], thread["thread_id"]) in split_thread_keys
        ]
        split_name_to_thread_count[split_name] = len(split_threads)

        with open(split_samples_path, mode="w", encoding="utf-8") as json_file:
            json.dump(samples, json_file, ensure_ascii=False, indent=2)

        with open(split_threads_path, mode="w", encoding="utf-8") as json_file:
            json.dump(split_threads, json_file, ensure_ascii=False, indent=2)

        print(
            f"\nSaved {split_name} split:\n"
            f"\tsamples: {len(samples)}\n"
            f"\tthreads: {len(split_threads)}\n"
            f"\tsamples path: {split_samples_path}\n"
            f"\tthreads path: {split_threads_path}"
        )

    save_split_summary_plot(
        split_name_to_sample_count=split_name_to_split_count,
        split_name_to_thread_count=split_name_to_thread_count,
        title=f"Split dataset summary\n({THREAD_GROUPING_STRATEGY})",
        output_path=split_datasets_dir / "split_summary.png",
    )

if __name__ == "__main__":
    main()
