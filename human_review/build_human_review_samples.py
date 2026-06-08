import json
import random
import sys
from copy import deepcopy
from datetime import datetime
from os.path import abspath, dirname
from pathlib import Path

# python human_review/build_human_review_samples.py

project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

from config.eval import (
    DATA_VARIANT_CONTEXT_EMAILS_MODE,
    DATA_VARIANT_TEST_SPLIT_NAME,
    QUERY_REWRITE_CACHE_DIR,
)
from config.llm_judge import ORACLE_DISCRIMINATOR_QUERY_REWRITE_CACHE_FILENAME
from config.human_review import (
    HUMAN_REVIEW_DATA_VARIANT,
    HUMAN_REVIEW_N_EVAL_SAMPLES_PER_FOLDER_URI,
    HUMAN_REVIEW_N_SAMPLES,
    HUMAN_REVIEW_NO_REQUEST_CATEGORY_PLACEHOLDER,
    HUMAN_REVIEW_ORACLE_ANSWERABILITY_CATEGORY_PLACEHOLDER,
    HUMAN_REVIEW_ORACLE_DISCRIMINATOR_TIMESTAMP,
    HUMAN_REVIEW_RANDOM_SEED,
    HUMAN_REVIEW_RESULTS_DIR,
)
from helpers.general import (
    resolve_oracle_discriminator_path,
    resolve_query_rewrite_cache_path,
)

def load_json(path):
    with open(path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)

def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as json_file:
        json.dump(data, json_file, ensure_ascii=False, indent=2)

def sample_items(items, rng):
    if len(items) <= HUMAN_REVIEW_N_SAMPLES:
        return deepcopy(items)
    selected_indexes = sorted(rng.sample(range(len(items)), HUMAN_REVIEW_N_SAMPLES))
    return [
        deepcopy(items[index])
        for index in selected_indexes
    ]

def build_no_request_review(no_requests_path, rng):
    no_requests = load_json(no_requests_path)
    review_entries = sample_items(no_requests, rng)
    for review_entry in review_entries:
        review_entry["human_review_no_request_is_correct"] = False
        review_entry["human_review_no_request_category"] = HUMAN_REVIEW_NO_REQUEST_CATEGORY_PLACEHOLDER
        review_entry["human_review_notes"] = ""
    return review_entries

def build_anonymized_request_review(query_rewrite_cache_path, rng):
    request_entries = load_json(query_rewrite_cache_path)
    source_results = [
        {
            "sample": request_entry["sample"],
            "anonymized_request": request_entry["anonymized_request"],
        }
        for request_entry in request_entries
        if request_entry.get("anonymized_request")
    ]
    review_entries = sample_items(source_results, rng)
    for review_entry in review_entries:
        review_entry["human_review_anonymized_request_is_good"] = False
        review_entry["human_review_notes"] = ""
    return review_entries

def build_oracle_review(oracle_path, rng):
    oracle_data = load_json(oracle_path)
    source_results = [
        result
        for result in oracle_data.get("results", [])
        if result.get("discriminator_result")
    ]
    review_results = sample_items(source_results, rng)
    for review_result in review_results:
        discriminator_result = review_result.get("discriminator_result") or {}
        for subquery in discriminator_result.get("subqueries") or []:
            subquery["human_review_subquery_exists"] = False
            subquery["human_review_answerability"] = "1|0|-1"
            subquery["human_review_answerability_category"] = HUMAN_REVIEW_ORACLE_ANSWERABILITY_CATEGORY_PLACEHOLDER
    return {
        **{
            key: value
            for key, value in oracle_data.items()
            if key != "results"
        },
        "n_samples": len(review_results),
        "results": review_results,
    }

def main():
    rng = random.Random(HUMAN_REVIEW_RANDOM_SEED)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = (
        Path(project_root)
        / HUMAN_REVIEW_RESULTS_DIR
        / DATA_VARIANT_TEST_SPLIT_NAME
        / timestamp
    )

    query_rewrite_cache_path = resolve_query_rewrite_cache_path(
        project_root=project_root,
        query_rewrite_cache_dir=QUERY_REWRITE_CACHE_DIR,
        split_name=DATA_VARIANT_TEST_SPLIT_NAME,
        context_emails_mode=DATA_VARIANT_CONTEXT_EMAILS_MODE,
        n_eval_samples_per_folder_uri=HUMAN_REVIEW_N_EVAL_SAMPLES_PER_FOLDER_URI,
        configured_cache_filename=ORACLE_DISCRIMINATOR_QUERY_REWRITE_CACHE_FILENAME,
    )
    no_requests_path = query_rewrite_cache_path.with_name(
        f"{query_rewrite_cache_path.stem}_no_requests"
        f"{query_rewrite_cache_path.suffix}"
    )
    oracle_path = resolve_oracle_discriminator_path(
        project_root=project_root,
        split_name=DATA_VARIANT_TEST_SPLIT_NAME,
        variant=HUMAN_REVIEW_DATA_VARIANT,
        timestamp=HUMAN_REVIEW_ORACLE_DISCRIMINATOR_TIMESTAMP,
    )

    no_request_review = build_no_request_review(no_requests_path, rng)
    anonymized_request_review = build_anonymized_request_review(query_rewrite_cache_path, rng)
    oracle_review = build_oracle_review(oracle_path, rng)

    no_request_review_path = output_dir / "no_request_human_review.json"
    anonymized_request_review_path = output_dir / "anonymized_request_human_review.json"
    oracle_review_path = output_dir / "oracle_human_review.json"
    write_json(no_request_review_path, no_request_review)
    write_json(anonymized_request_review_path, anonymized_request_review)
    write_json(oracle_review_path, oracle_review)

    print(
        "build_human_review_samples: review files prepared:\n"
        f"\tno-request source: {no_requests_path.relative_to(project_root)}\n"
        f"\tquery-rewrite source: {query_rewrite_cache_path.relative_to(project_root)}\n"
        f"\toracle source: {oracle_path.relative_to(project_root)}\n"
        f"\tno-request review: {no_request_review_path.relative_to(project_root)}\n"
        f"\tanonymized-request review: {anonymized_request_review_path.relative_to(project_root)}\n"
        f"\toracle review: {oracle_review_path.relative_to(project_root)}\n"
        f"\tn no-request items: {len(no_request_review)}\n"
        f"\tn anonymized-request items: {len(anonymized_request_review)}\n"
        f"\tn oracle items: {len(oracle_review['results'])}"
    )

if __name__ == "__main__":
    main()
