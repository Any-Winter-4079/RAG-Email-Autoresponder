import json
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path

# python finetune/compute_m3_scores.py --model models/sampled_negs-exp8-bs8-effective_-1-filters_off --run-description sampled_negs-exp8-bs8-effective_-1-filters_off

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FlagEmbedding"))

split_name = "dev"
data_variant = "lm_summary_chunks"
oracle_timestamps = [
    "2026-05-14_19-31-38",
    "2026-05-15_01-52-23",
]
score_keys = ["dense", "sparse"]
batch_size = 16
max_query_length = 512
max_passage_length = 2048
results_dir = project_root / "finetune" / "m3_score_results"

def resolve_oracle_discriminator_path(project_root, split_name, variant, timestamp):
    return (
        Path(project_root)
        / "eval"
        / "results"
        / "run_oracle_discriminator"
        / split_name
        / timestamp
        / variant
        / "oracle_discriminator.json"
    )

def get_text_from_payload(payload):
    if "question" in payload and "answer" not in payload:
        return payload["question"]
    if "question" in payload and "answer" in payload:
        return f"Q: {payload['question']}\nA: {payload['answer']}"
    return payload["text"]

oracle_paths = [
    resolve_oracle_discriminator_path(
        project_root=project_root,
        split_name=split_name,
        variant=data_variant,
        timestamp=oracle_timestamp,
    )
    for oracle_timestamp in oracle_timestamps
]

def get_oracle_chunk_text(chunk):
    return get_text_from_payload(chunk["payload"]).strip()

def load_oracle_labeled_samples():
    samples = []
    for oracle_path in oracle_paths:
        with open(oracle_path, "r", encoding="utf-8") as oracle_file:
            oracle_output = json.load(oracle_file)
        for result in oracle_output["results"]:
            if result.get("generation_failed"):
                continue
            reranker_query = result["reranker_query"]
            discriminator_result = result["discriminator_result"]
            positives = []
            negatives = []
            for subquery in discriminator_result.get("subqueries") or []:
                positives.extend(
                    get_oracle_chunk_text(chunk)
                    for chunk in subquery["supporting_chunks"]
                    if chunk["payload"] is not None
                )
                negatives.extend(
                    get_oracle_chunk_text(chunk)
                    for chunk in subquery["insufficient_chunks"]
                    if chunk["payload"] is not None
                )
            positives = sorted(set(positives))
            negatives = sorted(set(negatives) - set(positives))
            if reranker_query and positives and negatives:
                samples.append({
                    "query": reranker_query,
                    "positives": positives,
                    "negatives": negatives,
                })
    return samples

def score_samples(model_name_or_path, samples):
    from FlagEmbedding import BGEM3FlagModel

    model = BGEM3FlagModel(model_name_or_path, use_fp16=True)
    sentence_pairs = []
    sample_slices = []
    for sample in samples:
        passages = sample["positives"] + sample["negatives"]
        sample_slices.append((len(sentence_pairs), len(sample["positives"]), len(sample["negatives"])))
        sentence_pairs.extend((sample["query"], passage) for passage in passages)

    scores = model.compute_score(
        sentence_pairs,
        batch_size=batch_size,
        max_query_length=max_query_length,
        max_passage_length=max_passage_length,
        weights_for_different_modes=[1.0, 1.0, 0.0],
    )
    
    scored_samples = []
    for start_index, n_positives, n_negatives in sample_slices:
        end_index = start_index + n_positives + n_negatives
        scored_samples.append({
            "n_positives": n_positives,
            "n_negatives": n_negatives,
            "scores": {
                score_key: scores[score_key][start_index:end_index]
                for score_key in score_keys
            },
        })
    return scored_samples

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-description", required=True)
    return parser.parse_args()

args = parse_args()
samples = load_oracle_labeled_samples()
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
print("oracle label paths:")
for oracle_path in oracle_paths:
    print(f"\t{oracle_path.relative_to(project_root)}")
print(f"model: {args.model}")
print(f"n samples: {len(samples)}")
n_positive_chunks = sum(len(sample["positives"]) for sample in samples)
n_negative_chunks = sum(len(sample["negatives"]) for sample in samples)
print(
    f"supporting chunks: {n_positive_chunks} total, "
    f"{n_positive_chunks / len(samples):.2f} per sample"
)
print(
    f"insufficient chunks: {n_negative_chunks} total, "
    f"{n_negative_chunks / len(samples):.2f} per sample"
)
print(f"supporting/insufficient ratio: {n_positive_chunks / n_negative_chunks:.2f}")

scored_samples = score_samples(args.model, samples)

results_dir.mkdir(parents=True, exist_ok=True)
result_path = results_dir / f"{run_timestamp}.json"
with open(result_path, "w", encoding="utf-8") as result_file:
    json.dump(
        {
            "timestamp": run_timestamp,
            "run_description": args.run_description,
            "model": args.model,
            "split_name": split_name,
            "data_variant": data_variant,
            "label_source": "oracle_discriminator_supporting_vs_insufficient_chunks",
            "score_source": "BGEM3FlagModel.compute_score",
            "oracle_timestamps": oracle_timestamps,
            "oracle_paths": [str(oracle_path.relative_to(project_root)) for oracle_path in oracle_paths],
            "score_keys": score_keys,
            "batch_size": batch_size,
            "max_query_length": max_query_length,
            "max_passage_length": max_passage_length,
            "n_samples": len(samples),
            "n_positive_chunks": n_positive_chunks,
            "n_negative_chunks": n_negative_chunks,
            "positive_chunks_per_sample": n_positive_chunks / len(samples),
            "negative_chunks_per_sample": n_negative_chunks / len(samples),
            "positive_to_negative_chunk_ratio": n_positive_chunks / n_negative_chunks,
            "samples": scored_samples,
        },
        result_file,
        ensure_ascii=False,
        indent=2,
    )
print(f"\nwrote M3 score results to {result_path.relative_to(project_root)}")
