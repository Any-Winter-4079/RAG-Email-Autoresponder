import json
import math
import sys
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "FlagEmbedding"))

from config.encoder import EMBEDDING_ENCODERS
from helpers.eval import get_text_from_payload
from helpers.general import resolve_oracle_discriminator_path
from FlagEmbedding import BGEM3FlagModel

split_name = "dev"
data_variant = "lm_summary_chunks"
oracle_timestamps = [
    "2026-05-14_19-31-38",
    "2026-05-15_01-52-23",
]
oracle_paths = [
    resolve_oracle_discriminator_path(
        project_root=project_root,
        split_name=split_name,
        variant=data_variant,
        timestamp=oracle_timestamp,
    )
    for oracle_timestamp in oracle_timestamps
]
encoder_names = ["bge_m3", "bge_m3_muia"]
score_keys = ["dense", "sparse", "sparse+dense"]
score_weights_for_different_modes = [2 / 3, 1 / 3, 0.0]
max_examples = None
batch_size = 16
max_query_length = 512
max_passage_length = 2048
results_dir = project_root / "finetune" / "oracle_score_results"

def get_oracle_chunk_text(chunk):
    return get_text_from_payload(chunk["payload"]).strip()

def load_oracle_samples():
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
    if max_examples is not None:
        samples = samples[:max_examples]
    return samples

def score_samples(model_name, samples):
    model_config = EMBEDDING_ENCODERS[model_name]
    model = BGEM3FlagModel(
        model_config["model_name"],
        use_fp16=model_config.get("use_fp16", False),
    )
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
        weights_for_different_modes=score_weights_for_different_modes,
    )

    model_results = {}
    for score_key in score_keys:
        score_values = scores[score_key]
        n_scores = len(score_values)
        n_nan_scores = sum(math.isnan(score) for score in score_values)
        n_positive_inf_scores = sum(score == math.inf for score in score_values)
        n_negative_inf_scores = sum(score == -math.inf for score in score_values)
        n_non_finite_scores = n_nan_scores + n_positive_inf_scores + n_negative_inf_scores
        n_samples_with_non_finite_scores = 0
        pair_accuracies = []
        mean_margins = []
        best_margins = []
        reciprocal_ranks = []
        for start_index, n_positives, n_negatives in sample_slices:
            sample_scores = scores[score_key][start_index:start_index + n_positives + n_negatives]
            if any(not math.isfinite(score) for score in sample_scores):
                n_samples_with_non_finite_scores += 1
            positive_scores = sample_scores[:n_positives]
            negative_scores = sample_scores[n_positives:]
            finite_positive_scores = [score for score in positive_scores if math.isfinite(score)]
            finite_negative_scores = [score for score in negative_scores if math.isfinite(score)]
            pair_accuracies.append(mean(
                1 if positive_score > negative_score else 0.5 if positive_score == negative_score else 0
                for positive_score in positive_scores
                for negative_score in negative_scores
            ))
            if finite_positive_scores and finite_negative_scores:
                mean_margins.append(mean(finite_positive_scores) - mean(finite_negative_scores))
                best_margins.append(max(finite_positive_scores) - max(finite_negative_scores))
            ranked_labels = [
                label
                for _, label in sorted(
                    [(score, 1) for score in positive_scores] + [(score, 0) for score in negative_scores],
                    reverse=True,
                )
            ]
            reciprocal_ranks.append(1 / (ranked_labels.index(1) + 1))

        model_results[score_key] = {
            "pair_accuracy": mean(pair_accuracies),
            "mean_margin": mean(mean_margins),
            "best_margin": mean(best_margins),
            "mrr": mean(reciprocal_ranks),
            "n_scores": n_scores,
            "n_non_finite_scores": n_non_finite_scores,
            "n_nan_scores": n_nan_scores,
            "n_positive_inf_scores": n_positive_inf_scores,
            "n_negative_inf_scores": n_negative_inf_scores,
            "n_samples_with_non_finite_scores": n_samples_with_non_finite_scores,
            "non_finite_score_rate": n_non_finite_scores / n_scores,
            "non_finite_sample_rate": n_samples_with_non_finite_scores / len(sample_slices),
        }
    return model_results

def get_run_description():
    parser = ArgumentParser()
    parser.add_argument("--run-description", default=None)
    args = parser.parse_args()
    if args.run_description is not None:
        return args.run_description
    return input("Oracle score run description: ").strip()

samples = load_oracle_samples()
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
run_description = get_run_description()
print("oracle paths:")
for oracle_path in oracle_paths:
    print(f"\t{oracle_path.relative_to(project_root)}")
print(f"n samples: {len(samples)}")
print(f"score weights: dense={score_weights_for_different_modes[0]:.4f}, sparse={score_weights_for_different_modes[1]:.4f}, colbert={score_weights_for_different_modes[2]:.4f}")
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

results = {
    encoder_name: score_samples(encoder_name, samples)
    for encoder_name in encoder_names
}
for encoder_name, encoder_results in results.items():
    print(f"\n{encoder_name}:")
    for score_key, metrics in encoder_results.items():
        print(
            f"\t{score_key}: "
            f"pair_accuracy {metrics['pair_accuracy']:.4f}, "
            f"mean_margin {metrics['mean_margin']:.4f}, "
            f"best_margin {metrics['best_margin']:.4f}, "
            f"mrr {metrics['mrr']:.4f}, "
            f"non_finite_scores {metrics['n_non_finite_scores']}/{metrics['n_scores']} "
            f"({metrics['non_finite_score_rate']:.2%}; "
            f"nan {metrics['n_nan_scores']}, "
            f"+inf {metrics['n_positive_inf_scores']}, "
            f"-inf {metrics['n_negative_inf_scores']}), "
            f"samples_with_non_finite {metrics['n_samples_with_non_finite_scores']}/{len(samples)} "
            f"({metrics['non_finite_sample_rate']:.2%})"
        )

base_encoder_name, tuned_encoder_name = encoder_names
comparison_results = {}
print(f"\n{tuned_encoder_name} - {base_encoder_name}:")
for score_key in score_keys:
    comparison_results[score_key] = {
        "pair_accuracy": results[tuned_encoder_name][score_key]["pair_accuracy"] - results[base_encoder_name][score_key]["pair_accuracy"],
        "mean_margin": results[tuned_encoder_name][score_key]["mean_margin"] - results[base_encoder_name][score_key]["mean_margin"],
        "best_margin": results[tuned_encoder_name][score_key]["best_margin"] - results[base_encoder_name][score_key]["best_margin"],
        "mrr": results[tuned_encoder_name][score_key]["mrr"] - results[base_encoder_name][score_key]["mrr"],
    }
    print(
        f"\t{score_key}: "
        f"pair_accuracy {comparison_results[score_key]['pair_accuracy']:.4f}, "
        f"mean_margin {comparison_results[score_key]['mean_margin']:.4f}, "
        f"best_margin {comparison_results[score_key]['best_margin']:.4f}, "
        f"mrr {comparison_results[score_key]['mrr']:.4f}"
    )

results_dir.mkdir(parents=True, exist_ok=True)
result_path = results_dir / f"{run_timestamp}.json"
with open(result_path, "w", encoding="utf-8") as result_file:
    json.dump(
        {
            "timestamp": run_timestamp,
            "run_description": run_description,
            "split_name": split_name,
            "data_variant": data_variant,
            "oracle_timestamps": oracle_timestamps,
            "oracle_paths": [str(oracle_path.relative_to(project_root)) for oracle_path in oracle_paths],
            "encoder_names": encoder_names,
            "score_keys": score_keys,
            "score_weights_for_different_modes": score_weights_for_different_modes,
            "max_examples": max_examples,
            "batch_size": batch_size,
            "max_query_length": max_query_length,
            "max_passage_length": max_passage_length,
            "n_samples": len(samples),
            "n_positive_chunks": n_positive_chunks,
            "n_negative_chunks": n_negative_chunks,
            "positive_chunks_per_sample": n_positive_chunks / len(samples),
            "negative_chunks_per_sample": n_negative_chunks / len(samples),
            "positive_to_negative_chunk_ratio": n_positive_chunks / n_negative_chunks,
            "results": results,
            "comparison": comparison_results,
        },
        result_file,
        ensure_ascii=False,
        indent=2,
    )
print(f"\nwrote oracle score results to {result_path.relative_to(project_root)}")
