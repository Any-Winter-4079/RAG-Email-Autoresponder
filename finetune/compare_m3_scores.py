import json
import math
from argparse import ArgumentParser
from datetime import datetime
from pathlib import Path
from statistics import mean



project_root = Path(__file__).resolve().parent.parent
score_results_dir = project_root / "finetune" / "m3_score_results"
comparison_results_dir = project_root / "finetune" / "m3_score_comparisons"
score_keys = ["dense", "sparse", "sparse+dense"]

def resolve_result_path(path_or_timestamp):
    path = Path(path_or_timestamp)
    if path.exists():
        return path.resolve()
    if path.suffix == ".json":
        return score_results_dir / path.name
    return score_results_dir / f"{path_or_timestamp}.json"

def load_score_result(path_or_timestamp):
    result_path = resolve_result_path(path_or_timestamp)
    with open(result_path, "r", encoding="utf-8") as result_file:
        result = json.load(result_file)
    return result_path, result

def get_sample_scores(sample, score_key, dense_weight, sparse_weight):
    if score_key in ("dense", "sparse"):
        return sample["scores"][score_key]
    dense_scores = sample["scores"]["dense"]
    sparse_scores = sample["scores"]["sparse"]
    weight_sum = dense_weight + sparse_weight
    return [
        (dense_weight * dense_score + sparse_weight * sparse_score) / weight_sum
        for dense_score, sparse_score in zip(dense_scores, sparse_scores)
    ]

def calculate_metrics(result, score_key, dense_weight, sparse_weight):
    score_values = [
        score
        for sample in result["samples"]
        for score in get_sample_scores(sample, score_key, dense_weight, sparse_weight)
    ]
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
    for sample in result["samples"]:
        sample_scores = get_sample_scores(sample, score_key, dense_weight, sparse_weight)
        if any(not math.isfinite(score) for score in sample_scores):
            n_samples_with_non_finite_scores += 1
        n_positives = sample["n_positives"]
        positive_scores = sample_scores[:n_positives]
        negative_scores = sample_scores[n_positives:]
        finite_positive_scores = [score for score in positive_scores if math.isfinite(score)]
        finite_negative_scores = [score for score in negative_scores if math.isfinite(score)]
        pair_accuracies.append(mean(
            1 if positive_score > negative_score else 0
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

    return {
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
        "non_finite_sample_rate": n_samples_with_non_finite_scores / len(result["samples"]),
    }

def validate_matching_samples(m3_result, m3_sft_result):
    if m3_result["n_samples"] != m3_sft_result["n_samples"]:
        raise ValueError("M3 and M3 SFT results have different sample counts.")
    for index, (m3_sample, m3_sft_sample) in enumerate(zip(m3_result["samples"], m3_sft_result["samples"])):
        if (
            m3_sample["n_positives"] != m3_sft_sample["n_positives"]
            or m3_sample["n_negatives"] != m3_sft_sample["n_negatives"]
        ):
            raise ValueError(f"M3 and M3 SFT sample counts differ at sample {index}.")


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--m3-result", required=True)
    parser.add_argument("--m3-sft-result", required=True)
    parser.add_argument("--dense-weight", type=float, default=2 / 3)
    parser.add_argument("--sparse-weight", type=float, default=1 / 3)
    args = parser.parse_args()
    if args.dense_weight + args.sparse_weight == 0:
        parser.error("--dense-weight and --sparse-weight cannot both be 0")
    return args

args = parse_args()
m3_result_path, m3_result = load_score_result(args.m3_result)
m3_sft_result_path, m3_sft_result = load_score_result(args.m3_sft_result)
validate_matching_samples(m3_result, m3_sft_result)
run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
score_weights_for_different_modes = [args.dense_weight, args.sparse_weight, 0.0]

print(f"M3 result: {m3_result_path.relative_to(project_root)}")
print(f"M3 SFT result: {m3_sft_result_path.relative_to(project_root)}")
print(
    f"score weights: dense={args.dense_weight:.4f}, "
    f"sparse={args.sparse_weight:.4f}, colbert=0.0000"
)

results = {
    "m3": {},
    "m3_sft": {},
}
comparison_results = {}
for score_key in score_keys:
    results["m3"][score_key] = calculate_metrics(
        m3_result, score_key, args.dense_weight, args.sparse_weight
    )
    results["m3_sft"][score_key] = calculate_metrics(
        m3_sft_result, score_key, args.dense_weight, args.sparse_weight
    )
    comparison_results[score_key] = {
        "pair_accuracy": results["m3_sft"][score_key]["pair_accuracy"] - results["m3"][score_key]["pair_accuracy"],
        "mean_margin": results["m3_sft"][score_key]["mean_margin"] - results["m3"][score_key]["mean_margin"],
        "best_margin": results["m3_sft"][score_key]["best_margin"] - results["m3"][score_key]["best_margin"],
        "mrr": results["m3_sft"][score_key]["mrr"] - results["m3"][score_key]["mrr"],
    }

for result_name, result_metrics in results.items():
    print(f"\n{result_name}:")
    for score_key, metrics in result_metrics.items():
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
            f"samples_with_non_finite {metrics['n_samples_with_non_finite_scores']}/{m3_result['n_samples']} "
            f"({metrics['non_finite_sample_rate']:.2%})"
        )

print("\nm3_sft - m3:")
for score_key, metrics in comparison_results.items():
    print(
        f"\t{score_key}: "
        f"pair_accuracy {metrics['pair_accuracy']:.4f}, "
        f"mean_margin {metrics['mean_margin']:.4f}, "
        f"best_margin {metrics['best_margin']:.4f}, "
        f"mrr {metrics['mrr']:.4f}"
    )

comparison_results_dir.mkdir(parents=True, exist_ok=True)
comparison_result_path = comparison_results_dir / f"{run_timestamp}.json"
with open(comparison_result_path, "w", encoding="utf-8") as result_file:
    json.dump(
        {
            "timestamp": run_timestamp,
            "m3_result_path": str(m3_result_path.relative_to(project_root)),
            "m3_sft_result_path": str(m3_sft_result_path.relative_to(project_root)),
            "m3_result_timestamp": m3_result["timestamp"],
            "m3_sft_result_timestamp": m3_sft_result["timestamp"],
            "m3_run_description": m3_result["run_description"],
            "m3_sft_run_description": m3_sft_result["run_description"],
            "score_keys": score_keys,
            "score_weights_for_different_modes": score_weights_for_different_modes,
            "n_samples": m3_result["n_samples"],
            "n_positive_chunks": m3_result["n_positive_chunks"],
            "n_negative_chunks": m3_result["n_negative_chunks"],
            "positive_chunks_per_sample": m3_result["positive_chunks_per_sample"],
            "negative_chunks_per_sample": m3_result["negative_chunks_per_sample"],
            "positive_to_negative_chunk_ratio": m3_result["positive_to_negative_chunk_ratio"],
            "results": results,
            "comparison": comparison_results,
        },
        result_file,
        ensure_ascii=False,
        indent=2,
    )
print(f"\nwrote M3 score comparison to {comparison_result_path.relative_to(project_root)}")
