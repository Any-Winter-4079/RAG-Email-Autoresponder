import json
from argparse import ArgumentParser
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
results_dir = project_root / "finetune" / "oracle_score_results"

parser = ArgumentParser()
parser.add_argument("timestamps", nargs="*")
args = parser.parse_args()

if args.timestamps:
    result_paths = [
        Path(timestamp) if timestamp.endswith(".json") else results_dir / f"{timestamp}.json"
        for timestamp in args.timestamps
    ]
else:
    result_paths = sorted(results_dir.glob("*.json"))

def get_score_label(score_key, weights):
    if score_key == "sparse+dense" and weights == [2 / 3, 1 / 3, 0.0]:
        return "dense/sparse (2:1)"
    if score_key == "sparse+dense" and weights == [1.0, 1.0, 0.0]:
        return "dense/sparse (1:1)"
    return score_key

def short_description(description):
    max_length = 40
    if len(description) <= max_length:
        return description
    return f"{description[:max_length - 3]}..."

rows = []
for result_path in result_paths:
    with open(result_path, "r", encoding="utf-8") as result_file:
        result = json.load(result_file)
    base_encoder_name, tuned_encoder_name = result["encoder_names"]
    weights = result["score_weights_for_different_modes"]
    for score_key in result["score_keys"]:
        base_metrics = result["results"][base_encoder_name][score_key]
        tuned_metrics = result["results"][tuned_encoder_name][score_key]
        comparison_metrics = result["comparison"][score_key]
        rows.append([
            result["timestamp"],
            short_description(result["run_description"]),
            get_score_label(score_key, weights),
            base_metrics["pair_accuracy"],
            tuned_metrics["pair_accuracy"],
            comparison_metrics["pair_accuracy"],
            base_metrics["mrr"],
            tuned_metrics["mrr"],
            comparison_metrics["mrr"],
        ])

headers = [
    "timestamp",
    "description",
    "score",
    "base acc",
    "sft acc",
    "diff acc",
    "base mrr",
    "sft mrr",
    "diff mrr",
]
widths = [19, 40, 18, 8, 8, 8, 8, 8, 8]
print("  ".join(header.ljust(width) for header, width in zip(headers, widths)))
print("  ".join("-" * width for width in widths))
for row in rows:
    print(
        f"{row[0].ljust(widths[0])}  "
        f"{row[1].ljust(widths[1])}  "
        f"{row[2].ljust(widths[2])}  "
        f"{row[3]:>{widths[3]}.4f}  "
        f"{row[4]:>{widths[4]}.4f}  "
        f"{row[5]:>{widths[5]}.4f}  "
        f"{row[6]:>{widths[6]}.4f}  "
        f"{row[7]:>{widths[7]}.4f}  "
        f"{row[8]:>{widths[8]}.4f}"
    )
