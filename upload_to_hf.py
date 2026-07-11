from argparse import ArgumentParser
from pathlib import Path

DEFAULT_MODEL_DIR = Path(
    "models/sampled_negs-exp8-bs8-group-similar-effective16-keep_query_negatives_sample_rest_denom"
)
DEFAULT_REPO_ID = "Edue3r4t5y6/bge-m3-MUIA"

from huggingface_hub import HfApi


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    parser.add_argument("--repo-id", default=DEFAULT_REPO_ID)
    parser.add_argument(
        "--commit-message",
        default="Upload best BGE-M3 MUIA fine-tuned checkpoint",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"model directory does not exist: {model_dir}")
    if not model_dir.is_dir():
        raise NotADirectoryError(f"model path is not a directory: {model_dir}")

    api = HfApi()
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="model",
        exist_ok=True,
    )
    api.upload_folder(
        repo_id=args.repo_id,
        repo_type="model",
        folder_path=str(model_dir),
        commit_message=args.commit_message,
        ignore_patterns=[".DS_Store"],
    )

    print(f"upload_to_hf: uploaded {model_dir} to {args.repo_id}")


if __name__ == "__main__":
    main()
