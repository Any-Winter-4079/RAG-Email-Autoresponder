import json
from pathlib import Path

from config.data import SPLIT_DATASETS_DIR
from config.eval import VALID_CONTEXT_EMAILS_MODES

#######################################
# Helper 1: Write eval output to file #
#######################################
def write_eval_output_to_file(data_variant_results_dir, output_name, eval_output, data_variant):
    json_path = data_variant_results_dir / f"{output_name}.json"
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(eval_output, json_file, ensure_ascii=False, indent=2)
    print(f"\twrote {data_variant}/{json_path.name}")
    print()

###################################################
# Helper 2: Get rerank text from collection point #
###################################################
def get_text_to_rerank_from_payload(payload):
    variant = payload["variant"]
    if variant == "lm_q_and_a_for_q_only_chunks":
        return payload["question"]
    if variant == "lm_q_and_a_chunks":
        return f"Q: {payload['question']}\nA: {payload['answer']}"
    return payload["text"]

######################################################
# Helper 3: Load selected split samples for eval run #
######################################################
def load_selected_split_samples(
        project_root,
        split_name,
        context_emails_mode,
        n_eval_samples_per_folder_uri,
        ):
    split_samples_path = Path(project_root) / SPLIT_DATASETS_DIR / f"{split_name}.json"
    with open(split_samples_path, "r", encoding="utf-8") as split_samples_file:
        all_split_samples = json.load(split_samples_file)

    if context_emails_mode not in VALID_CONTEXT_EMAILS_MODES:
        raise ValueError(
            "load_selected_split_samples: invalid context emails mode:\n"
            f"\t{context_emails_mode}\n"
            f"\tvalid modes: {sorted(VALID_CONTEXT_EMAILS_MODES)}"
        )

    if context_emails_mode == "without_context":
        all_split_samples = [
            sample
            for sample in all_split_samples
            if not sample["context_emails"]
        ]
    elif context_emails_mode == "with_context":
        all_split_samples = [
            sample
            for sample in all_split_samples
            if sample["context_emails"]
        ]

    all_folder_uris = sorted({sample["folder_uri"] for sample in all_split_samples})
    folder_uri_to_n_selected = {
        folder_uri: 0
        for folder_uri in all_folder_uris
    }
    split_samples = []
    for sample in all_split_samples:
        folder_uri = sample["folder_uri"]
        if folder_uri_to_n_selected[folder_uri] >= n_eval_samples_per_folder_uri:
            continue
        split_samples.append(sample)
        folder_uri_to_n_selected[folder_uri] += 1
        if all(
            n_selected >= n_eval_samples_per_folder_uri
            for n_selected in folder_uri_to_n_selected.values()
        ):
            break

    return split_samples, folder_uri_to_n_selected
