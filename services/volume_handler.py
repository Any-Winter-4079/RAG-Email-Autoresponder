from config.general import modal_secret, rag_volume, VOLUME_PATH
from config.modal_apps import VOLUME_HANDLER_APP_NAME
from config.encoder_cpu import (
    image,
    MODAL_TIMEOUT,
    SCALEDOWN_WINDOW,
    MIN_CONTAINERS
)
import modal

# modal run services/volume_handler.py::delete_volume_folders
# modal run services/volume_handler.py::count_lm_output_tokens

# Modal
app = modal.App(VOLUME_HANDLER_APP_NAME)

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def delete_volume_folders():
    import os
    import shutil
    from config.general import LEGACY_VOLUME_FOLDERS

    rag_volume.reload()

    volume_root = os.path.abspath(VOLUME_PATH)
    for folder_name in LEGACY_VOLUME_FOLDERS:
        if not folder_name:
            print("delete_volume_folders: empty folder name, skipping")
            continue

        folder_path = os.path.abspath(os.path.join(volume_root, folder_name))
        if os.path.commonpath([volume_root, folder_path]) != volume_root:
            raise ValueError(
                f"delete_volume_folders: refusing to delete path outside volume root: '{folder_name}'"
            )

        if not os.path.exists(folder_path):
            print(f"delete_volume_folders: '{folder_name}' does not exist, skipping")
            continue
        if not os.path.isdir(folder_path):
            print(f"delete_volume_folders: '{folder_name}' is not a directory, skipping")
            continue

        shutil.rmtree(folder_path)
        print(f"delete_volume_folders: deleted '{folder_name}'")

    rag_volume.commit()

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def count_lm_output_tokens():
    import os
    import json
    import glob
    from config.crawler_agent import (
        LM_CLEANED_TEXT_CHUNKS_PATH,
        LM_ABSTRACT_CHUNKS_PATH,
        LM_SUMMARY_CHUNKS_PATH,
        LM_Q_AND_A_CHUNKS_PATH,
    )

    rag_volume.reload()

    variant_to_path = {
        "lm_cleaned_text_chunks": LM_CLEANED_TEXT_CHUNKS_PATH,
        "lm_abstract_chunks": LM_ABSTRACT_CHUNKS_PATH,
        "lm_summary_chunks": LM_SUMMARY_CHUNKS_PATH,
        "lm_q_and_a_chunks": LM_Q_AND_A_CHUNKS_PATH,
    }

    grand_total_output_tokens = 0
    for variant, variant_path in variant_to_path.items():
        jsonl_paths = sorted(glob.glob(os.path.join(variant_path, "*.jsonl")))
        if not jsonl_paths:
            print(f"count_lm_output_tokens: {variant}\n\tno jsonl files found")
            continue

        latest_jsonl_path = jsonl_paths[-1]
        n_rows = 0
        total_output_tokens = 0

        with open(latest_jsonl_path, "r", encoding="utf-8") as jsonl_file:
            for line in jsonl_file:
                row = json.loads(line)
                n_rows += 1
                if "decoder_token_count" in row:
                    total_output_tokens += row["decoder_token_count"]
                elif "pairs" in row:
                    total_output_tokens += sum(
                        pair["decoder_token_count"]
                        for pair in row["pairs"]
                    )

        grand_total_output_tokens += total_output_tokens
        print(
            f"count_lm_output_tokens: {variant}\n"
            f"\tlatest file: {os.path.basename(latest_jsonl_path)}\n"
            f"\tn rows: {n_rows}\n"
            f"\toutput tokens: {total_output_tokens:,}"
        )

    print(
        f"count_lm_output_tokens: total tokens: {grand_total_output_tokens:,}"
    )
