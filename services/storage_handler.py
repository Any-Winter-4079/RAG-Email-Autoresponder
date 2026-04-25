from config.general import modal_secret, rag_volume, VOLUME_PATH
from config.modal_apps import STORAGE_HANDLER_APP_NAME
from config.encoder_cpu import (
    image,
    MODAL_TIMEOUT,
    SCALEDOWN_WINDOW,
    MIN_CONTAINERS,
)
import modal

app = modal.App(STORAGE_HANDLER_APP_NAME)

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def write_chunk_records(records, json_path, txt_path, label, decoder_path, encoder_path):
    import json
    import os

    rag_volume.reload()

    volume_root = os.path.abspath(VOLUME_PATH)
    for path in [json_path, txt_path]:
        parent_dir = os.path.abspath(os.path.dirname(path))
        if os.path.commonpath([volume_root, parent_dir]) != volume_root:
            raise ValueError(
                "write_chunk_records: refusing to write outside volume root:\n"
                f"\t{path}"
            )
        os.makedirs(parent_dir, exist_ok=True)

    with open(json_path, "w", encoding="utf-8") as f_json, open(txt_path, "w", encoding="utf-8") as f_txt:
        for i, chunk in enumerate(records):
            separator = "=" * 150
            f_json.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            if "text" in chunk:
                content = chunk["text"]
                token_info = f"Tokens {decoder_path}: {chunk['decoder_token_count']:,} | Tokens {encoder_path}: {chunk['encoder_token_count']:,}"
            elif "pairs" in chunk:
                content = "\n".join([f"Q: {pair['question']}\nA: {pair['answer']}" for pair in chunk["pairs"]])
                if "FOR Q ONLY" in label.upper():
                    max_decoder_tokens = max((pair["decoder_token_count_q"] for pair in chunk["pairs"]))
                    max_encoder_tokens = max((pair["encoder_token_count_q"] for pair in chunk["pairs"]))
                    token_info = (
                        f"Pairs: {len(chunk['pairs'])} | "
                        f"Tokens (max) {decoder_path}: {max_decoder_tokens:,} | "
                        f"Tokens (max) {encoder_path}: {max_encoder_tokens:,}"
                    )
                else:
                    max_decoder_tokens = max((
                        pair["decoder_token_count_q"] + pair["decoder_token_count_a"]
                        for pair in chunk["pairs"]
                    ))
                    max_encoder_tokens = max((
                        pair["encoder_token_count_q"] + pair["encoder_token_count_a"]
                        for pair in chunk["pairs"]
                    ))
                    token_info = (
                        f"Pairs: {len(chunk['pairs'])} | "
                        f"Tokens (max) {decoder_path}: {max_decoder_tokens:,} | "
                        f"Tokens (max) {encoder_path}: {max_encoder_tokens:,}"
                    )
            else:
                content = ""
                token_info = ""

            if "subchunk_index" in chunk:
                chunk_label = f"{chunk['chunk_index']}.{chunk['subchunk_index']}"
            else:
                chunk_label = str(i + 1)
            if "url" in chunk:
                header = f"{label} CHUNK {chunk_label} [Source: {chunk['url']}] | {token_info}"
            else:
                header = f"{label} CHUNK {chunk_label} [Thread ID: {chunk['thread_id']}] | {token_info}"
            f_txt.write(f"\n{separator}\n{header}\n{separator}\n{content}\n")

    rag_volume.commit()
    print(
        "write_chunk_records: wrote records:\n"
        f"\tlabel: {label}\n"
        f"\tjson_path: {json_path}\n"
        f"\ttxt_path: {txt_path}\n"
        f"\tn_records: {len(records)}"
    )

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
    volumes={VOLUME_PATH: rag_volume},
)
def read_jsonl_records(variant_path, file_start, timestamp=None):
    import glob
    import json
    import os

    rag_volume.reload()

    volume_root = os.path.abspath(VOLUME_PATH)
    variant_path = os.path.abspath(variant_path)
    if os.path.commonpath([volume_root, variant_path]) != volume_root:
        raise ValueError(
            "read_jsonl_records: refusing to read outside volume root:\n"
            f"\t{variant_path}"
        )

    if timestamp is not None:
        jsonl_path = os.path.join(variant_path, f"{file_start}{timestamp}.jsonl")
        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(
                "read_jsonl_records: jsonl path not found:\n"
                f"\t{jsonl_path}"
            )
    else:
        jsonl_paths = glob.glob(os.path.join(variant_path, f"{file_start}*.jsonl"))
        if not jsonl_paths:
            raise FileNotFoundError(
                "read_jsonl_records: no jsonl files found:\n"
                f"\t{variant_path}"
            )
        jsonl_path = max(jsonl_paths, key=os.path.getctime)

    basename = os.path.basename(jsonl_path)
    resolved_timestamp = basename[len(file_start):-len(".jsonl")]
    with open(jsonl_path, "r", encoding="utf-8") as jsonl_file:
        records = [
            json.loads(line)
            for line in jsonl_file
            if line.strip()
        ]

    print(
        "read_jsonl_records: read records:\n"
        f"\tjsonl_path: {jsonl_path}\n"
        f"\tresolved_timestamp: {resolved_timestamp}\n"
        f"\tn_records: {len(records)}"
    )
    return {
        "jsonl_path": jsonl_path,
        "timestamp": resolved_timestamp,
        "records": records,
    }
