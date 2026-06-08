import sys
from os.path import dirname, abspath

project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

from pathlib import Path
import json

import modal

from config.data import (
    KNOWLEDGE_BASE_PATH,
    TRAIN_THREADS_DATASET_PATH,
    EMAIL_KNOWLEDGE_BASE_PRE_CURATOR_STATISTICS_PLOT_PATH,
    EMAIL_KNOWLEDGE_BASE_POST_CURATOR_STATISTICS_PLOT_PATH,
    EMAIL_KNOWLEDGE_BASE_NO_UPM_AUTHOR_THREADS_PATH,
    EMAIL_KNOWLEDGE_BASE_NO_UPM_AUTHOR_THREAD_CHUNKS_PATH,
    EMAIL_KNOWLEDGE_BASE_NO_USEFUL_INFORMATION_THREAD_CHUNKS_PATH,
    EMAIL_KNOWLEDGE_BASE_REUSE_CURATION,
    EMAIL_KNOWLEDGE_BASE_REUSE_TIMESTAMP,
    UPM_DOMAINS,
)
from config.decoder import (
    EMAIL_KNOWLEDGE_BASE_CURATOR_PROFILE,
    MODEL_PROFILES,
)
from config.modal_apps import CURATOR_APP_NAME
from config.modal_functions import (
    RUN_EMAIL_KNOWLEDGE_BASE_CURATOR_PIPELINE_FUNCTION_NAME,
)
from helpers.curator import (
    build_email_knowledge_base_threads,
    save_email_knowledge_base_curator_plots,
    split_threads_by_upm_author,
)

THREADS_PATH = Path(project_root) / TRAIN_THREADS_DATASET_PATH
LOCAL_KNOWLEDGE_BASE_PATH = Path(project_root) / KNOWLEDGE_BASE_PATH
PRE_CURATOR_STATISTICS_PLOT_PATH = Path(
    project_root,
    EMAIL_KNOWLEDGE_BASE_PRE_CURATOR_STATISTICS_PLOT_PATH,
)
POST_CURATOR_STATISTICS_PLOT_PATH = Path(
    project_root,
    EMAIL_KNOWLEDGE_BASE_POST_CURATOR_STATISTICS_PLOT_PATH,
)
NO_UPM_AUTHOR_THREADS_PATH = Path(
    project_root,
    EMAIL_KNOWLEDGE_BASE_NO_UPM_AUTHOR_THREADS_PATH,
)
NO_UPM_AUTHOR_THREAD_CHUNKS_PATH = Path(
    project_root,
    EMAIL_KNOWLEDGE_BASE_NO_UPM_AUTHOR_THREAD_CHUNKS_PATH,
)
NO_USEFUL_INFORMATION_THREAD_CHUNKS_PATH = Path(
    project_root,
    EMAIL_KNOWLEDGE_BASE_NO_USEFUL_INFORMATION_THREAD_CHUNKS_PATH,
)

def main():
    def write_json(path, data):
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=2)

    # first, load the local threads unless the curator stage is being reused
    threads = None
    no_upm_author_threads = []
    if not EMAIL_KNOWLEDGE_BASE_REUSE_CURATION:
        threads = build_email_knowledge_base_threads(
            THREADS_PATH,
            LOCAL_KNOWLEDGE_BASE_PATH,
        )
        threads, no_upm_author_threads = split_threads_by_upm_author(
            threads,
            UPM_DOMAINS,
        )
        write_json(NO_UPM_AUTHOR_THREADS_PATH, no_upm_author_threads)

    # then run the remote curator pipeline and bring back summary and review data
    run_email_knowledge_base_curator_pipeline = modal.Function.from_name(
        CURATOR_APP_NAME,
        RUN_EMAIL_KNOWLEDGE_BASE_CURATOR_PIPELINE_FUNCTION_NAME,
    )
    reuse_timestamp = (
        str(EMAIL_KNOWLEDGE_BASE_REUSE_TIMESTAMP).strip()
        if EMAIL_KNOWLEDGE_BASE_REUSE_TIMESTAMP else None
    )
    result = run_email_knowledge_base_curator_pipeline.remote(
        threads=threads,
        reuse_curation=EMAIL_KNOWLEDGE_BASE_REUSE_CURATION,
        reuse_timestamp=reuse_timestamp,
    )

    # finally (and only if fresh curator run), produce decoder-stage plots
    if not EMAIL_KNOWLEDGE_BASE_REUSE_CURATION:
        result["curator_run_data"]["curator_statistics"]["n_no_upm_author_threads"] = (
            len(no_upm_author_threads)
        )
        write_json(
            NO_UPM_AUTHOR_THREAD_CHUNKS_PATH,
            result["curator_run_data"].get("no_upm_author_thread_chunks", []),
        )
        write_json(
            NO_USEFUL_INFORMATION_THREAD_CHUNKS_PATH,
            result["curator_run_data"].get("no_useful_information_thread_chunks", []),
        )
        max_input_tokens = MODEL_PROFILES[EMAIL_KNOWLEDGE_BASE_CURATOR_PROFILE].get(
            "max_input_tokens"
        )
        save_email_knowledge_base_curator_plots(
            result["curator_run_data"],
            max_input_tokens,
            PRE_CURATOR_STATISTICS_PLOT_PATH,
            POST_CURATOR_STATISTICS_PLOT_PATH,
        )

    print(
        "curate_email_knowledge_base: completed: "
        f"timestamp {result['encode_timestamp']}; "
        f"variants {list(result['variant_to_n_records'])}"
    )

if __name__ == "__main__":
    main()
