import sys
from os.path import dirname, abspath

project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

from pathlib import Path

import modal

from config.data import (
    KNOWLEDGE_BASE_PATH,
    TRAIN_THREADS_DATASET_PATH,
    EMAIL_KNOWLEDGE_BASE_PRE_CURATOR_STATISTICS_PLOT_PATH,
    EMAIL_KNOWLEDGE_BASE_POST_CURATOR_STATISTICS_PLOT_PATH,
    EMAIL_KNOWLEDGE_BASE_REUSE_CURATION,
    EMAIL_KNOWLEDGE_BASE_REUSE_TIMESTAMP,
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

def main():
    # first, load the local threads unless the curator stage is being reused
    threads = None
    if not EMAIL_KNOWLEDGE_BASE_REUSE_CURATION:
        threads = build_email_knowledge_base_threads(
            THREADS_PATH,
            LOCAL_KNOWLEDGE_BASE_PATH,
        )

    # then run the remote curator pipeline and only bring back the small summary data
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
