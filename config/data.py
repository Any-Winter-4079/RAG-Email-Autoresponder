DATASET_PATH = "data/email_messages/messages.csv"
KNOWLEDGE_BASE_PATH = "data/knowledge_base/knowledge_base.json"
THREAD_GROUPING_STRATEGY = "decoder_based"
VALID_THREAD_GROUPING_STRATEGIES = {
    "rule_based",
    "decoder_based",
}
THREADS_PATH = f"data/threads/{THREAD_GROUPING_STRATEGY}/threads.json"
DISCARDED_THREADS_PATH = f"data/discarded_threads/{THREAD_GROUPING_STRATEGY}/discarded_internal_threads.json"

SPLIT_DATASETS_DIR = f"data/split_datasets/{THREAD_GROUPING_STRATEGY}"
TRAIN_THREADS_DATASET_PATH = f"{SPLIT_DATASETS_DIR}/train_threads.json"
DEV_THREADS_DATASET_PATH = f"{SPLIT_DATASETS_DIR}/dev_threads.json"
TEST_THREADS_DATASET_PATH = f"{SPLIT_DATASETS_DIR}/test_threads.json"

THREAD_GROUPER_MAX_EMAILS = 16
THREAD_GROUPER_MAX_RULE_BASED_THREADS = 8
THREAD_GROUPER_PRE_DECODER_STATISTICS_PLOT_PATH = (
    f"data/threads/{THREAD_GROUPING_STRATEGY}/thread_grouping_pre_decoder_statistics.png"
)
THREAD_GROUPER_POST_DECODER_STATISTICS_PLOT_PATH = (
    f"data/threads/{THREAD_GROUPING_STRATEGY}/thread_grouping_post_decoder_statistics.png"
)

EMAIL_KNOWLEDGE_BASE_MAX_EMAILS = 12
EMAIL_KNOWLEDGE_BASE_MAX_THREADS = 6
EMAIL_KNOWLEDGE_BASE_PRE_CURATOR_STATISTICS_PLOT_PATH = (
    "data/knowledge_base/email_knowledge_base_pre_curator_statistics.png"
)
EMAIL_KNOWLEDGE_BASE_POST_CURATOR_STATISTICS_PLOT_PATH = (
    "data/knowledge_base/email_knowledge_base_post_curator_statistics.png"
)
EMAIL_KNOWLEDGE_BASE_FILE_START = "email_knowledge_base_"
EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX = "email_"
EMAIL_KNOWLEDGE_BASE_REUSE_CURATION = True
EMAIL_KNOWLEDGE_BASE_REUSE_TIMESTAMP = None
EMAIL_KNOWLEDGE_BASE_RECREATE_COLLECTIONS = True

SHUFFLE_SEED = 42
UPM_DOMAINS = [
    "upm.es",
    "fi.upm.es",
    "etsii.upm.es",
    "relay.fi.upm.es"
]
AUTOMATED_OUTBOUND_TEMPLATES = [
    "Nos es grato comunicarte que has sido admitido",
    "Nos es grato comunicarte que has sido admitida",
    "le comunicamos que le hemos dado un acceso condicionado a la finalización de sus estudios",
    "Como sabrás, tu admisión está condicionada a la realización de",
    "el número de alumnos que se preinscriben es mucho mayor que el número de plazas disponibles",
    "En primer lugar, deseo expresarte mi pesar por haber tenido que denegar"
]
PRE_ENROLLMENT_TEMPLATES = [
    "El documento que recibe a continuación es la solicitud de preinscripción al Master/Doctorado que usted coordina/gestiona."
]
REMOVE_INTERNAL_UPM_MESSAGES = True
TRAIN_SPLIT_PCT = 0.7
DEV_SPLIT_PCT = 0.15
# TEST_SPLIT_PCT = 1 - (TRAIN_SPLIT_PCT + DEV_SPLIT_PCT)

M3_FINETUNE_QUERY_TYPES = [
    "reranker",
]
M3_FINETUNE_DATA_VARIANTS = [
    "lm_summary_chunks",
]
M3_FINETUNE_ORACLE_DISCRIMINATOR_SPLIT = "train"
M3_FINETUNE_VARIANT_TO_ORACLE_DISCRIMINATOR_TIMESTAMP = {
    "lm_summary_chunks": None,
}
M3_FINETUNE_VARIANT_TO_RRF_TIMESTAMP = {
    "lm_summary_chunks": None,
}
