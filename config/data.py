DATASET_PATH = "data/email_messages/messages.csv"
KNOWLEDGE_BASE_MESSAGES_PATH = "data/knowledge_base_messages/messages_for_knowledge_base.jsonl"
THREAD_GROUPING_STRATEGY = "decoder_based"
VALID_THREAD_GROUPING_STRATEGIES = {
    "rule_based",
    "decoder_based",
}
MESSAGES_WITH_THREADS_DATASET_PATH = (
    f"data/email_messages/{THREAD_GROUPING_STRATEGY}/messages_with_threads.csv"
)
MESSAGES_WITH_THREADS_JSON_DATASET_PATH = (
    f"data/email_messages/{THREAD_GROUPING_STRATEGY}/messages_with_threads.json"
)
WEAK_THREADS_BY_SIZE_DATASET_PATH = (
    f"data/email_messages/{THREAD_GROUPING_STRATEGY}/weak_threads_by_size.json"
)
LM_THREADS_BY_SIZE_DATASET_PATH = (
    f"data/email_messages/{THREAD_GROUPING_STRATEGY}/lm_threads_by_size.json"
)
SPLIT_DATASETS_DIR = f"data/split_datasets/{THREAD_GROUPING_STRATEGY}"
TRAIN_THREADS_DATASET_PATH = f"{SPLIT_DATASETS_DIR}/train_threads.json"
DEV_THREADS_DATASET_PATH = f"{SPLIT_DATASETS_DIR}/dev_threads.json"
TEST_THREADS_DATASET_PATH = f"{SPLIT_DATASETS_DIR}/test_threads.json"
DISCARDED_THREADS_DIR = f"data/discarded_threads/{THREAD_GROUPING_STRATEGY}"
EMAIL_THREAD_CANDIDATES_PATH = "data/knowledge_base_messages/email_thread_candidates.json"
EMAIL_LM_ABSTRACT_CHUNKS_PATH = "data/knowledge_base_messages/email_lm_abstract_chunks.jsonl"
EMAIL_LM_SUMMARY_CHUNKS_PATH = "data/knowledge_base_messages/email_lm_summary_chunks.jsonl"
EMAIL_LM_CLEANED_TEXT_CHUNKS_PATH = "data/knowledge_base_messages/email_lm_cleaned_text_chunks.jsonl"
EMAIL_LM_Q_AND_A_CHUNKS_PATH = "data/knowledge_base_messages/email_lm_q_and_a_chunks.jsonl"
EMAIL_KNOWLEDGE_BASE_VARIANT_TO_PATH = {
    "email_lm_summary_chunks": EMAIL_LM_SUMMARY_CHUNKS_PATH,
    "email_lm_cleaned_text_chunks": EMAIL_LM_CLEANED_TEXT_CHUNKS_PATH,
    "email_lm_q_and_a_chunks": EMAIL_LM_Q_AND_A_CHUNKS_PATH,
}
EMAIL_KNOWLEDGE_BASE_VARIANT_TO_BASE_VARIANT = {
    "email_lm_summary_chunks": "lm_summary_chunks",
    "email_lm_cleaned_text_chunks": "lm_cleaned_text_chunks",
    "email_lm_q_and_a_chunks": "lm_q_and_a_chunks",
}
EMAIL_KNOWLEDGE_BASE_RECREATE_COLLECTIONS = True
EMAIL_THREAD_MAX_THREADS_TO_CURATE = None
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
