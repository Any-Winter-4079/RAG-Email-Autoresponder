from config.crawler_agent import ALLOWED_URL_HOST_TO_CATEGORY
from config.modal_apps import DECODER_LEGACY_APP_NAME
from config.modal_functions import RUN_LOCAL_LM_OR_VLM_LEGACY_FUNCTION_NAME
from config.modal_apps import DECODER_LATEST_APP_NAME
from config.modal_functions import RUN_LOCAL_LM_OR_VLM_LATEST_FUNCTION_NAME
from config.modal_functions import RUN_LOCAL_LM_OR_VLM_LATEST_FUNCTION_NAME

COMMON_PACKAGES = [
    "torchvision",
    "accelerate",
    "Pillow",
    "requests",
    "hf_transfer",
]
SCALEDOWN_WINDOW = 60 # seconds
TIMEOUT = 900 # seconds
MIN_CONTAINERS = 0 # 0 to make sure we don't pay 24/7
FLASH_ATTENTION_RELEASE = "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/flash_attn-2.8.3+cu12torch2.8cxx11abiFALSE-cp311-cp311-linux_x86_64.whl"
FLASH_ATTENTION_IMAGE = "anywinter4079/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04-runpod-clone"
FLASH_ATTENTION_RUN_COMMANDS = ("python -m pip install --upgrade pip && "
                                "pip config set global.extra-index-url https://download.pytorch.org/whl/cu128")
FLASH_ATTENTION_TORCH_VERSION = "torch==2.8.0+cu128"
USE_FLASH_ATTENTION_IMAGE = False # TODO: fix flash_attn image (should be faster than image without flash_attn)
NO_FLASH_ATTENTION_PYTHON_VERSION = "3.11"
NO_FLASH_ATTENTION_TORCH_VERSION = "torch==2.8.0"

NO_MESSAGE_OPENING_TAG = "<nomessage>"
NO_MESSAGE_CLOSING_TAG = "</nomessage>"
MESSAGE_OPENING_TAG = "<message>"
MESSAGE_CLOSING_TAG = "</message>"

ABSTRACT_OPENING_TAG = "<abstract>"
ABSTRACT_CLOSING_TAG = "</abstract>"

SUMMARY_OPENING_TAG = "<summary>"
SUMMARY_CLOSING_TAG = "</summary>"

CLEANED_TEXT_OPENING_TAG = "<cleanedtext>"
CLEANED_TEXT_CLOSING_TAG = "</cleanedtext>"
TRANSLATION_OPENING_TAG = "<translation>"
TRANSLATION_CLOSING_TAG = "</translation>"
QUERIES_OPENING_TAG = "<queries>"
QUERIES_CLOSING_TAG = "</queries>"
QUERY_OPENING_TAG = "<query>"
QUERY_CLOSING_TAG = "</query>"
KEYWORD_QUERIES_OPENING_TAG = "<keywordqueries>"
KEYWORD_QUERIES_CLOSING_TAG = "</keywordqueries>"
NATURAL_QUERIES_OPENING_TAG = "<naturalqueries>"
NATURAL_QUERIES_CLOSING_TAG = "</naturalqueries>"
HYDE_QUERIES_OPENING_TAG = "<hydequeries>"
HYDE_QUERIES_CLOSING_TAG = "</hydequeries>"
QUESTION_QUERIES_OPENING_TAG = "<questionqueries>"
QUESTION_QUERIES_CLOSING_TAG = "</questionqueries>"

NO_REQUEST_OPENING_TAG = "<norequest>"
NO_REQUEST_CLOSING_TAG = "</norequest>"
NO_USEFUL_INFORMATION_OPENING_TAG = "<nousefulinformation>"
NO_USEFUL_INFORMATION_CLOSING_TAG = "</nousefulinformation>"
RERANKER_QUERY_OPENING_TAG = "<rerankerquery>"
RERANKER_QUERY_CLOSING_TAG = "</rerankerquery>"
QUERY_REWRITER_SECTION_TO_MAX_QUERIES = {
    "keyword": 8,
    "natural": 8,
    "hyde": 8,
    "question": 8,
}
QUERY_REWRITER_TEMPERATURE = 0.1
QUERY_REWRITER_TEMPERATURE_TAG = str(QUERY_REWRITER_TEMPERATURE).replace(".", "p")
QUERY_REWRITER_CACHE_TAG = (
    f"kw{QUERY_REWRITER_SECTION_TO_MAX_QUERIES['keyword']}"
    f"_nat{QUERY_REWRITER_SECTION_TO_MAX_QUERIES['natural']}"
    f"_hyde{QUERY_REWRITER_SECTION_TO_MAX_QUERIES['hyde']}"
    f"_q{QUERY_REWRITER_SECTION_TO_MAX_QUERIES['question']}"
    f"_temp{QUERY_REWRITER_TEMPERATURE_TAG}"
)
OLD_MUIA_SUBJECT_AND_SEMINAR_CODE_MAPPING_TEXT = (
    "S1: Metodología de la investigación | Research Methodology\n"
    "S2: Gestión de proyectos y análisis de riesgos | Project Management and Risk Analysis\n"
    "S3: Aspectos éticos y legales de la Inteligencia Artificial | Ethical and Legal Aspects of Artificial Intelligence\n"
    "S4: Inteligencia Artificial e inclusión | Artificial Intelligence and Inclusion\n"
    "A1: Sistemas de ayuda a la decisión | Decision Support Systems\n"
    "A2: Decisión participativa y negociación | Participatory Decision-Making and Negotiation\n"
    "A3: Métodos de simulación | Simulation Methods\n"
    "S5: Análisis de decisiones | Decision Analysis\n"
    "A4: Redes bayesianas | Bayesian Networks\n"
    "A5: Aprendizaje automático | Machine Learning\n"
    "A6: Redes de neuronas artificiales y Deep Learning | Artificial Neural Networks and Deep Learning\n"
    "A7: Inteligencia artificial explicable | Explainable Artificial Intelligence\n"
    "S6: Aprendizaje automático | Machine Learning\n"
    "A8: Búsqueda inteligente basada en metaheurísticas | Intelligent Search Based on Metaheuristics\n"
    "A9: Computación evolutiva | Evolutionary Computation\n"
    "A10: Biología programable: Computación con ADN e Ingeniería de biocircuitos | Programmable Biology: DNA Computing and Biocircuit Engineering\n"
    "S7: Computación natural | Natural Computing\n"
    "A11: Programación lógica | Logic Programming\n"
    "A12: Sistemas multiagente | Multi-Agent Systems\n"
    "A13: Ingeniería ontológica | Ontological Engineering\n"
    "A14: Modelos de razonamiento | Reasoning Models\n"
    "S8: Representación del conocimiento y razonamiento | Knowledge Representation and Reasoning\n"
    "S9: Lógica borrosa | Fuzzy Logic\n"
    "S10: Computación cognitiva | Cognitive Computing\n"
    "A15: Visión por computador | Computer Vision\n"
    "A16: Robots autónomos | Autonomous Robots\n"
    "S11: Robótica cognitiva y percepción | Cognitive Robotics and Perception\n"
    "S12: Principios de la locomoción robótica | Principles of Robotic Locomotion\n"
    "A17: Informática biomédica | Biomedical Informatics\n"
    "A18: Ingeniería lingüística | Linguistic Engineering\n"
    "A19: Ciencia de la web | Web Science\n"
    "A20: Deep Learning para el Procesamiento del Lenguaje Natural | Deep Learning for Natural Language Processing\n"
    "S13: Aplicaciones de la Inteligencia Artificial | Applications of Artificial Intelligence\n"
    "S14: Procesamiento del lenguaje natural | Natural Language Processing\n"
    "S15: Planificación automática | Automated Planning\n"
    "S16: IA Generativa y Prompt Engineering: Aplicaciones y Retos | Generative AI and Prompt Engineering: Applications and Challenges"
)

QUESTIONS_OPENING_TAG = "<questions>"
QUESTIONS_CLOSING_TAG = "</questions>"
QUESTION_OPENING_TAG = "<question>"
QUESTION_CLOSING_TAG = "</question>"
ANSWER_OPENING_TAG = "<answer>"
ANSWER_CLOSING_TAG = "</answer>"
SCORES_OPENING_TAG = "<scores>"
SCORES_CLOSING_TAG = "</scores>"
SCORE_OPENING_TAG = "<score>"
SCORE_CLOSING_TAG = "</score>"
ANSWERABILITY_OPENING_TAG = "<answerability>"
ANSWERABILITY_CLOSING_TAG = "</answerability>"
SUBQUERIES_OPENING_TAG = "<subqueries>"
SUBQUERIES_CLOSING_TAG = "</subqueries>"
SUBQUERY_OPENING_TAG = "<subquery>"
SUBQUERY_CLOSING_TAG = "</subquery>"
SUBQUERY_TEXT_OPENING_TAG = "<subquerytext>"
SUBQUERY_TEXT_CLOSING_TAG = "</subquerytext>"
SUBQUERY_ANSWERABILITY_OPENING_TAG = "<subqueryanswerability>"
SUBQUERY_ANSWERABILITY_CLOSING_TAG = "</subqueryanswerability>"
SUBQUERY_CONFIDENCE_OPENING_TAG = "<subqueryconfidence>"
SUBQUERY_CONFIDENCE_CLOSING_TAG = "</subqueryconfidence>"
SUBQUERY_SUPPORTING_CHUNK_IDS_OPENING_TAG = "<subquerysupportingchunkids>"
SUBQUERY_SUPPORTING_CHUNK_IDS_CLOSING_TAG = "</subquerysupportingchunkids>"
SUBQUERY_INSUFFICIENT_CHUNK_IDS_OPENING_TAG = "<subqueryinsufficientchunkids>"
SUBQUERY_INSUFFICIENT_CHUNK_IDS_CLOSING_TAG = "</subqueryinsufficientchunkids>"
SUBQUERY_RATIONALE_OPENING_TAG = "<subqueryrationale>"
SUBQUERY_RATIONALE_CLOSING_TAG = "</subqueryrationale>"
CHUNK_ID_OPENING_TAG = "<chunkid>"
CHUNK_ID_CLOSING_TAG = "</chunkid>"
DRAFT_ANSWER_OPENING_TAG = "<draftanswer>"
DRAFT_ANSWER_CLOSING_TAG = "</draftanswer>"
RATIONALE_OPENING_TAG = "<rationale>"
RATIONALE_CLOSING_TAG = "</rationale>"
Q_AND_A_MAX_PAIRS = 20
NON_MUIA_TITULOS_PROPIOS = [
    "Máster en Fundamentos y Aplicaciones de la Inteligencia Artificial",
    "Máster en Gestión del Aseguramiento, Protección y Defensa del Software, Operaciones y Sistemas",
    "Máster en Agile Coaching y Transformación Agile",
    "Máster en dirección y gestión de proyectos software (a distancia)",
    "Máster en Consultoría en Gestión de Empresas",
]
NON_MUIA_OFFICIAL_MASTERS = [
    "Máster Universitario en Ingeniería Informática",
    "Máster Universitario en Innovación Digital",
    "Máster Universitario en Ingeniería del Software - European Master in Software Engineering",
    "Máster Universitario en Ciencia de Datos",
    "Máster Universitario en Software y Sistemas",
    "Máster Interuniversitario en Métodos Formales en Ingeniería Informática (UPM, UCM, UAM)",
]
SOURCE_HOST_CATEGORY_TEXT = ", ".join(
    [f"{host}: {category}" for host, category in ALLOWED_URL_HOST_TO_CATEGORY.items()]
)
SOURCE_HOST_CATEGORY_COUNT = len(ALLOWED_URL_HOST_TO_CATEGORY)
SOURCE_CATEGORY_TEXT = ", ".join(ALLOWED_URL_HOST_TO_CATEGORY.values())

THREAD_OPENING_TAG = "<thread>"
THREAD_CLOSING_TAG = "</thread>"
THREAD_MESSAGE_OPENING_TAG = "<message>"
THREAD_MESSAGE_CLOSING_TAG = "</message>"
THREAD_FROM_OPENING_TAG = "<from>"
THREAD_FROM_CLOSING_TAG = "</from>"
THREAD_TO_OPENING_TAG = "<to>"
THREAD_TO_CLOSING_TAG = "</to>"
THREAD_SUBJECT_OPENING_TAG = "<subject>"
THREAD_SUBJECT_CLOSING_TAG = "</subject>"
THREAD_BODY_OPENING_TAG = "<body>"
THREAD_BODY_CLOSING_TAG = "</body>"

THREAD_GROUPER_MAX_EMAILS = 24
EMAIL_WRITER_PROFILE = "email_writer"
THREAD_GROUPER_PROFILE = "thread_grouper"
DATA_CLEANER_PROFILE = "data_cleaner"
EMAIL_KNOWLEDGE_BASE_CURATOR_PROFILE = "email_knowledge_base_curator"
QUERY_REWRITER_PROFILE = "query_rewriter"
LLM_JUDGE_PROFILE = "llm_judge"

QWEN3_MODEL_FAMILY = "qwen3"
GEMMA4_MODEL_FAMILY = "gemma4"

QWEN3_8B_FP8_MODEL_NAME_OR_PATH = "Qwen/Qwen3-8B-FP8"
QWEN3_14B_FP8_MODEL_NAME_OR_PATH = "Qwen/Qwen3-14B-FP8"
GEMMA4_26B_A4B_IT_MODEL_NAME_OR_PATH = "google/gemma-4-26B-A4B-it"
QWEN3_DEFAULT_SAMPLING = {
    "temperature": 0.1,
    "top_p": 0.8,
    "top_k": 20,
}
GEMMA4_DEFAULT_SAMPLING = {
    "temperature": 1.0,
    "top_p": 0.95,
    "top_k": 64,
}
QWEN3_8B_FP8_MODEL_SETTINGS = {
    "model_family": QWEN3_MODEL_FAMILY,
    "model_name_or_path": QWEN3_8B_FP8_MODEL_NAME_OR_PATH,
    **QWEN3_DEFAULT_SAMPLING,
}
QWEN3_14B_FP8_MODEL_SETTINGS = {
    "model_family": QWEN3_MODEL_FAMILY,
    "model_name_or_path": QWEN3_14B_FP8_MODEL_NAME_OR_PATH,
    **QWEN3_DEFAULT_SAMPLING,
}
GEMMA4_26B_A4B_IT_MODEL_SETTINGS = {
    "model_family": GEMMA4_MODEL_FAMILY,
    "model_name_or_path": GEMMA4_26B_A4B_IT_MODEL_NAME_OR_PATH,
    **GEMMA4_DEFAULT_SAMPLING,
}
EMAIL_WRITER_SETTINGS = {
    "decoder_app_name": DECODER_LEGACY_APP_NAME,
    "decoder_function_name": RUN_LOCAL_LM_OR_VLM_LEGACY_FUNCTION_NAME,
    "provider": "local",
    **QWEN3_8B_FP8_MODEL_SETTINGS,
    "enable_thinking": True,
    "is_vision_model": False,
    "max_context_tokens": 32_768,
    "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
    "return_prompt_text": True,
}
THREAD_GROUPER_SETTINGS = {
    "provider": "local",
    **GEMMA4_26B_A4B_IT_MODEL_SETTINGS,
    "decoder_app_name": DECODER_LATEST_APP_NAME,
    "decoder_function_name": RUN_LOCAL_LM_OR_VLM_LATEST_FUNCTION_NAME,
    "enable_thinking": False,
    "is_vision_model": False,
    # Approximate Gemma 4 26B A4B IT memory budget in bf16:
    # - full model weights: about 26B parameters * 2 bytes ~= 52 GB (~48.4 GiB)
    # - H100 SXM memory: 81920 MiB = 80 GiB
    # - approximate KV cache per token:
    # 25 (sliding layers) * 8 (sliding KV heads) * 256 (head dim) * 2 (K and V) * 2 (bytes) = 204_800 bytes / token
    # + 5 (global layers) * 2 (global KV heads) * 512 (global head dim) * 2 (K and V) * 2 (bytes) = 20_480 bytes / token
    # = 225_280 bytes / token, so total KV cache is approximately sequence_length * 225_280 bytes for batch size 1.
    # - rough shorthand: 80 - 48.4 ~= 31.6 GiB left for KV cache and runtime overhead
    # - safer KV+overhead budget: about 20-24 GiB
    # - approximate KV-only sequence-length budget:
    #   20 GiB / 225_280 ~= 95_325 tokens
    #   24 GiB / 225_280 ~= 114_390 tokens
    "max_input_tokens": 32_768,
    "max_new_tokens": 32_768,
    "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
    "return_prompt_text": True,
}
QUERY_REWRITER_SETTINGS = {
    "decoder_app_name": DECODER_LATEST_APP_NAME,
    "decoder_function_name": RUN_LOCAL_LM_OR_VLM_LATEST_FUNCTION_NAME,
    "provider": "local",
    **GEMMA4_26B_A4B_IT_MODEL_SETTINGS,
    "enable_thinking": True,
    "is_vision_model": False,
    "max_new_tokens": 8_192,
    "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
    "return_prompt_text": False,
}

DATA_CLEANER_PROVIDER = "local"
DATA_CLEANER_PROVIDER_TO_SETTINGS = {
    "local": {
        "decoder_app_name": DECODER_LEGACY_APP_NAME,
        "decoder_function_name": RUN_LOCAL_LM_OR_VLM_LEGACY_FUNCTION_NAME,
        "provider": "local",
        **QWEN3_8B_FP8_MODEL_SETTINGS,
        "enable_thinking": True,
        "is_vision_model": False,
        "page_history_first_n": 5,
        "page_history_last_n": 15,
        "max_chunk_size": 1024,
        "max_new_tokens": 8_192,
        "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
        "return_prompt_text": True,
    },
    "openai": {
        "provider": "openai",
        "model_name_or_path": "gpt-5-nano",
        "enable_thinking": True,
        "reasoning_effort": "high",
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 20,
        "is_vision_model": False,
        "page_history_first_n": 2,
        "page_history_last_n": 4,
        "max_chunk_size": 2048,
        "max_new_tokens": 8_192,
        "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
        "return_prompt_text": True,
    },
}
DATA_CLEANER_SETTINGS = DATA_CLEANER_PROVIDER_TO_SETTINGS[DATA_CLEANER_PROVIDER]

EMAIL_KNOWLEDGE_BASE_CURATOR_SETTINGS = {
    "decoder_app_name": DECODER_LATEST_APP_NAME,
    "decoder_function_name": RUN_LOCAL_LM_OR_VLM_LATEST_FUNCTION_NAME,
    "provider": "local",
    **GEMMA4_26B_A4B_IT_MODEL_SETTINGS,
    "enable_thinking": True,
    "is_vision_model": False,
    "max_new_tokens": 8_192,
    "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
    "return_prompt_text": True,
}

LLM_JUDGE_PROVIDER = "openai"
LLM_JUDGE_PROVIDER_TO_SETTINGS = {
    "local": {
        "decoder_app_name": DECODER_LEGACY_APP_NAME,
        "decoder_function_name": RUN_LOCAL_LM_OR_VLM_LEGACY_FUNCTION_NAME,
        "provider": "local",
        **QWEN3_8B_FP8_MODEL_SETTINGS,
        "enable_thinking": True,
        "is_vision_model": False,
        "max_new_tokens": 4_096,
        "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
        "return_prompt_text": False,
    },
    "openai": {
        "provider": "openai",
        "model_name_or_path": "gpt-5.4-mini",
        "enable_thinking": True,
        "reasoning_effort": "medium",
        "is_vision_model": False,
        "max_new_tokens": 16_384,
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 20,
        "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
        "return_prompt_text": False,
    },
}
LLM_JUDGE_SETTINGS = LLM_JUDGE_PROVIDER_TO_SETTINGS[LLM_JUDGE_PROVIDER]
LLM_JUDGE_MAX_SUPPORTING_CHUNK_IDS = 16
LLM_JUDGE_MAX_INSUFFICIENT_CHUNK_IDS = 16

DIRECTOR_EMAIL = "masteria.dia@fi.upm.es"
DIRECTOR_NAME = "Damiano Zanardini"
DEPARTMENT_PHONE = "+34 910672909"

# anonymized
EXAMPLE_STUDENT_NAME = "Marco Conti"
EXAMPLE_STUDENT_EMAIL = "marco.conti@uxg.edu"
EXAMPLE_STAFF_NAME = "Laura Medina"
EXAMPLE_STAFF_EMAIL = "laura.medina@fi.upm.es"
EXAMPLE_COLLEAGUE_NAME = "Alex Perez"
EXAMPLE_COLLEAGUE_EMAIL = "alex.perez@fi.upm.es"
EXAMPLE_STUDENT_REP_NAME = "Javier Ruiz"
EXAMPLE_STUDENT_REP_EMAIL = "javier.ruiz@alumnos.upm.es"
EXAMPLE_DIRECTOR_PEER_NAME = "Elena Torres"
EXAMPLE_DIRECTOR_PEER_EMAIL = "elena.torres@fi.upm.es"
EXAMPLE_PROF1_EMAIL = "carmen.santos@fi.upm.es"
EXAMPLE_PROF2_EMAIL = "luis.martin@fi.upm.es"

MODEL_PROFILES = {
    EMAIL_WRITER_PROFILE: {
        **EMAIL_WRITER_SETTINGS,
        "system_prompt": "You are a concise, professional corporate email assistant.",
        "prompt_template": (
            "You are taking the role of {my_name}, {my_description}. You are reading an email sent to you.\n"
            "Your task is to write a professional, concise reply.\n"
            "IMPORTANT: Respond in the SAME LANGUAGE as the Original Email (e.g., Spanish -> Spanish, English -> English).\n\n"
            "### INSTRUCTIONS:\n"
            f"1. If you have enough information to reply, wrap your response in {MESSAGE_OPENING_TAG}...{MESSAGE_CLOSING_TAG} tags.\n"
            f"2. If you lack context or cannot reply, output {NO_MESSAGE_OPENING_TAG}I cannot reply because...{NO_MESSAGE_CLOSING_TAG}.\n"
            "3. Do not include subject lines or greetings/signatures outside the tags.\n\n"
            "### SAMPLE RESPONSE:\n"
            "Context from knowledge base:\n"
            "---\n"
            "Calendar available at 3 PM.\n"
            "---\n"
            "Conversation so far (oldest to newest):\n"
            "---\n"
            "(no prior messages)\n"
            "---\n"
            "Current email to answer:\n"
            "---\n"
            "Subject: Meeting request\n"
            "From: john.doe@example.com\n"
            "Body: Can we meet at 3 PM?\n"
            "---\n"
            "Output:\n"
            f"{MESSAGE_OPENING_TAG}\n"
            "Hi John,\n\n"
            "3 PM works for me. See you then.\n\n"
            "Best,\n"
            "{my_name}\n"
            f"{MESSAGE_CLOSING_TAG}\n\n"
            "### ACTUAL TASK:\n"
            "Context from knowledge base:\n"
            "---\n"
            "{rag_context}\n"
            "---\n"
            "Conversation so far (oldest to newest):\n"
            "---\n"
            "{thread_context}\n"
            "---\n"
            "Current email to answer:\n"
            "---\n"
            "Subject:\n"
            "{subject}\n"
            "From:\n"
            "{sender}\n"
            "Body:\n"
            "{body}\n"
            "---\n"
            "Output:\n"
        ),
    },
    THREAD_GROUPER_PROFILE: {
        **THREAD_GROUPER_SETTINGS,
        "system_prompt": (
            "You are an expert email thread reconstruction assistant."
        ),
        "production_task_description_start": (
            "You have received {inbox_count} inbox emails and {sent_count} sent emails "
            "from {my_name} ({my_description}) INBOX and SENT folders. "
            "Your task is to group all emails into threads and remove quoted text carefully. "
            f"All email addresses that belong to {{my_name}} are: {{my_email_addresses}}. "
            "Each email includes: 'id', 'threadID', 'from', 'to', 'date', 'subject', 'body'. "
            "'id' is the email server id and is not the thread id. "
            "'threadID' is not part of the original communication (and must not be written in the output XML). "
            "It is a weak hint produced by an automated subject-based grouping and can be wrong. "
            "Your task is to output XML, reconstructing the threads and removing quoted text when it is already part of another email."
        ),
        "dataset_task_description_start": (
            "You have received {email_count} emails from a single mailbox folder. "
            "Your task is to group all emails into threads and remove quoted text carefully. "
            "The input emails are already in approximate chronological order. "
            "Each email includes: 'id', 'threadID', 'from', 'to', 'subject', 'body', "
            "and may optionally include 'date'. "
            "'id' is a stable input identifier and is not the thread id. "
            "'threadID' is not part of the original communication (and must not be written in the output XML). "
            "It is a weak hint produced by an automated subject-based grouping and can be wrong. "
            "Your task is to output XML, reconstructing the threads and removing quoted text when it is already part of another email."
        ),
        "production_example": (
            "Input emails:\n"
            "Inbox:\n"
            f"{{'id': b'440', 'threadID': 1, 'from': '{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> undefined', 'to': '{EXAMPLE_PROF1_EMAIL}, \"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined, {EXAMPLE_PROF2_EMAIL} undefined undefined undefined', 'date': datetime.datetime(2020, 5, 4, 9, 8, 59, tzinfo=datetime.timezone.utc), 'subject': 'Erasmus', 'body': \"Good morning. I am a student from Italy currently studying at my university. I would like to join the Erasmus program at Universidad Politécnica de Madrid. Would it be possible to take courses from both the MSc in Artificial Intelligence and the Máster Universitario en Automática y Robótica, or should I choose just one? Thank you\"}}\n"
            f"{{'id': b'448', 'threadID': 3, 'from': '{EXAMPLE_STAFF_NAME} <{EXAMPLE_STAFF_EMAIL}>', 'to': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'date': datetime.datetime(2020, 6, 11, 13, 32, 15, tzinfo=datetime.timezone.utc), 'subject': 'Fwd: Fichero egresados', 'body': \"Hola, {DIRECTOR_NAME}. Te reenvío aquí lo último que hice yo. Es de mayo de 2020. No sé si {EXAMPLE_COLLEAGUE_NAME} hizo alguno posterior. ¿Te sirve? {EXAMPLE_STAFF_NAME}\"}}\n"
            f"{{'id': b'432', 'threadID': 4, 'from': '{EXAMPLE_STUDENT_REP_NAME} <{EXAMPLE_STUDENT_REP_EMAIL}>', 'to': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'date': datetime.datetime(2020, 6, 12, 8, 55, 27, tzinfo=datetime.timezone.utc), 'subject': 'Orla/graduación de alumnos del máster en IA', 'body': \"Hola {DIRECTOR_NAME}, como delegado del máster en Inteligencia Artificial, me gustaría trasladarte la consulta de varios alumnos acerca de si se va a hacer orla / acto de graduación para los estudiantes del máster, o por si el contrario corre bajo nuestra cuenta hacerlo. ¡Un saludo! {EXAMPLE_STUDENT_REP_NAME} Máster Universitario en Inteligencia Artificial\"}}\n"
            "Sent:\n"
            f"{{'id': b'441', 'threadID': 1, 'from': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'to': '{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> undefined', 'date': datetime.datetime(2020, 5, 4, 11, 16, 0, tzinfo=datetime.timezone.utc), 'subject': 'Re: Erasmus', 'body': \"I don't know about the other Degree; it is probably taught in another School, so that it can be tricky for you to attend both. Besides, I'm not sure you can follow courses from different School at the same time during your Erasmus. Please ask orex@fi.upm.es: they are in charge of the Erasmus Programme at Escuela Técnica Superior de Ingenieros Informáticos. regards -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}\\n\\nOn Mon, 4 May 2020 at 09:08, {EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> wrote:\\n> Good morning. I am a student from Italy currently studying at my university. I would like to join the Erasmus program at Universidad Politécnica de Madrid. Would it be possible to take courses from both the MSc in Artificial Intelligence and the Máster Universitario en Automática y Robótica, or should I choose just one? Thank you\"}}\n"
            f"{{'id': b'3278', 'threadID': 2, 'from': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'to': '{EXAMPLE_STAFF_EMAIL}', 'date': datetime.datetime(2020, 6, 10, 10, 39, 0, tzinfo=datetime.timezone.utc), 'subject': 'Estudio de egresados', 'body': \"Hola {EXAMPLE_STAFF_NAME}. Estoy con el informe de titulación. Cuál es el último informe de egresados que tenemos? Hay unos cuantos datos que estaría bien actualizar, como los ex-alumnos que están en el extranjero, los que están realizando un doctorado, etc. Además, dice aquí: El último estudio sobre empleabilidad realizado por el Observatorio Académico sobre titulaciones de postgrado de la ETSIINF es una encuesta fue realizada en el curso 2019/20. Tenemos uno más reciente? Gracias! -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}\"}}\n"
            f"{{'id': b'3279', 'threadID': 3, 'from': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'to': '{EXAMPLE_STAFF_EMAIL}', 'date': datetime.datetime(2020, 6, 11, 15, 0, 0, tzinfo=datetime.timezone.utc), 'subject': 'Fwd: Fichero egresados', 'body': \"Gracias {EXAMPLE_STAFF_NAME}. A ver lo que puedo sacar de aquí. Un saludo -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}\"}}\n"
            f"{{'id': b'433', 'threadID': 4, 'from': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'to': '{EXAMPLE_DIRECTOR_PEER_EMAIL}', 'date': datetime.datetime(2020, 6, 12, 9, 10, 0, tzinfo=datetime.timezone.utc), 'subject': 'Fwd: Orla/graduación de alumnos del máster en IA', 'body': \"Hola {EXAMPLE_DIRECTOR_PEER_NAME}. Este mensaje me pilla tan de sorpresa que no sé ni cómo empezar a contestar. No se supone que acabamos de estar en el Wanda? A ver si tú lo sabes interpretar...\"}}\n"
            "Output:\n"
            f"{THREAD_OPENING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> undefined{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}{EXAMPLE_PROF1_EMAIL}, \"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined, {EXAMPLE_PROF2_EMAIL} undefined undefined undefined{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Erasmus{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Good morning. I am a student from Italy currently studying at my university. I would like to join the Erasmus program at Universidad Politécnica de Madrid. Would it be possible to take courses from both the MSc in Artificial Intelligence and the Máster Universitario en Automática y Robótica, or should I choose just one? Thank you{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> undefined{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Re: Erasmus{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}I don't know about the other Degree; it is probably taught in another School, so that it can be tricky for you to attend both. Besides, I'm not sure you can follow courses from different School at the same time during your Erasmus. Please ask orex@fi.upm.es: they are in charge of the Erasmus Programme at Escuela Técnica Superior de Ingenieros Informáticos. regards -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_CLOSING_TAG}\n"
            f"{THREAD_OPENING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}{EXAMPLE_STAFF_EMAIL}{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Estudio de egresados{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Hola {EXAMPLE_STAFF_NAME}. Estoy con el informe de titulación. Cuál es el último informe de egresados que tenemos? Hay unos cuantos datos que estaría bien actualizar, como los ex-alumnos que están en el extranjero, los que están realizando un doctorado, etc. Además, dice aquí: El último estudio sobre empleabilidad realizado por el Observatorio Académico sobre titulaciones de postgrado de la ETSIINF es una encuesta fue realizada en el curso 2019/20. Tenemos uno más reciente? Gracias! -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}{EXAMPLE_STAFF_NAME} <{EXAMPLE_STAFF_EMAIL}>{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Fwd: Fichero egresados{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Hola, {DIRECTOR_NAME}. Te reenvío aquí lo último que hice yo. Es de mayo de 2020. No sé si {EXAMPLE_COLLEAGUE_NAME} hizo alguno posterior. ¿Te sirve? {EXAMPLE_STAFF_NAME}{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}{EXAMPLE_STAFF_EMAIL}{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Fwd: Fichero egresados{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Gracias {EXAMPLE_STAFF_NAME}. A ver lo que puedo sacar de aquí. Un saludo -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_CLOSING_TAG}\n"
            f"{THREAD_OPENING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}{EXAMPLE_STUDENT_REP_NAME} <{EXAMPLE_STUDENT_REP_EMAIL}>{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Orla/graduación de alumnos del máster en IA{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Hola {DIRECTOR_NAME}, como delegado del máster en Inteligencia Artificial, me gustaría trasladarte la consulta de varios alumnos acerca de si se va a hacer orla / acto de graduación para los estudiantes del máster, o por si el contrario corre bajo nuestra cuenta hacerlo. ¡Un saludo! {EXAMPLE_STUDENT_REP_NAME} Máster Universitario en Inteligencia Artificial{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}{EXAMPLE_DIRECTOR_PEER_EMAIL}{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Fwd: Orla/graduación de alumnos del máster en IA{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Hola {EXAMPLE_DIRECTOR_PEER_NAME}. Este mensaje me pilla tan de sorpresa que no sé ni cómo empezar a contestar. No se supone que acabamos de estar en el Wanda? A ver si tú lo sabes interpretar...{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_CLOSING_TAG}"
        ),
        "dataset_example": (
            "Input emails:\n"
            f"{{'id': '440', 'threadID': 1, 'from': '{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> undefined', 'to': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'subject': 'Erasmus', 'body': \"Good morning. I am a student from Italy currently studying at my university. I would like to join the Erasmus program at Universidad Politécnica de Madrid. Would it be possible to take courses from both the MSc in Artificial Intelligence and the Máster Universitario en Automática y Robótica, or should I choose just one? Thank you\"}}\n"
            f"{{'id': '441', 'threadID': 1, 'from': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'to': '{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> undefined', 'subject': 'Re: Erasmus', 'body': \"I don't know about the other Degree; it is probably taught in another School, so that it can be tricky for you to attend both. Besides, I'm not sure you can follow courses from different School at the same time during your Erasmus. Please ask orex@fi.upm.es: they are in charge of the Erasmus Programme at Escuela Técnica Superior de Ingenieros Informáticos. regards -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}\\n\\nOn Mon, 4 May 2020, {EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> wrote:\\n> Good morning. I am a student from Italy currently studying at my university. I would like to join the Erasmus program at Universidad Politécnica de Madrid. Would it be possible to take courses from both the MSc in Artificial Intelligence and the Máster Universitario en Automática y Robótica, or should I choose just one? Thank you\"}}\n"
            f"{{'id': '3278', 'threadID': 2, 'from': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'to': '{EXAMPLE_STAFF_EMAIL}', 'subject': 'Estudio de egresados', 'body': \"Hola {EXAMPLE_STAFF_NAME}. Estoy con el informe de titulación. Cuál es el último informe de egresados que tenemos? Hay unos cuantos datos que estaría bien actualizar, como los ex-alumnos que están en el extranjero, los que están realizando un doctorado, etc. Además, dice aquí: El último estudio sobre empleabilidad realizado por el Observatorio Académico sobre titulaciones de postgrado de la ETSIINF es una encuesta fue realizada en el curso 2019/20. Tenemos uno más reciente? Gracias! -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}\"}}\n"
            f"{{'id': '3279', 'threadID': 3, 'from': '{EXAMPLE_STAFF_NAME} <{EXAMPLE_STAFF_EMAIL}>', 'to': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'subject': 'Fwd: Fichero egresados', 'body': \"Hola, {DIRECTOR_NAME}. Te reenvío aquí lo último que hice yo. Es de mayo de 2020. No sé si {EXAMPLE_COLLEAGUE_NAME} hizo alguno posterior. ¿Te sirve? {EXAMPLE_STAFF_NAME}\"}}\n"
            f"{{'id': '3280', 'threadID': 3, 'from': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'to': '{EXAMPLE_STAFF_EMAIL}', 'subject': 'Fwd: Fichero egresados', 'body': \"Gracias {EXAMPLE_STAFF_NAME}. A ver lo que puedo sacar de aquí. Un saludo\\n\\nOn Wed, 11 Jun 2020, {EXAMPLE_STAFF_NAME} wrote:\\n> Hola, {DIRECTOR_NAME}. Te reenvío aquí lo último que hice yo. Es de mayo de 2020. No sé si {EXAMPLE_COLLEAGUE_NAME} hizo alguno posterior. ¿Te sirve? {EXAMPLE_STAFF_NAME}\"}}\n"
            f"{{'id': '3281', 'threadID': 4, 'from': '{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}>', 'to': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}>', 'subject': 'Questions about the programme', 'body': \"A\\n\\n\\nIs the programme taught in Spanish?\\n\\nPlease answer YES or NO\\n\\n\\nB\\n\\n\\nHow long does the master's degree last?\\n\\nPlease indicate the duration\\n\\n\\nC\\n\\n\\nIs the format online?\\n\\nPlease answer YES or NO\\n\\n\\nD\\n\\n\\nDo students work on real case studies?\\n\\nPlease answer YES or NO\"}}\n"
            "Output:\n"
            f"{THREAD_OPENING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> undefined{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Erasmus{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Good morning. I am a student from Italy currently studying at my university. I would like to join the Erasmus program at Universidad Politécnica de Madrid. Would it be possible to take courses from both the MSc in Artificial Intelligence and the Máster Universitario en Automática y Robótica, or should I choose just one? Thank you{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> undefined{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Re: Erasmus{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}I don't know about the other Degree; it is probably taught in another School, so that it can be tricky for you to attend both. Besides, I'm not sure you can follow courses from different School at the same time during your Erasmus. Please ask orex@fi.upm.es: they are in charge of the Erasmus Programme at Escuela Técnica Superior de Ingenieros Informáticos. regards -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_CLOSING_TAG}\n"
            f"{THREAD_OPENING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}{EXAMPLE_STAFF_EMAIL}{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Estudio de egresados{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Hola {EXAMPLE_STAFF_NAME}. Estoy con el informe de titulación. Cuál es el último informe de egresados que tenemos? Hay unos cuantos datos que estaría bien actualizar, como los ex-alumnos que están en el extranjero, los que están realizando un doctorado, etc. Además, dice aquí: El último estudio sobre empleabilidad realizado por el Observatorio Académico sobre titulaciones de postgrado de la ETSIINF es una encuesta fue realizada en el curso 2019/20. Tenemos uno más reciente? Gracias! -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}{EXAMPLE_STAFF_NAME} <{EXAMPLE_STAFF_EMAIL}>{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Fwd: Fichero egresados{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Hola, {DIRECTOR_NAME}. Te reenvío aquí lo último que hice yo. Es de mayo de 2020. No sé si {EXAMPLE_COLLEAGUE_NAME} hizo alguno posterior. ¿Te sirve? {EXAMPLE_STAFF_NAME}{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}{EXAMPLE_STAFF_EMAIL}{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Fwd: Fichero egresados{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}Gracias {EXAMPLE_STAFF_NAME}. A ver lo que puedo sacar de aquí. Un saludo{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_CLOSING_TAG}\n"
            f"{THREAD_OPENING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}>{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}>{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}Questions about the programme{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}A Is the programme taught in Spanish? Please answer YES or NO\nB How long does the master's degree last? Please indicate the duration\nC Is the format online? Please answer YES or NO\nD Do students work on real case studies? Please answer YES or NO{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_CLOSING_TAG}"
        ),
        "prompt_template": (
            "{task_description_start}\n\n"
            "### RULES:\n"
            "1. Preserve chronological order within each thread. Output messages in oldest-to-newest order, so the first message inside each thread is the oldest and the last message is the newest.\n"
            "2. Remove quoted text only when the quoted content appears elsewhere in the input as the same text "
            "with fewer or no quote markers. Keep the least-quoted instance.\n"
            "   - Example: if B contains \"> A\" and A appears elsewhere unquoted, remove \"> A\" from B.\n"
            "   - Example: if B contains only \"> A\" and A does not appear elsewhere, keep it.\n"
            "   - Example: if B has \"> A\" and C has \"> B\\n> A\", keep \"> A\" in B and remove the quoted part from C.\n"
            "   Reply headers such as 'En ... escribió:', 'On ... wrote:', 'De/Enviado/Para/Asunto' are a few indicators of quoted blocks.\n"
            f"3. Keep only the cleaned body inside {THREAD_BODY_OPENING_TAG}...{THREAD_BODY_CLOSING_TAG} tags.\n"
            "4. Normalize whitespace in cleaned bodies without removing information: trim unnecessary spaces, merge hard-wrapped lines that belong to the same paragraph, and collapse repeated blank lines to a single blank line.\n"
            "5. Do not hallucinate or add new messages.\n"
            "6. Output ONLY the thread XML, nothing else.\n\n"
            "### OUTPUT FORMAT:\n"
            f"{THREAD_OPENING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            f"{THREAD_FROM_OPENING_TAG}...{THREAD_FROM_CLOSING_TAG}\n"
            f"{THREAD_TO_OPENING_TAG}...{THREAD_TO_CLOSING_TAG}\n"
            f"{THREAD_SUBJECT_OPENING_TAG}...{THREAD_SUBJECT_CLOSING_TAG}\n"
            f"{THREAD_BODY_OPENING_TAG}...{THREAD_BODY_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_MESSAGE_OPENING_TAG}\n"
            "...\n"
            f"{THREAD_MESSAGE_CLOSING_TAG}\n"
            f"{THREAD_CLOSING_TAG}\n"
            f"{THREAD_OPENING_TAG}\n"
            "...\n"
            f"{THREAD_CLOSING_TAG}\n\n"
            "### SAMPLE RESPONSE:\n"
            "---\n"
            "{example_message}\n"
            "---\n\n"
            "### ACTUAL TASK:\n"
            "---\n"
            "Input emails:\n"
            "{emails_section}\n"
            "Output:\n"
        ),
    },
    DATA_CLEANER_PROFILE: {
        **DATA_CLEANER_SETTINGS,
        "system_prompt": (
            "You are an expert Knowledge Curator for RAG."
        ),
        "prompt_template": (
            "Your job is to take source texts and convert them into four outputs: cleaned text, abstract, summary, and Q&A pairs.\n"
            "To help you do this well, the following instructions explain the wider product that this knowledge-curation step supports.\n\n"
            "This data will be used by a retrieval system for current and prospective students interested in MUIA (Master's Degree in Artificial Intelligence), coordinated by the Department of Artificial Intelligence (DIA) at FI-UPM.\n"
            "UPM offers many degrees, master's programs, and PhD programs. FI-UPM also offers several different informatics-related degrees, master's programs, and PhD programs. The Department of Artificial Intelligence also appears in contexts that are not only about MUIA.\n"
            "Because of that, scope and contextualization are critical, especially in Q&A pairs.\n"
            "Do not confuse MUIA with other UPM programs.\n"
            f"Known non-MUIA UPM títulos propios include: {', '.join(NON_MUIA_TITULOS_PROPIOS)}.\n"
            f"Known non-MUIA official master's programs include: {', '.join(NON_MUIA_OFFICIAL_MASTERS)}.\n"
            "Your task is to transform the following raw text chunk (a fragment of content from a source URL) into a high-quality, English-language knowledge base entry.\n"
            f"Content usually comes from one of these {SOURCE_HOST_CATEGORY_COUNT} sources: {SOURCE_HOST_CATEGORY_TEXT}. If a source URL is outside this mapping, assume it belongs to the most general category unless the text clearly indicates another category.\n"
            f"This matters because many chunks include numbers (prices, credits, dates, requirements), and those facts are only correct when tied to the right scope ({SOURCE_CATEGORY_TEXT}).\n"
            "For each chunk, you must produce four outputs: cleaned text, abstract, summary, and Q&A pairs.\n\n"
            "### CONTEXTUAL INFORMATION:\n"
            "1. You will receive 'Page History' containing abstracts and summaries from previous chunks. Each entry includes its **Chunk Index** so you can determine its position in the document. **Note**: This history may be non-consecutive (e.g., Chunks 0-4 to provide you with beginning-of-document context followed by Chunks 35-39 to provide you with latest-chunks context).\n"
            "2. You will receive the 'Previous Chunk Cleaned Text' to ensure grammatical continuity with the current input (for example, a question or link might have been cut off and this may give you the chance to reconstruct it).\n\n"
            "### GENERAL INSTRUCTIONS:\n"
            "1. Write all outputs in English, regardless of the source language.\n"
            "2. Remove non-content noise from all outputs: navigation menus, footer boilerplate, image placeholders, table-of-contents dots, page-number artifacts, and similar formatting residue. Keep links only when they carry useful factual information.\n\n"
            "### OUTPUT-SPECIFIC INSTRUCTIONS:\n"
            f"1. **Cleaned Text**: Fix broken sentences and formatting issues (for example, content split across multiple lines), removing non-content noise (for example, menus, footers, image placeholders, table-of-contents artifacts, etc.), writing cleaned text in English, and outputting it inside {CLEANED_TEXT_OPENING_TAG} tags. This output is for cleanup, not relevance filtering: keep all remaining content and do not drop content based on relevance.\n"
            "2. **Abstract, Summary, Q&A type of content**: Include student-facing information (for example, admission and selection rules, recognition/transfer of credits, enrollment process (steps, required documents, deadlines, and fees), teaching and course logistics (language of instruction, schedules, subjects, seminars, passing requirements, evaluation timing/dates, and project types), mobility/exchange options, housing/support resources, quality/ranking context, etc.). Exclude document-internal/meta content (for example, 'what appears in table 14'), dense legal/governance details, project funding amounts, generic news, and one-off trivia (for example, who received a past award), unless directly useful for student-facing decisions or procedures.\n"
            "3. **Abstract, Summary, Q&A style**: Write in clear, user-facing language for students/prospective students and staff contacting the MUIA team. Avoid document-internal or meta wording such as 'what is mentioned', 'what appears in table X', or 'referenced in the text'. Do not use ambiguous wording like 'this program', 'it', or 'the reservation fee' without naming what they refer to.\n"
            f"4. **Abstract**: Provide a concise 1-sentence overview inside {ABSTRACT_OPENING_TAG} tags. Do not mention scope/category by default. Only if the text clearly indicates a scope different from the provided Source Category, add a brief scope note.\n"
            f"5. **Summary**: Provide a detailed, reorganized summary of the key facts inside {SUMMARY_OPENING_TAG} tags. Do not force scope/category labels. Only if the text clearly indicates a scope different from the provided Source Category, add a brief scope note.\n"
            f"6. **Q&A Pairs**: Generate up to {Q_AND_A_MAX_PAIRS} Q&A pairs inside {QUESTIONS_OPENING_TAG} tags, using {QUESTION_OPENING_TAG} and {ANSWER_OPENING_TAG} for each pair. Write questions that sound like what a current or prospective student would naturally ask in a search box or in an email, not like what a curator would ask after reading the page. Prioritize interesting, actionable, and decision-relevant questions over trivial, meta, or catalog-style questions. Each question and answer must be fully self-contained because source text and page context may not be available at retrieval time: never use vague references such as 'this program', 'this page', 'this resource', or 'in the text'. Answers must contain **all specific details**. You are not required to hit the maximum number of Q&A pairs: generate fewer pairs when the chunk supports fewer distinct, useful questions, and avoid repetitive or near-duplicate pairs.\n"
            "### Q&A RETRIEVAL CONSTRAINTS:\n"
            "CRITICAL: Each question will be embedded and retrieved independently. There is no shared context between questions. Therefore every question must be fully self-contained and must explicitly name the exact entity it refers to.\n"
            "HARD RULE: Every question must explicitly include the exact name of the entity it refers to, such as the specific program, faculty, department, campus, scholarship, event, document, or service.\n"
            "NEVER use vague references such as 'the program', 'this program', 'the master's program', 'the doctorate', 'this degree', 'the faculty', 'the department', 'the campus', 'students', 'admission requirements', 'duration', 'structure', 'career outcomes', 'the official page', 'the website', 'it', 'its', or similar wording unless the exact entity name also appears in the same question.\n"
            "Repetition is required, not a problem. If multiple questions refer to the same entity, repeat the entity name in every question.\n"
            "Before finalizing each question, perform this check: if the question is read alone, without the source text, page history, summary, or any previous question, is the target entity still unambiguous? If not, rewrite it.\n"
            "Output only questions that pass this check.\n"
            "### Q&A RELEVANCE CONSTRAINTS:\n"
            "Questions must be genuinely useful for retrieval and useful for a student or prospective student. A good question helps a person decide whether to apply, enroll, plan studies, understand costs, understand requirements, compare options, or solve an academic or administrative issue.\n"
            "Prefer questions about admissions, deadlines, fees, scholarships, modality, language of instruction, credits, duration, schedules, curriculum structure, mobility, internships, final projects, research opportunities, career outcomes, contacts, official procedures, and important recognition or ranking information.\n"
            "Avoid low-value questions that merely restate labels, headings, organization charts, or isolated names unless that information clearly helps a student make a decision or solve a real problem.\n"
            "Bad question example: 'What is the full name of the department that the Department of Artificial Intelligence belongs to?' This is too indirect, unnatural, and low-value for student retrieval.\n"
            "Bad question example: 'What is the name of the course that focuses on medical data analysis?' This is too narrow and catalog-like unless the source clearly presents that course as an important decision point.\n"
            "Good question example: 'Which languages is the MUIA taught in?'\n"
            "Good question example: 'What is the cost of completing the MUIA?'\n"
            "Good question example: 'What scholarships are available for the MUIA?'\n"
            "Good question example: 'What career outcomes does the MUIA offer?'\n"
            "If a candidate question feels like a fact lookup that no normal student would ask, do not include it.\n"
            "If several candidate questions are all asking about the same fact, keep only the single most useful, natural, and general version.\n"
            "If a chunk contains only one or two genuinely useful questions, output only one or two questions.\n"
            # "### Q&A QUALITY GUIDELINES:\n"
            # "**Negative Examples (Avoid These Mistakes):**\n\n"
            # "Example 1 (Irrelevance):\n"
            # f"{QUESTION_OPENING_TAG}What is the note regarding the currency and language of the ticket pricing page?{QUESTION_CLOSING_TAG}\n"
            # f"{ANSWER_OPENING_TAG}The ticket pricing page is in Spanish.{ANSWER_CLOSING_TAG}\n"
            # "Critique: Not relevant. Do not include meta-commentary about the page format.\n\n"
            # "Example 2 (Outdated Information & Meta-pairs):\n"
            # f"{QUESTION_OPENING_TAG}When was the MLAS event held that is referenced in the text?{QUESTION_CLOSING_TAG}\n"
            # f"{ANSWER_OPENING_TAG}The 17th MLAS event was held in 2025, and the information is outdated as of 2026.{ANSWER_CLOSING_TAG}\n"
            # f"{QUESTION_OPENING_TAG}What are the ticket prices for buses 591, 865, and the light rail?{QUESTION_CLOSING_TAG}\n"
            # f"{ANSWER_OPENING_TAG}A single trip costs 2 euros, while a 10-ride ticket costs 12.20 euros.{ANSWER_CLOSING_TAG}\n"
            # "Critique: Good intention, but the fact that information may be outdated should be embedded in each specific pair, not as a separate meta-pair.\n"
            # "**Better Version:**\n"
            # f"{QUESTION_OPENING_TAG}What are the ticket prices for buses 591, 865, and the light rail?{QUESTION_CLOSING_TAG}\n"
            # f"{ANSWER_OPENING_TAG}As of 2025 (17th MLAS event), a single trip costs 2 euros, while a 10-ride ticket costs 12.20 euros.{ANSWER_CLOSING_TAG}\n\n"
            # "Example 3 (Missing Context):\n"
            # f"{QUESTION_OPENING_TAG}What is the recommended route after exiting the light-rail station?{QUESTION_CLOSING_TAG}\n"
            # f"{ANSWER_OPENING_TAG}Participants should turn right and walk down Avda. Montepríncipe, as indicated by the Google Maps link.{ANSWER_CLOSING_TAG}\n"
            # "Critique: Participants of what? Recommended route to where? Each pair must be standalone.\n"
            # "**Better Version:**\n"
            # f"{QUESTION_OPENING_TAG}What is the recommended route for MLAS participants after exiting the light-rail station?{QUESTION_CLOSING_TAG}\n"
            # f"{ANSWER_OPENING_TAG}MLAS participants should turn right after exiting the light-rail station and walk down Avda. Montepríncipe.{ANSWER_CLOSING_TAG}\n\n"
            # "Example 4 (Ambiguous Entities):\n"
            # f"{QUESTION_OPENING_TAG}Which metro lines connect to the campus via bus 865?{QUESTION_CLOSING_TAG}\n"
            # f"{ANSWER_OPENING_TAG}Bus 865 connects to the campus from Moncloa (metro lines 6 and 3).{ANSWER_CLOSING_TAG}\n"
            # "Critique: Which campus? Questions and answers must be specific.\n"
            # "**Better Version:**\n"
            # f"{QUESTION_OPENING_TAG}Which metro lines connect to the Montegancedo Campus via bus 865?{QUESTION_CLOSING_TAG}\n"
            # f"{ANSWER_OPENING_TAG}Bus 865 connects to the Montegancedo Campus from Moncloa (metro lines 6 and 3).{ANSWER_CLOSING_TAG}\n\n"
            # "Example 5 (False Specificity):\n"
            # f"{QUESTION_OPENING_TAG}Which group has collaborated with companies like Progenika Biopharma and Panda Security?{QUESTION_CLOSING_TAG}\n"
            # f"{ANSWER_OPENING_TAG}The Computational Intelligence Group (CIG) has collaborated with companies such as Progenika Biopharma and Panda Security.{ANSWER_CLOSING_TAG}\n"
            # "Critique: Asking 'Which group...' implies CIG is the only one. It is safer to invert the question.\n"
            # "**Better Version:**\n"
            # f"{QUESTION_OPENING_TAG}Which companies has the Computational Intelligence Group (CIG) collaborated with?{QUESTION_CLOSING_TAG}\n"
            # f"{ANSWER_OPENING_TAG}The Computational Intelligence Group (CIG) has collaborated with companies such as Progenika Biopharma and Panda Security.{ANSWER_CLOSING_TAG}\n"
            "### EXAMPLE TRANSFORMATION:\n"
            "**Page History Context**:\n"
            "[list of previous abstracts/summaries with chunk indices...]\n"
            "**Previous Chunk Cleaned Text**:\n"
            "[cleaned English text from previous chunk would be here...]\n"
            "**Input Text**:\n"
            "-----\n\n"
            "Reconocido en el ranking de mejores másteres de España en informática especializada, publicado por el periódico El Mundo, entre los tres mejores durante catorce ediciones.\n"
            "-----\n"
            "#### 0\n\n"
            "### Año académico\n\n"
            "#### 0\n\n"
            "### Créditos ECTS\n\n"
            "#### 0\n\n"
            "### Plazas\n\n"
            "Un máster con\n\n"
            "orientación\n\n"
            "investigadora\n"
            "-----\n"
            "Organizado e impartido por el [Departamento de Inteligencia Artificial (DIA)](https://dia.fi.upm.es/), forma parte de la oferta de postgrado de la Escuela Técnica Superior de Ingenieros Informáticos de la Universidad Politécnica de Madrid. Con una orientación investigadora, comenzó a impartirse en el curso académico 2010/11, y proporciona una formación de calidad en diversos campos de investigación actuales en inteligencia artificial.\n\n"
            "**Output**:\n"
            f"{ABSTRACT_OPENING_TAG}\n"
            "Overview of the research-oriented Master in Artificial Intelligence (MUIA) and its national recognition.\n"
            f"{ABSTRACT_CLOSING_TAG}\n"
            f"{SUMMARY_OPENING_TAG}\n"
            "The Master in Artificial Intelligence (MUIA) is a research-oriented program organized by the Department of Artificial Intelligence (DIA) at the Technical University of Madrid (UPM). It has been consistently ranked as one of the top three masters in specialized informatics in Spain. It began in the 2010/11 academic year.\n"
            f"{SUMMARY_CLOSING_TAG}\n"
            f"{CLEANED_TEXT_OPENING_TAG}\n"
            "Recognized in the ranking of the best masters in Spain in specialized informatics, published by the newspaper El Mundo, among the top three for fourteen editions.\n\n"
            "A master's degree with a research orientation.\n\n"
            "Organized and taught by the [Department of Artificial Intelligence (DIA)](https://dia.fi.upm.es/), it is part of the postgraduate offer of the School of Computer Engineering of the Technical University of Madrid (UPM). With a research orientation, it began in the 2010/11 academic year, and provides quality training in various current research fields in artificial intelligence.\n"
            f"{CLEANED_TEXT_CLOSING_TAG}\n"
            f"{QUESTIONS_OPENING_TAG}\n"
            f"{QUESTION_OPENING_TAG}What is the primary orientation of the MUIA master's degree?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}Research-oriented.{ANSWER_CLOSING_TAG}\n"
            f"{QUESTION_OPENING_TAG}Where is the MUIA master's degree taught?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}It is taught at the School of Computer Engineering of the Technical University of Madrid (UPM).{ANSWER_CLOSING_TAG}\n"
            f"{QUESTION_OPENING_TAG}How has the MUIA program been recognized in national rankings?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}It has been recognized as one of the best masters in specialized informatics in Spain by El Mundo newspaper (among the top three for fourteen editions).{ANSWER_CLOSING_TAG}\n"
            f"{QUESTION_OPENING_TAG}When did the MUIA program begin?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}It began in the 2010/11 academic year.{ANSWER_CLOSING_TAG}\n"
            f"{QUESTIONS_CLOSING_TAG}\n\n"
            "### CURRENT TASK:\n"
            "**Current Date**: {datetime}\n"
            "**Source URL**: {source_url}\n"
            "**Source Category**: {source_category}\n"
            "**Page History Context**:\n{page_history_context}\n"
            "**Previous Chunk Cleaned Text**:\n{previous_chunk_context}\n"
            "**Input Text**:\n{text}\n\n"
            "**Output**:\n"
        ),
    },
    EMAIL_KNOWLEDGE_BASE_CURATOR_PROFILE: {
        **EMAIL_KNOWLEDGE_BASE_CURATOR_SETTINGS,
        "system_prompt": (
            "You are an expert Knowledge Curator for RAG."
        ),
        "prompt_template": (
            f"Your job is to do one of two things for an email thread: either convert it into reusable knowledge for retrieval, or, if the thread does not contain reusable institutional knowledge, output only {NO_USEFUL_INFORMATION_OPENING_TAG}{NO_USEFUL_INFORMATION_CLOSING_TAG}.\n"
            "This data will be used by a retrieval system for current and prospective students interested in MUIA (Master's Degree in Artificial Intelligence), coordinated by the Department of Artificial Intelligence (DIA) at FI-UPM.\n\n"
            "### GENERAL INSTRUCTIONS:\n"
            "1. Write all outputs in English.\n"
            "2. The input is not pre-anonymized. You must anonymize the outputs yourself by removing or generalizing personal names, personal email addresses, phone numbers, student identifiers, attachment references, and one-off case details that are not reusable.\n"
            "3. Treat institutional answers as the main source of reusable knowledge. This includes replies from the coordinator, the administrative office, official university accounts, or another institutional representative speaking in that role.\n"
            "4. Treat current or prospective students, applicants, and other non-institutional participants as external participants. Information provided only by external participants is not useful by itself unless it is needed to understand an institutional answer in the same thread.\n"
            "5. Keep reusable institutional knowledge such as admission criteria, eligibility rules, required documents, deadlines, conditions, next steps, procedures, generic contact points, decision meanings, and realistic alternatives offered by the institution.\n"
            "6. Remove email-only noise from all outputs: greetings, signatures, confidentiality notices, repeated acknowledgements, repeated disclaimers, and quoted or forwarded text that only duplicates information already present elsewhere in the thread.\n"
            "7. If quoted or forwarded text is needed to understand an institutional answer, keep only the minimum necessary information and rewrite it in clean reusable form.\n"
            "8. Do not invent policy, deadlines, requirements, or explanations that are not supported by the thread.\n"
            f"9. If the thread contains only a request, only an acknowledgement, or discussion without a reusable institutional answer, output only {NO_USEFUL_INFORMATION_OPENING_TAG}{NO_USEFUL_INFORMATION_CLOSING_TAG}.\n\n"
            "### OUTPUT-SPECIFIC INSTRUCTIONS:\n"
            f"1. **Cleaned Text**: Produce a cleaned version inside {CLEANED_TEXT_OPENING_TAG} tags. This is a cleanup output, not a summary and not a rewrite. Keep all remaining substantive institutional content, preserve the original facts, wording, and structure as much as possible, translate to English, anonymize when needed, and remove only noise such as greetings, signatures, repeated acknowledgements, and duplicated quoted text. This output is for cleanup, not relevance filtering: keep all remaining content and do not drop content based on relevance.\n"
            f"2. **Abstract**: Produce a concise 1-sentence overview inside {ABSTRACT_OPENING_TAG} tags.\n"
            f"3. **Summary**: Produce a detailed, reorganized summary inside {SUMMARY_OPENING_TAG} tags. It should usually be shorter than the cleaned text, but it must retain important specific details such as named regulations, deadlines, conditions, required documents, decision criteria, and alternatives when they appear in the thread.\n"
            f"4. **Q&A Pairs**: Generate up to {Q_AND_A_MAX_PAIRS} Q&A pairs inside {QUESTIONS_OPENING_TAG} tags, using {QUESTION_OPENING_TAG} and {ANSWER_OPENING_TAG} for each pair. Questions should sound like what a student or staff member would naturally ask later, not like curator notes. Keep each question and answer self-contained, anonymized, and free of vague references such as 'this student', 'this case', or 'this email'. Answers must contain the concrete details needed to stand alone.\n"
            "5. Prefer generalizable questions about requirements, denial reasons, deadlines, next steps, conditions, who to contact, what a decision means, or how a procedure works.\n"
            "6. If the thread contains little reusable knowledge, generate fewer Q&A pairs rather than inventing weak ones.\n"
            f"7. If the thread does not contain reusable institutional knowledge, output only {NO_USEFUL_INFORMATION_OPENING_TAG}{NO_USEFUL_INFORMATION_CLOSING_TAG}.\n\n"
            "### EXAMPLE 1:\n"
            "**Input Email Thread**:\n"
            "-----\n"
            "From: masteria.dia@fi.upm.es\n"
            "To: applicant@example.com\n"
            "Subject: Denial - Master Universitario en Inteligencia Artificial at UPM\n\n"
            "Dear applicant,\n\n"
            "I regret having to deny your admission to the MUIA. The program is highly demanded and the number of applicants is much higher than the number of available places.\n\n"
            "When making decisions, we prioritize applicants with strong computer science backgrounds and excellent previous academic results. The university of origin is also considered among other factors. Due to accreditation requirements such as Euro-Inf, professional experience cannot replace academic training, although it may still be considered.\n\n"
            "Frequent denial reasons include having a degree too far from computer science, not having a sufficiently high average grade, coming from a university that is not considered among the strongest, or not proving Spanish proficiency. The main general constraint is also the limited number of available places.\n\n"
            "If you remain interested in studying in our department, you may consider the Master's Degree in Data Science or several department-specific degrees as alternatives.\n\n"
            "Best regards,\n"
            "Coordinator\n"
            "-----\n"
            "**Output**:\n"
            f"{ABSTRACT_OPENING_TAG}\n"
            "Common reasons for denial of admission to the MUIA and possible alternative study options.\n"
            f"{ABSTRACT_CLOSING_TAG}\n"
            f"{SUMMARY_OPENING_TAG}\n"
            "Admission to the MUIA is highly competitive because the number of applicants exceeds the number of available places. Selection prioritizes strong computer science backgrounds and excellent academic records, while university of origin may also be considered. Due to accreditation requirements such as Euro-Inf, professional experience cannot replace academic training, although it may still be considered. Common denial reasons include having a degree too far from computer science, not having a sufficiently high average grade, coming from a university that is not considered among the strongest, not proving Spanish proficiency, and the limited number of available places. Applicants who are not admitted may consider the Master's Degree in Data Science or department-specific degrees as alternatives.\n"
            f"{SUMMARY_CLOSING_TAG}\n"
            f"{CLEANED_TEXT_OPENING_TAG}\n"
            "I regret having to deny your admission to the MUIA. The program is highly demanded, and the number of applicants is much higher than the number of available places.\n\n"
            "When making decisions, we prioritize applicants with strong computer science backgrounds and excellent previous academic results. The university of origin is also considered among other factors. Due to accreditation requirements such as Euro-Inf, professional experience cannot replace academic training, although it may still be considered.\n\n"
            "Frequent denial reasons include having a degree too far from computer science, not having a sufficiently high average grade, coming from a university that is not considered among the strongest, or not proving Spanish proficiency. The limited number of available places is also a major general constraint.\n\n"
            "If you remain interested in studying in our department, you may consider the Master's Degree in Data Science or department-specific degrees as alternatives.\n"
            f"{CLEANED_TEXT_CLOSING_TAG}\n"
            f"{QUESTIONS_OPENING_TAG}\n"
            f"{QUESTION_OPENING_TAG}What are common reasons for denial of admission to the MUIA?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}Common reasons for denial of admission to the MUIA include having a degree too far from computer science, not having a sufficiently high average grade, coming from a university that is not considered among the strongest, not proving Spanish proficiency, and the limited number of available places compared with the number of qualified applicants.{ANSWER_CLOSING_TAG}\n"
            f"{QUESTION_OPENING_TAG}Can professional experience replace academic training for admission to the MUIA?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}No. Due to accreditation requirements such as Euro-Inf, professional experience cannot replace academic training for admission to the MUIA, although it may still be considered as an additional factor.{ANSWER_CLOSING_TAG}\n"
            f"{QUESTION_OPENING_TAG}What alternatives can be considered after denial of admission to the MUIA?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}Applicants who are not admitted to the MUIA may consider the Master's Degree in Data Science or department-specific degrees as alternative study options.{ANSWER_CLOSING_TAG}\n"
            f"{QUESTIONS_CLOSING_TAG}\n\n"
            "### EXAMPLE 2:\n"
            "**Input Email Thread**:\n"
            "-----\n"
            "From: tramitacion.master.oficial@upm.es\n"
            "To: applicant@example.com\n"
            "Cc: masteria.dia@fi.upm.es\n"
            "Subject: Admission to the Master Universitario en Inteligencia Artificial at UPM\n\n"
            "Dear student,\n\n"
            "Once your master's pre-registration documents have been reviewed, you have been granted conditional access pending completion of your studies. This does not imply final admission, because final admission still depends on the master's program coordinators.\n\n"
            "According to Royal Decree 822/2021, it is possible to enroll in a master's program with only the bachelor's final project and up to 9 ECTS still pending. Therefore, before 15 October 2022, you must send an official academic transcript showing both the completed credits and the credits still pending. This document must be sent through this same channel in PDF format and must be smaller than two megabytes.\n\n"
            "Kind regards,\n"
            "Administrative Office\n"
            "-----\n"
            "**Output**:\n"
            f"{ABSTRACT_OPENING_TAG}\n"
            "Meaning of conditional access to the MUIA and the pending-document requirement before final admission.\n"
            f"{ABSTRACT_CLOSING_TAG}\n"
            f"{SUMMARY_OPENING_TAG}\n"
            "Conditional access means that the documentation is valid for access to master's studies, but it does not imply final admission, which still depends on the master's program coordinators. Under Royal Decree 822/2021, a student may enroll with only the bachelor's final project and up to 9 ECTS still pending. In that case, an official academic transcript showing completed and pending credits must be sent before 15 October 2022 through the same communication channel, in PDF format, and with a file size smaller than two megabytes.\n"
            f"{SUMMARY_CLOSING_TAG}\n"
            f"{CLEANED_TEXT_OPENING_TAG}\n"
            "Once the master's pre-registration documents have been reviewed, the applicant has been granted conditional access pending completion of studies. This does not imply final admission, because final admission still depends on the master's program coordinators.\n\n"
            "According to Royal Decree 822/2021, it is possible to enroll in a master's program with only the bachelor's final project and up to 9 ECTS still pending.\n\n"
            "Therefore, before 15 October 2022, the applicant must send an official academic transcript showing both the completed credits and the credits still pending. This document must be sent through the same communication channel in PDF format and must be smaller than two megabytes.\n"
            f"{CLEANED_TEXT_CLOSING_TAG}\n"
            f"{QUESTIONS_OPENING_TAG}\n"
            f"{QUESTION_OPENING_TAG}Does conditional access to the MUIA mean that the student has already been admitted?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}No. Conditional access to the MUIA means that the applicant's documentation is valid for access to master's studies, but final admission still depends on the master's program coordinators.{ANSWER_CLOSING_TAG}\n"
            f"{QUESTION_OPENING_TAG}Can a student enroll in the MUIA with the bachelor's final project and some credits still pending?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}Yes. Under Royal Decree 822/2021, a student may enroll in the MUIA with only the bachelor's final project and up to 9 ECTS still pending.{ANSWER_CLOSING_TAG}\n"
            f"{QUESTION_OPENING_TAG}What document must be sent after receiving conditional access to the MUIA?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}After receiving conditional access to the MUIA, the student must send an official academic transcript showing both the completed credits and the credits still pending.{ANSWER_CLOSING_TAG}\n"
            f"{QUESTION_OPENING_TAG}What are the submission requirements for the transcript after conditional access to the MUIA?{QUESTION_CLOSING_TAG}\n"
            f"{ANSWER_OPENING_TAG}The transcript must be sent before 15 October 2022 through the same communication channel, in PDF format, and with a file size smaller than two megabytes.{ANSWER_CLOSING_TAG}\n"
            f"{QUESTIONS_CLOSING_TAG}\n\n"
            "### EXAMPLE 3:\n"
            "**Input Email Thread**:\n"
            "-----\n"
            "From: applicant@example.com\n"
            "To: masteria.dia@fi.upm.es\n"
            "Subject: Re: Erasmus information\n\n"
            "Thank you very much for the information. I understand everything now.\n"
            "-----\n"
            "**Output**:\n"
            f"{NO_USEFUL_INFORMATION_OPENING_TAG}{NO_USEFUL_INFORMATION_CLOSING_TAG}\n\n"
            "### CURRENT TASK:\n"
            "**Input Email Thread**:\n{thread_text}\n\n"
            "**Output**:\n"
        ),
    },
    QUERY_REWRITER_PROFILE: {
        **QUERY_REWRITER_SETTINGS,
        "system_prompt": (
            "You are an expert at generating retrieval queries."
        ),
        "prompt_template": (
            f"Your task is to do one of two things for the following email: either generate retrieval queries and one reranker query, or, if the latest email does not contain a real current request, output only {NO_REQUEST_OPENING_TAG}{NO_REQUEST_CLOSING_TAG}.\n"
            "When you do generate queries, the goal is to maximize the chances of retrieving similar and useful chunks from a knowledge base.\n\n"
            "### KNOWLEDGE BASE CONTEXT:\n"
            "You are part of a system that is meant to answer emails for the Master Universitario en Inteligencia Artificial (MUIA). The knowledge base was built by starting from webpages about that master's program and then following links into the Department of Artificial Intelligence, Escuela Técnica Superior de Ingenieros Informáticos (the school that hosts the program), and Universidad Politecnica de Madrid. "
            "Because of that, MUIA is the default scope of the system and of the emails. Department, school, and university content is additional scope that becomes relevant only when the email explicitly goes beyond the default MUIA context, for example by mentioning another program, or a broader university-level procedure. "
            "It mostly contains public academic and administrative information such as admissions, enrollment, tuition and payment procedures, scholarships, mobility and Erasmus information, schedules, subjects and seminars, TFM (master's thesis) rules and procedures, awards and honors, forms, deadlines, responsible offices, and contact points. It does not contain day-to-day internal case tracking, student-specific progress updates, or one-off operational follow-ups, and it is refreshed yearly rather than continuously. "
            "Those webpages were chunked. Another language model then cleaned the chunks, translated the content into English, generated summaries, "
            "and generated question-answer pairs. The resulting knowledge base contains three main collection types: cleaned text, summaries with explicit entities, and question-answer style text. "
            "That text is in English. The only recurring exception is that some key named entities appear in some chunks in Spanish and in other chunks in English, because the cleaning model sometimes kept those names in the original language instead of translating them. "
            "This happens with entities such as 'Matrícula de Honor', subject or seminar names, school names, office names, and some acronyms. It does not apply to ordinary descriptive words such as 'certificado', 'admisión', or 'participación'.\n\n"
            "### REQUEST DETECTION:\n"
            "- First determine whether the latest email contains a real current request.\n"
            f"- If it does not, output only {NO_REQUEST_OPENING_TAG}{NO_REQUEST_CLOSING_TAG} and nothing else.\n"
            "- Judge the latest state of the conversation, not the topic named in the subject line by itself.\n"
            "- If the latest email only acknowledges receipt, says thank you, confirms understanding, sends a file that was already requested, closes the thread, or contains generic politeness with no current request, treat it as no-request.\n"
            "- If quoted or forwarded history contains an older request but the latest unquoted message no longer asks for anything, treat it as no-request.\n"
            "- Do not manufacture a request from the subject line when the latest body does not ask for information, clarification, or action.\n"
            "- If you decide the email is no-request, stop there and do not generate queries or a reranker query.\n\n"
            "### RETRIEVAL CONTEXT AND RULES:\n"
            "The following retrieval rules apply only when the latest email does contain a real current request and you are generating queries rather than outputting no-request.\n"
            "Retrieval uses several encoders, including sparse frequency-based encoders and dense encoders. Long emails often contain many words that are not useful for retrieval. "
            "Because of this, generate a mix of query styles: some very short keyword-focused queries for sparse frequency-based encoders, some clear natural-language queries for dense encoders, "
            "some HYDE-style queries that look like a short hypothetical answer or document snippet that would ideally answer the email, and some queries written as questions to match question-answer pairs in the knowledge base. This also applies to keyword queries: keep the rest of every keyword query in English, and switch only the named entity itself between English and Spanish when the knowledge-base rule above explicitly allows it. KEYWORD QUERIES ARE NOT SPANISH NOUN LISTS. DO NOT let one Spanish keyword cause the whole keyword query to switch into Spanish. "
            "Natural queries must not be written as questions. Write them as natural-language search formulations or concise declarative retrieval requests, without a question mark. "
            "For HYDE-style queries, do not write generic abstractions such as 'the subject requires specific prior knowledge'. Instead, write short answer-shaped snippets that look like plausible document text we would hope to retrieve. "
            "HYDE queries should be more document-like and more instance-level than natural queries: include plausible concrete details, concepts, requirements, or outcomes that a real matching chunk might contain. "
            "Generate multiple different plausible hypotheses when appropriate, because only some of them may match the real document. "
            "Avoid meta-document formulations such as 'the course guide explains...', 'the evaluation section states...', or 'the page contains information about...'. Prefer direct candidate answers instead. "
            "Do not hedge HYDE queries with formulations such as 'may require', 'would depend on', or 'should identify'. Write them as if they were short snippets copied from a matching chunk. "
            "However, do not invent unsupported administrative facts such as exact dates, deadlines, official procedures, prices, document names, grades, or people. It is acceptable to hypothesize plausible domain concepts for an academic topic when the email strongly suggests them, and it is always acceptable to restate concrete facts that already appear in the email.\n\n"
            "### SUBJECT AND SEMINAR MAPPING RULES:\n"
            "The following is the current best mapping we have between subject and seminar codes and subject and seminar names in Spanish and English. "
            "We provide this mapping because people usually mention the subject or seminar name in an email, not the code, but the cleaner model sometimes kept the Spanish name, "
            "sometimes translated the name into English, and sometimes replaced the name with the code. Use this mapping only when the email is clearly about a specific subject or seminar that appears in this mapping. "
            "If the email is about the master's program in general, ignore this mapping completely. Also ignore it when the email merely contains words that happen to overlap with part of a subject or seminar name. "
            "If the email explicitly mentions a concrete subject or seminar and the mapping suggests a code, generate queries using only these two routes: the English name together with the code, and the Spanish name together with the code. HARD RULE: NEVER generate a code-only query. NEVER write a seminar or subject code such as S1, S2, A4, or S16 by itself without the corresponding subject or seminar name. If a generated query contains only the code, that query is WRONG and must be rewritten.\n"
            f"{OLD_MUIA_SUBJECT_AND_SEMINAR_CODE_MAPPING_TEXT}\n\n"
            "### ADDITIONAL RULES:\n"
            "The following additional rules also apply only when you are generating queries rather than outputting no-request.\n"
            "- Prioritize the main administrative or academic request in the email.\n"
            "- Anchor the queries on what is being asked now, in the latest state of the email. If quoted or forwarded text provides necessary background to understand the current request, use that background too, but do not let stale earlier details take over the queries.\n"
            "- If the email contains multiple distinct requests, generate queries for the different requests.\n"
            "- If an email asks for different facts such as prerequisites, final exam, schedule, credits, deadline, or contact, prefer separate queries when combining them would make the query too specific or would require a single chunk to contain all the facts. It is still acceptable to combine related facts when they are likely to appear together in the same chunk.\n"
            "- Remove greetings, signatures, politeness filler, and irrelevant conversational text.\n"
            "- Queries are used as-is against a knowledge base that can contain other programs, departments, and university entities. Do not use vague references such as 'this master's', 'the program', or 'that subject'. Make the query concrete through the actual subject, seminar, office, procedure, URL, professor or staff name, or external program or university explicitly mentioned in the email. HARD RULE: NEVER use dates, years, long numbers, student names, applicant names, or generic person references to make the query concrete.\n"
            "- Treat the following as the default context of the email system: the Master Universitario en Inteligencia Artificial, the Department of Artificial Intelligence, Escuela Técnica Superior de Ingenieros Informáticos, and Universidad Politécnica de Madrid. NEVER mention any of those default names in the query. This ban also includes shorthand or translated forms such as MUIA, MIA, Master in Artificial Intelligence, UPM, ETSIINF, or the default department name in English or Spanish. If a query mentions one of those default names without an explicit contrast with another named program or institution, that query is WRONG and must be rewritten. Reason: those default names appear across many chunks in the knowledge base, so mentioning them can create easy but low-value matches that are about the right institution but not about the real issue in the email.\n"
            "- If the email explicitly contrasts the default MUIA context with another named program, department, school, or university, mention the external entity that creates that contrast and keep the default MUIA context implicit. Reason: the external entity is what disambiguates retrieval. For example, a mobility case should produce both some queries about a mobility program with an external university and some queries with the explicit external university name. Reason: if a mobility agreement with that university exists, the university name may appear in the knowledge base, but if it does not, the more general mobility-program wording still has a chance to retrieve the relevant procedure.\n"
            "- When the email includes an acronym, abbreviation, or short code together with its meaning, diversify across forms. Do not let all queries depend on the acronym alone. Prefer a mix where some queries use the expanded meaning, some use both the acronym and the expanded meaning, and only a minority rely on the acronym by itself.\n"
            "- Make the queries specific and self-contained.\n"
            "- Question queries should usually sound like broad, natural student questions that directly ask for the needed information. Prefer formulations such as 'what', 'how', or 'which' over generating many narrow yes/no variants. Use more specific yes/no questions only for a minority of the question queries when they seem especially useful.\n"
            "- Prefer queries about the underlying policy, procedure, requirement, approval rule, eligibility criterion, responsible office, or document type, rather than incidental case-specific details that are unlikely to appear in the knowledge base.\n"
            "- Avoid centering queries on details such as paper titles, conference names, journal names, exact acceptance dates, exact submission dates, or one-off filenames unless the email is explicitly asking where to find that exact item.\n"
            "- Do not use vague placeholders such as 'case', 'issue', 'matter', 'request', or 'situation' as if they were identifying retrieval terms. The queries are used as-is, and the knowledge base contains yearly refreshed public information, not day-to-day internal case updates. So a query such as 'updates to the situation' is bad twice: it is vague, and it is asking for a type of information the knowledge base does not contain.\n"
            "- When an email describes a very specific case, try to abstract it to the more general administrative or academic question that would govern the answer.\n"
            "- Natural queries must not be written as questions. Write them as natural-language search formulations or concise declarative retrieval requests, without a question mark.\n"
            "- Example for a TFM award email that mentions a specific conference acceptance:\n"
            "  Bad query: 'Computing in Cardiology 2024 accepted paper June 25'\n"
            "  Better queries: 'TFM matrícula de honor publication acceptance requirement', 'best TFM award publication criteria accepted before deadline', 'TFM prize eligibility accepted publication evidence'\n"
            f"- Output retrieval queries in all four categories: keyword, natural, hyde, and question. Try to output exactly this many queries per category: keyword = {QUERY_REWRITER_SECTION_TO_MAX_QUERIES['keyword']}, natural = {QUERY_REWRITER_SECTION_TO_MAX_QUERIES['natural']}, hyde = {QUERY_REWRITER_SECTION_TO_MAX_QUERIES['hyde']}, question = {QUERY_REWRITER_SECTION_TO_MAX_QUERIES['question']}. If it is genuinely difficult to reach the target for a category without producing low-quality duplicates, you may output fewer for that category. Do not exceed these counts, and do not skip a category just because you think another one is better.\n"
            f"- Also output exactly one reranker query inside {RERANKER_QUERY_OPENING_TAG} and {RERANKER_QUERY_CLOSING_TAG} tags.\n"
            "- The reranker query should be a single clean rewrite of the current request in the email. It should read like a direct request or concise email-style summary of what is currently being asked, not like a keyword list and not like a label such as 'Current request: ...'.\n"
            "- For the reranker query, remove greetings, signatures, and stale quoted or forwarded material when they are not needed, but keep the background information that is necessary to understand the current request.\n"
            "- HARD RULE FOR ALL OUTPUTS, INCLUDING THE RERANKER QUERY: NEVER write student or applicant names, student IDs, national IDs, passport or NIE numbers, application numbers, enrollment numbers, bank details, payment references, or any other personally identifiable information. If those details appear in the email, replace them with a generic role such as 'student', 'applicant', or 'Erasmus student'. The only person names that may appear are university-side names such as professor, coordinator, or staff names, and only when the email is explicitly about that person as a retrieval target. If a generated query contains student-side personal data, that query is WRONG and must be rewritten.\n\n"
            "### OUTPUT FORMAT:\n"
            "Output either the no-request tag or the retrieval queries and reranker query, never both.\n"
            f"If the email is no-request, output only {NO_REQUEST_OPENING_TAG}{NO_REQUEST_CLOSING_TAG} and stop.\n"
            "Otherwise output only the retrieval queries and reranker query inside these tags.\n"
            f"{QUERIES_OPENING_TAG}\n"
            f"{KEYWORD_QUERIES_OPENING_TAG}\n"
            f"{QUERY_OPENING_TAG}keyword query here{QUERY_CLOSING_TAG}\n"
            f"{KEYWORD_QUERIES_CLOSING_TAG}\n"
            f"{NATURAL_QUERIES_OPENING_TAG}\n"
            f"{QUERY_OPENING_TAG}natural query here{QUERY_CLOSING_TAG}\n"
            f"{NATURAL_QUERIES_CLOSING_TAG}\n"
            f"{HYDE_QUERIES_OPENING_TAG}\n"
            f"{QUERY_OPENING_TAG}hypothetical answer or document snippet here{QUERY_CLOSING_TAG}\n"
            f"{HYDE_QUERIES_CLOSING_TAG}\n"
            f"{QUESTION_QUERIES_OPENING_TAG}\n"
            f"{QUERY_OPENING_TAG}question query here{QUERY_CLOSING_TAG}\n"
            f"{QUESTION_QUERIES_CLOSING_TAG}\n"
            f"{QUERIES_CLOSING_TAG}\n"
            f"{RERANKER_QUERY_OPENING_TAG}clean reranker query here{RERANKER_QUERY_CLOSING_TAG}\n\n"
            "### NO-REQUEST EXAMPLE:\n"
            "Email subject:\n"
            "---\n"
            "Enrollment procedure\n"
            "---\n"
            "Email body:\n"
            "---\n"
            "Thank you very much. Understood. Have a good day.\n"
            "---\n"
            "Output:\n"
            f"{NO_REQUEST_OPENING_TAG}{NO_REQUEST_CLOSING_TAG}\n\n"
            "### REQUEST EXAMPLE:\n"
            "Email subject:\n"
            "---\n"
            "Possible transfer and collaboration with the University Paul Sabatier of Toulouse\n"
            "---\n"
            "Email body:\n"
            "---\n"
            "Buenos días,\n"
            "me llamo Laura Pérez y soy estudiante del Máster en Ciencia de Datos de la UPM. Mi número de matrícula es 222222.\n"
            "Me gustaría cambiarme a vuestro máster sin terminar este y, si fuera posible, hacerlo además en colaboración con la Universidad Paul Sabatier de Toulouse.\n"
            "¿Es posible y qué oficina o procedimiento tendría que seguir?\n"
            "---\n"
            "Output:\n"
            f"{QUERIES_OPENING_TAG}\n"
            f"{KEYWORD_QUERIES_OPENING_TAG}\n"
            f"{QUERY_OPENING_TAG}Data Science master transfer{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}switch masters from Data Science{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}master transfer admission pathway{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}external university mobility procedure{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}mobility agreement Paul Sabatier{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Paul Sabatier Toulouse collaboration{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}master transfer plus mobility procedure{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}master transfer mobility combination{QUERY_CLOSING_TAG}\n"
            f"{KEYWORD_QUERIES_CLOSING_TAG}\n"
            f"{NATURAL_QUERIES_OPENING_TAG}\n"
            f"{QUERY_OPENING_TAG}Master's transfer procedure from the Master in Data Science{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Transfer requirements for a student in the Master in Data Science{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Office responsible for master's transfer requests{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Mobility or collaboration options with the University Paul Sabatier of Toulouse{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Procedure to request mobility with an external university{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Contact point for mobility applications involving the University Paul Sabatier of Toulouse{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Whether a student in the Master in Data Science can transfer and also request mobility with the University Paul Sabatier of Toulouse{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Administrative procedure for combining a master's transfer with external university mobility{QUERY_CLOSING_TAG}\n"
            f"{NATURAL_QUERIES_CLOSING_TAG}\n"
            f"{HYDE_QUERIES_OPENING_TAG}\n"
            f"{QUERY_OPENING_TAG}Transfers from the Master in Data Science are handled through the standard transfer or admission procedure{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}The responsible office for transfer requests reviews applications from students changing masters{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Students leaving the Master in Data Science must follow the standard enrollment pathway{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Mobility with the University Paul Sabatier of Toulouse is handled through the corresponding agreement and application procedure{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}External university mobility requests are handled through a defined office and procedure rather than through student-specific case tracking{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}The mobility program includes the University Paul Sabatier of Toulouse as a destination university{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}A master's transfer and external university mobility are processed as separate administrative steps{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Transfer requirements and mobility requirements are listed in separate procedures{QUERY_CLOSING_TAG}\n"
            f"{HYDE_QUERIES_CLOSING_TAG}\n"
            f"{QUESTION_QUERIES_OPENING_TAG}\n"
            f"{QUERY_OPENING_TAG}How can I transfer from the Master in Data Science?{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}What are the transfer requirements for a student in the Master in Data Science?{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Which office handles transfer requests?{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Is there a mobility or collaboration option with the University Paul Sabatier of Toulouse?{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}What is the procedure to request mobility with an external university?{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Which office handles mobility or collaboration requests with external universities?{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Can a student in the Master in Data Science transfer and also request mobility with the University Paul Sabatier of Toulouse?{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}What administrative steps combine a master's transfer with external university mobility?{QUERY_CLOSING_TAG}\n"
            f"{QUESTION_QUERIES_CLOSING_TAG}\n"
            f"{QUERIES_CLOSING_TAG}\n"
            f"{RERANKER_QUERY_OPENING_TAG}I need information about transferring from the Master in Data Science and about whether there is a mobility or collaboration option with the University Paul Sabatier of Toulouse, including the responsible office and procedure.{RERANKER_QUERY_CLOSING_TAG}\n\n"
            "Email subject:\n"
            "---\n"
            "{subject}\n"
            "---\n"
            "Email body:\n"
            "---\n"
            "{body}\n"
            "---\n"
            "Output:\n"
        ),
    },
    LLM_JUDGE_PROFILE: {
        **LLM_JUDGE_SETTINGS,
        "system_prompt": (
            "You are an expert answerability judge for retrieval."
        ),
        "prompt_template": (
            "You will receive one user query and several knowledge-base chunks, each with an explicit chunk ID.\n"
            "Your task is to decide whether the query is answerable using only the provided chunks.\n\n"
            "### CONTEXT:\n"
            "The query is an anonymized rewrite of an email. "
            "Student names, IDs, grades, internal case history, and other personal details may have been removed or generalized. "
            "The email is about the MUIA master's program, which is an official master's program taught by the Department of Artificial Intelligence at the School of Computer Engineering of the Universidad Politécnica de Madrid (UPM).\n"
            "- Evidence can appear at different levels. A chunk specifically about MUIA is the strongest kind of evidence. If there is no more specific MUIA chunk, a general UPM administrative rule can still be treated as applicable to MUIA when it clearly applies to UPM official master's programs or to UPM students. This is especially relevant for matters such as matrícula, cancellation, modification, deadlines, documentation, SEPA, and other general administrative procedures. Do not transfer rules from another named master's program, another degree, or another program to MUIA. If both a general UPM rule and a more specific MUIA rule are present, prefer the MUIA rule.\n"
            "- Judge answerability at the level of the exact public issue being asked, such as a policy, procedure, requirement, limit, approval rule, eligibility criterion, responsible office, or document type. A single query may contain more than one sub-request, and the final label must reflect the whole anonymized request rather than only one part of it. A subquery is a separable public question that could reasonably require different evidence or receive a different answerability label. This can happen even when the overall topic is the same. For example, 'How do I access Moodle?' and 'Where do I find the class schedule with room locations?' should usually be treated as two subqueries because they ask for different information and may rely on different chunks. First identify whether the query is really a single issue or whether it contains multiple subqueries that should be judged separately. Do not downgrade the label just because student-specific identity, private case state, or internal follow-up details were anonymized away.\n"
            "- The emails used for evaluation may be several years old, while the knowledge base is crawled later and refreshed yearly. The goal is to judge whether the current knowledge base contains public information that would be useful to answer the type of request, not whether it reproduces the exact historical year or date mentioned in the original case.\n"
            "- This judgment is part of an evaluation pipeline used to understand where performance succeeds or fails across the system, including knowledge-base coverage, retrieval and encoding quality, and later training of discriminator-style models or filtered subsets for encoder training. Because of that, focus on whether the chunks contain transferable public information that would help answer the request type, not on whether they exactly mirror the original historical case.\n"
            "- Treat years, exact calendar dates, student names, and exact external subject titles as incidental case details by default. Do not lower answerability just because those exact details are not repeated in the chunks. If an email mentions something like 2022, a specific day, or a named student only because it comes from the old case description, that does not by itself require a chunk tied to that same year, date, or person. Judge whether the chunks provide the governing rule, limit, procedure, requirement, or responsible office for the issue. A later or undated rule, procedure, limit, or office can still count as useful support when it addresses the same issue.\n"
            "- Generic topic overlap is not enough, and a chunk can still deserve label -1 if it is on the right topic but does not help answer the exact public issue.\n"
            "Example: if the anonymized query is 'Can I recognize 60 credits from another university for these subjects?', the chunks do not need to mention the exact student or decide whether those exact subjects will be recognized. "
            "The important public issue is the recognition rule or credit limit. "
            "If the query is only asking whether that recognition is possible, chunks that only explain the credit-recognition procedure without the relevant limit or rule should usually receive label -1. "
            "A procedure-only chunk should move that example to label 0 only if the anonymized query is also explicitly asking how to carry out the request.\n\n"
            "### OUTPUT RULES:\n"
            "- Subqueries: The query may contain one or more subqueries. Output one subquery block for each subquery present in the query. If the query is really about a single issue, output exactly one subquery block. Each subquery block must contain the subquery text, the subquery answerability label, the subquery confidence, the subquery supporting chunk IDs, the subquery insufficient chunk IDs, and the subquery rationale.\n"
            f"- Subquery answerability: At the subquery level, use label 1 when that subquery is fully answerable, label 0 when that subquery is only partially answerable because an important public piece is still missing, and label -1 when that subquery does not have meaningful support. For subquery labels 1 and 0, supporting chunk IDs may be used, with **at most {LLM_JUDGE_MAX_SUPPORTING_CHUNK_IDS} IDs**. Put a chunk in supporting chunk IDs only if it contributes usable information toward answering that same subquery, even if it is not enough for a full answer. Supporting chunk IDs must be ordered from strongest support to weakest support. If there are only a few strong supporting chunks, return only those few strong chunks. Fewer strong items are better than padding the list with weak or borderline positives. If a chunk is only loosely related, a near miss, or not clearly usable for answering the subquery, do not place it in supporting chunk IDs. For subquery label -1, the supporting chunk ID list must be empty. Insufficient chunk IDs may be used for any subquery label, with **at most {LLM_JUDGE_MAX_INSUFFICIENT_CHUNK_IDS} IDs**. Put a chunk in insufficient chunk IDs only if it is topically related but does not contribute usable information toward answering that same subquery, so it would not by itself justify label 0. You may use insufficient chunk IDs to show near-miss chunks that are thematically related but still do not answer the subquery. Insufficient chunk IDs must be ordered from closest miss to farthest miss. Again, do not fill the list just because capacity remains: fewer strong near misses are better than weak items. The same chunk ID must never appear in both supporting chunk IDs and insufficient chunk IDs for the same subquery. Subquery confidence must be a decimal number between 0.0 and 1.0 with exactly one decimal place.\n"
            "- Top-level answerability: The top-level answerability is the summary of the subquery answerability. If there is only one subquery, the top-level answerability must match that subquery label. If every subquery receives label 1, the top-level answerability must be 1. If at least one subquery receives label 0 or 1 but the full query is not fully answerable, the top-level answerability must be 0. If every subquery receives label -1, the top-level answerability must be -1.\n"
            "- Draft answer: If the top-level answerability label is 1 or 0, the draft answer is mandatory and must not be empty. It must be a brief reply of 1 to 2 short sentences that directly answers the query and sounds like a real email reply to the user. Write it as if you were replying from the MUIA side, using the role and account of the master's coordination/secretariat at masteria.dia@fi.upm.es. If the relevant office is that same MUIA coordination/secretariat side, answer directly instead of telling the student to contact that same email address. You may use facts contained in the chunks, but do not refer to the retrieval or judging process itself. Do not say things like 'the chunks say', 'the provided information indicates', 'the available information shows', or 'the retrieved evidence suggests'. When the chunks provide a concrete rule, limit, requirement, deadline, or office, state that concrete information explicitly instead of giving a vague procedural reply. If the top-level answerability label is -1, leave the draft answer empty.\n"
            "- Subquery rationale: Each subquery must have its own rationale. For subquery label 0, it is mandatory and must briefly say what useful evidence was found and what important public information is still missing for that same subquery. For subquery label -1, it is mandatory and must say whether no relevant topic was found at all, or whether related chunks were found but they did not answer the exact public issue, and briefly explain why they were not enough for that same subquery. For subquery label 1, the rationale is optional and should usually be omitted unless the case is borderline.\n"
            "- General constraints: Use only the provided chunks. Do not use outside knowledge. Base the decision on whether the query can actually be answered, not on keyword overlap or broad topic overlap. Output only the requested tags.\n\n"
            "### OUTPUT EXAMPLE:\n"
            f"{SUBQUERIES_OPENING_TAG}\n"
            f"{SUBQUERY_OPENING_TAG}\n"
            f"{SUBQUERY_TEXT_OPENING_TAG}What is the maximum recognition limit?{SUBQUERY_TEXT_CLOSING_TAG}\n"
            f"{SUBQUERY_ANSWERABILITY_OPENING_TAG}1{SUBQUERY_ANSWERABILITY_CLOSING_TAG}\n"
            f"{SUBQUERY_CONFIDENCE_OPENING_TAG}0.9{SUBQUERY_CONFIDENCE_CLOSING_TAG}\n"
            f"{SUBQUERY_SUPPORTING_CHUNK_IDS_OPENING_TAG}\n"
            f"{CHUNK_ID_OPENING_TAG}3{CHUNK_ID_CLOSING_TAG}\n"
            f"{CHUNK_ID_OPENING_TAG}8{CHUNK_ID_CLOSING_TAG}\n"
            f"{SUBQUERY_SUPPORTING_CHUNK_IDS_CLOSING_TAG}\n"
            f"{SUBQUERY_INSUFFICIENT_CHUNK_IDS_OPENING_TAG}\n"
            f"{SUBQUERY_INSUFFICIENT_CHUNK_IDS_CLOSING_TAG}\n"
            f"{SUBQUERY_RATIONALE_OPENING_TAG}{SUBQUERY_RATIONALE_CLOSING_TAG}\n"
            f"{SUBQUERY_CLOSING_TAG}\n"
            f"{SUBQUERY_OPENING_TAG}\n"
            f"{SUBQUERY_TEXT_OPENING_TAG}Which office or form is required for the request?{SUBQUERY_TEXT_CLOSING_TAG}\n"
            f"{SUBQUERY_ANSWERABILITY_OPENING_TAG}-1{SUBQUERY_ANSWERABILITY_CLOSING_TAG}\n"
            f"{SUBQUERY_CONFIDENCE_OPENING_TAG}0.8{SUBQUERY_CONFIDENCE_CLOSING_TAG}\n"
            f"{SUBQUERY_SUPPORTING_CHUNK_IDS_OPENING_TAG}\n"
            f"{SUBQUERY_SUPPORTING_CHUNK_IDS_CLOSING_TAG}\n"
            f"{SUBQUERY_INSUFFICIENT_CHUNK_IDS_OPENING_TAG}\n"
            f"{CHUNK_ID_OPENING_TAG}12{CHUNK_ID_CLOSING_TAG}\n"
            f"{SUBQUERY_INSUFFICIENT_CHUNK_IDS_CLOSING_TAG}\n"
            f"{SUBQUERY_RATIONALE_OPENING_TAG}The chunks provide the recognition limit, but they do not identify the exact office, form, or step-by-step procedure needed to answer this subquery.{SUBQUERY_RATIONALE_CLOSING_TAG}\n"
            f"{SUBQUERY_CLOSING_TAG}\n"
            f"{SUBQUERIES_CLOSING_TAG}\n"
            f"{ANSWERABILITY_OPENING_TAG}0{ANSWERABILITY_CLOSING_TAG}\n"
            f"{DRAFT_ANSWER_OPENING_TAG}The maximum recognition is 50% of the credits of the master's program. However, I cannot confirm the exact office or form needed for this request.{DRAFT_ANSWER_CLOSING_TAG}\n"
            "\n"
            "### INPUT:\n"
            "Query:\n"
            "{query}\n\n"
            "Candidate Chunks:\n"
            "{chunks}\n\n"
            "Output:\n"
        ),
    }
}
