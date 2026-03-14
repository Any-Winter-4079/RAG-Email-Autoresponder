import modal
from config.crawler_agent import ALLOWED_URL_HOST_TO_CATEGORY

COMMON_PACKAGES = [
    "torchvision",
    "transformers==4.57.0",
    "accelerate",
    "peft==0.17.1",
    "Pillow",
    "requests",
    "hf_transfer",
]
GPU = "L40S"
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
QUESTION_QUERIES_OPENING_TAG = "<questionqueries>"
QUESTION_QUERIES_CLOSING_TAG = "</questionqueries>"
QUERY_REWRITER_MAX_QUERIES = 18
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

THREAD_GROUPER_MAX_EMAILS = 20
EMAIL_WRITER_PROFILE = "email_writer"
THREAD_GROUPER_PROFILE = "thread_grouper"
DATA_CLEANER_PROFILE = "data_cleaner"
QUERY_REWRITER_PROFILE = "query_rewriter"
LLM_JUDGE_PROFILE = "llm_judge"

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
        "provider": "local",
        "model_name_or_path": "Qwen/Qwen3-8B-FP8",
        "enable_thinking": True,
        "is_vision_model": False,
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
        "max_context_tokens": 32768,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
        "return_prompt_text": True
    },
    THREAD_GROUPER_PROFILE: {
        "provider": "local",
        "model_name_or_path": "Qwen/Qwen3-8B-FP8",
        "enable_thinking": True,
        "is_vision_model": False,
        "system_prompt": (
            "You are an expert email thread reconstruction assistant."
        ),
        "max_context_tokens": 32768,
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
        "production_example": (
            "Input emails:\n"
            "Inbox:\n"
            f"{{'id': b'440', 'threadID': 1, 'from': '{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> undefined', 'to': '{EXAMPLE_PROF1_EMAIL}, \"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined, {EXAMPLE_PROF2_EMAIL} undefined undefined undefined', 'date': datetime.datetime(2020, 5, 4, 9, 8, 59, tzinfo=datetime.timezone.utc), 'subject': 'Erasmus', 'body': \"Good morning. I am a student from Italy currently studying at my university. I would like to join the Erasmus program at Universidad Politécnica de Madrid. Would it be possible to take courses from both the MSc in Artificial Intelligence and the Máster Universitario en Automática y Robótica, or should I choose just one? Thank you\"}}\n"
            f"{{'id': b'448', 'threadID': 3, 'from': '{EXAMPLE_STAFF_NAME} <{EXAMPLE_STAFF_EMAIL}>', 'to': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'date': datetime.datetime(2020, 6, 11, 13, 32, 15, tzinfo=datetime.timezone.utc), 'subject': 'Fwd: Fichero egresados', 'body': \"Hola, {DIRECTOR_NAME}. Te reenvío aquí lo último que hice yo. Es de mayo de 2020. No sé si {EXAMPLE_COLLEAGUE_NAME} hizo alguno posterior. ¿Te sirve? {EXAMPLE_STAFF_NAME}\"}}\n"
            f"{{'id': b'432', 'threadID': 4, 'from': '{EXAMPLE_STUDENT_REP_NAME} <{EXAMPLE_STUDENT_REP_EMAIL}>', 'to': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'date': datetime.datetime(2020, 6, 12, 8, 55, 27, tzinfo=datetime.timezone.utc), 'subject': 'Orla/graduación de alumnos del máster en IA', 'body': \"Hola {DIRECTOR_NAME}, como delegado del máster en Inteligencia Artificial, me gustaría trasladarte la consulta de varios alumnos acerca de si se va a hacer orla / acto de graduación para los estudiantes del máster, o por si el contrario corre bajo nuestra cuenta hacerlo. ¡Un saludo! {EXAMPLE_STUDENT_REP_NAME} Máster Universitario en Inteligencia Artificial\"}}\n"
            "Sent:\n"
            f"{{'id': b'441', 'threadID': 1, 'from': '\"[MUIA] {DIRECTOR_NAME}\" <{DIRECTOR_EMAIL}> undefined', 'to': '{EXAMPLE_STUDENT_NAME} <{EXAMPLE_STUDENT_EMAIL}> undefined', 'date': datetime.datetime(2020, 5, 4, 11, 16, 0, tzinfo=datetime.timezone.utc), 'subject': 'Erasmus', 'body': \"I don't know about the other Degree; it is probably taught in another School, so that it can be tricky for you to attend both. Besides, I'm not sure you can follow courses from different School at the same time during your Erasmus. Please ask orex@fi.upm.es: they are in charge of the Erasmus Programme at Escuela Técnica Superior de Ingenieros Informáticos. regards -- {DIRECTOR_NAME} Coordinador del Máster Universitario en Inteligencia Artificial Universidad Politécnica de Madrid Escuela Técnica Superior de Ingenieros Informáticos Campus de Montegancedo S/N, 28660, Boadilla del Monte, Madrid SPAIN Departamento de Inteligencia Artificial {DEPARTMENT_PHONE} {DIRECTOR_EMAIL}\"}}\n"
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
            f"{THREAD_SUBJECT_OPENING_TAG}Erasmus{THREAD_SUBJECT_CLOSING_TAG}\n"
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
        "prompt_template": (
            "{task_description_start}\n\n"
            "### RULES:\n"
            "1. Preserve chronological order within each thread.\n"
            "2. Remove quoted text only when the quoted content appears elsewhere in the input as the same text "
            "with fewer or no quote markers. Keep the least-quoted instance.\n"
            "   - Example: if B contains \"> A\" and A appears elsewhere unquoted, remove \"> A\" from B.\n"
            "   - Example: if B contains only \"> A\" and A does not appear elsewhere, keep it.\n"
            "   - Example: if B has \"> A\" and C has \"> B\\n> A\", keep \"> A\" in B and remove the quoted part from C.\n"
            "   Reply headers such as 'En ... escribió:', 'On ... wrote:', 'De/Enviado/Para/Asunto' are a few indicators of quoted blocks.\n"
            f"3. Keep only the cleaned body inside {THREAD_BODY_OPENING_TAG}...{THREAD_BODY_CLOSING_TAG} tags.\n"
            "4. Do not hallucinate or add new messages.\n"
            "5. Output ONLY the thread XML, nothing else.\n\n"
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
            "Inbox:\n"
            "{inbox_emails}\n"
            "Sent:\n"
            "{sent_emails}\n"
            "Output:\n"
        ),
        "max_new_tokens": 8192,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
        "return_prompt_text": True
    },
    DATA_CLEANER_PROFILE: {
        "provider": "local", # local or openai
        "model_name_or_path": "Qwen/Qwen3-8B-FP8", # gpt-5-mini or Qwen/Qwen3-8B-FP8
        "enable_thinking": True,
        "reasoning_effort": "minimal",
        "is_vision_model": False,
        "system_prompt": (
            "You are an expert Knowledge Curator for RAG."
        ),
        "prompt_template": (
            "Your job is to take source texts and convert them into four outputs: cleaned text, abstract, summary, and Q&A pairs.\n"
            "To help you do this well, the following instructions explain the wider product that this knowledge-curation step supports.\n\n"
            "This data will be used by a retrieval system for current and prospective students interested in MUIA (Master's Degree in Artificial Intelligence) at UPM, coordinated by the Department of Artificial Intelligence (DIA) at FI-UPM.\n"
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
            "1. You will receive 'Page History' containing abstracts and summaries from previous chunks. Each entry includes its **Chunk Index** so you can determine its position in the document. **Note**: This history may be non-consecutive (e.g., Chunks 0-4 to provide you with beginning-of-document context followed by Chunks 35-40 to provide you with latest-chunks context).\n"
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
        "max_chunk_size": 1024,
        "max_new_tokens": 8192,
        "temperature": 0.1,
        "top_p": 0.8,
        "top_k": 20,
        "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
        "return_prompt_text": True
    },
    QUERY_REWRITER_PROFILE: {
        "provider": "local",
        "model_name_or_path": "Qwen/Qwen3-8B-FP8",
        "enable_thinking": True,
        "is_vision_model": False,
        "system_prompt": (
            "You are an expert at generating retrieval queries."
        ),
        "prompt_template": (
            f"Your task is to generate up to {QUERY_REWRITER_MAX_QUERIES} retrieval queries for the following email.\n"
            "The goal is to maximize the chances of retrieving similar and useful chunks from a knowledge base.\n\n"
            "### KNOWLEDGE BASE CONTEXT:\n"
            "The knowledge base was built by crawling webpages mainly related to the Master Universitario en Inteligencia Artificial, "
            "the Department of Artificial Intelligence, the Facultad or Escuela that hosts the program, and Universidad Politecnica de Madrid. "
            "Those webpages were chunked. Another language model then cleaned the chunks, translated most content into English, generated summaries, "
            "and generated question-answer pairs. Because of this pipeline, the knowledge base may contain English cleaned text, English summaries with explicit entities, "
            "question-like text, and occasional mixed English and Spanish entity names.\n\n"
            "Retrieval uses several encoders, including sparse frequency-based encoders and dense encoders. Long emails often contain many words that are not useful for retrieval. "
            "Because of this, generate a mix of query styles: some very short keyword-focused queries for sparse frequency-based encoders, some clear natural-language queries for dense encoders, "
            "and some queries written as questions to match question-answer pairs in the knowledge base. Write the queries in English, because that is the main language of the knowledge base. "
            "However, because the cleaner model sometimes kept entity names in Spanish instead of translating them, generate additional English queries where only the relevant entity remains in Spanish while the rest of the query stays in English.\n\n"
            "### CURRENT SUBJECT AND SEMINAR CODE MAPPING:\n"
            "The following is the current best mapping we have between subject and seminar codes and subject and seminar names in Spanish and English. "
            "We provide this mapping because people usually mention the subject or seminar name in an email, not the code, but the cleaner model sometimes kept the Spanish name, "
            "sometimes translated the name into English, and sometimes replaced the name with the code. If the email is not about a concrete subject or seminar that appears in this mapping, ignore this mapping completely. "
            "Do not use it for overall program-level mentions such as a master's program name, and do not use it just because some words overlap with a subject or seminar name. "
            "If the email explicitly mentions a concrete subject or seminar and the mapping suggests a code, generate one English query with the name in English, one English query where that entity remains in Spanish, and one English query that uses the code instead of the subject or seminar name.\n"
            f"{OLD_MUIA_SUBJECT_AND_SEMINAR_CODE_MAPPING_TEXT}\n\n"
            "### WHAT TO DO:\n"
            "1. Prioritize the main administrative or academic request in the email.\n"
            "2. If the email contains multiple distinct requests, generate queries for the different requests.\n"
            "3. Remove greetings, signatures, politeness filler, and irrelevant conversational text.\n"
            "4. Queries are used as-is against a knowledge base that can contain other programs, departments, and university entities. Do not use vague references such as 'this master's', 'the program', or 'that subject'. Name the concrete program, degree, subject, seminar, faculty, department, procedure, URL, date, number, or person when relevant.\n"
            "5. For each query, include only the entities that help identify the target content. Do not include an entity just because it appears in the email.\n"
            "6. Make the queries specific and self-contained.\n"
            f"7. Output up to {QUERY_REWRITER_MAX_QUERIES} queries.\n\n"
            "### OUTPUT FORMAT:\n"
            f"Output only the queries inside these tags. Use all three sections, and put one or more query tags inside each section, totalling up to {QUERY_REWRITER_MAX_QUERIES}.\n"
            f"{QUERIES_OPENING_TAG}\n"
            f"{KEYWORD_QUERIES_OPENING_TAG}\n"
            f"{QUERY_OPENING_TAG}keyword query here{QUERY_CLOSING_TAG}\n"
            f"{KEYWORD_QUERIES_CLOSING_TAG}\n"
            f"{NATURAL_QUERIES_OPENING_TAG}\n"
            f"{QUERY_OPENING_TAG}natural query here{QUERY_CLOSING_TAG}\n"
            f"{NATURAL_QUERIES_CLOSING_TAG}\n"
            f"{QUESTION_QUERIES_OPENING_TAG}\n"
            f"{QUERY_OPENING_TAG}question query here{QUERY_CLOSING_TAG}\n"
            f"{QUESTION_QUERIES_CLOSING_TAG}\n"
            f"{QUERIES_CLOSING_TAG}\n\n"
            "### EXAMPLE:\n"
            "Email:\n"
            "---\n"
            "Hola, me gustaria saber que conocimientos previos necesito para Redes bayesianas y si hay examen final.\n"
            "---\n"
            "Output:\n"
            f"{QUERIES_OPENING_TAG}\n"
            f"{KEYWORD_QUERIES_OPENING_TAG}\n"
            f"{QUERY_OPENING_TAG}A4 prerequisites{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Redes bayesianas final exam{QUERY_CLOSING_TAG}\n"
            f"{KEYWORD_QUERIES_CLOSING_TAG}\n"
            f"{NATURAL_QUERIES_OPENING_TAG}\n"
            f"{QUERY_OPENING_TAG}Bayesian Networks prerequisites in the Master in Artificial Intelligence{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Final exam for Redes bayesianas in the Master in Artificial Intelligence{QUERY_CLOSING_TAG}\n"
            f"{NATURAL_QUERIES_CLOSING_TAG}\n"
            f"{QUESTION_QUERIES_OPENING_TAG}\n"
            f"{QUERY_OPENING_TAG}What prior knowledge is required for Bayesian Networks A4?{QUERY_CLOSING_TAG}\n"
            f"{QUERY_OPENING_TAG}Does Redes bayesianas A4 have a final exam?{QUERY_CLOSING_TAG}\n"
            f"{QUESTION_QUERIES_CLOSING_TAG}\n"
            f"{QUERIES_CLOSING_TAG}\n\n"
            "Email:\n"
            "---\n"
            "{text}\n"
            "---\n"
            "Output:\n"
        ),
        "max_new_tokens": 8192,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
        "return_prompt_text": False
    },
    LLM_JUDGE_PROFILE: {
        "provider": "local",
        "model_name_or_path": "Qwen/Qwen3-8B-FP8",
        "enable_thinking": True,
        "is_vision_model": False,
        "system_prompt": (
            "You are an expert retrieval relevance judge."
        ),
        "prompt_template": (
            "You will receive one user query and several candidate chunks.\n"
            "Your task is to score how useful each candidate chunk is for answering the user query.\n\n"
            "### SCORING CRITERIA:\n"
            "Score each candidate chunk between 0.0 and 1.0 based on how useful it is for answering the query.\n"
            "- 0.0: unrelated or useless\n"
            "- 0.25: weak relevance\n"
            "- 0.5: partially useful\n"
            "- 0.75: strongly relevant\n"
            "- 1.0: directly answers the query or provides the key missing facts\n\n"
            "### SCORING RULES:\n"
            "1. Return one score for each candidate chunk, in the same order as the input.\n"
            "2. Each score must be a decimal number between 0.0 and 1.0.\n"
            "3. Use 0.0 for no relevance, 1.0 for highly useful direct relevance, and intermediate values for partial usefulness.\n"
            "4. Score usefulness for answering the query, not just keyword overlap.\n"
            "5. Output only the scores inside the required tags.\n\n"
            "### OUTPUT FORMAT:\n"
            f"{SCORES_OPENING_TAG}\n"
            f"{SCORE_OPENING_TAG}0.0{SCORE_CLOSING_TAG}\n"
            f"{SCORE_OPENING_TAG}0.5{SCORE_CLOSING_TAG}\n"
            f"{SCORE_OPENING_TAG}1.0{SCORE_CLOSING_TAG}\n"
            f"{SCORES_CLOSING_TAG}\n\n"
            "### INPUT:\n"
            "Query:\n"
            "{query}\n\n"
            "Candidate Chunks:\n"
            "{chunks}\n\n"
            "Output:\n"
        ),
        "max_new_tokens": 2048,
        "temperature": 0.3,
        "top_p": 0.8,
        "top_k": 20,
        "use_flash_attention_2": USE_FLASH_ATTENTION_IMAGE,
        "return_prompt_text": False
    }
}

_image_flash_attention_base = (
    modal.Image.from_registry(FLASH_ATTENTION_IMAGE)
    .run_commands(FLASH_ATTENTION_RUN_COMMANDS)
    .pip_install(
        FLASH_ATTENTION_TORCH_VERSION,
        *COMMON_PACKAGES
        )
    .pip_install(FLASH_ATTENTION_RELEASE)
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
_image_no_flash_attention_base = (
    modal.Image.debian_slim(python_version=NO_FLASH_ATTENTION_PYTHON_VERSION)
    .pip_install(
        NO_FLASH_ATTENTION_TORCH_VERSION,
        *COMMON_PACKAGES
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)
base_image = _image_flash_attention_base if USE_FLASH_ATTENTION_IMAGE else _image_no_flash_attention_base
# Run `image.add_local_*` commands last in your image build to avoid rebuilding images with every local file change.
image = base_image.add_local_python_source("config", "helpers")
