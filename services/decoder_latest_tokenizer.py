from config.general import modal_secret
from config.modal_apps import DECODER_LATEST_TOKENIZER_APP_NAME
from config.decoder_latest_tokenizer import (
    image,
    MODAL_TIMEOUT,
    SCALEDOWN_WINDOW,
    MIN_CONTAINERS,
)
import modal

# Modal
app = modal.App(DECODER_LATEST_TOKENIZER_APP_NAME)

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
)
def count_decoder_latest_tokens(texts, model_name_or_path):
    from helpers.decoder import count_text_tokens

    return count_text_tokens(texts, model_name_or_path)

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
)
def truncate_decoder_latest_texts(texts, max_tokens, model_name_or_path):
    from helpers.decoder import truncate_texts_to_tokens

    return truncate_texts_to_tokens(texts, max_tokens, model_name_or_path)
