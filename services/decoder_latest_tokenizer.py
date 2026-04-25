from functools import lru_cache

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

@lru_cache(maxsize=4)
def get_tokenizer(model_name_or_path):
    from transformers import AutoProcessor, AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
    except Exception:
        processor = AutoProcessor.from_pretrained(
            model_name_or_path,
            trust_remote_code=True,
        )
        return processor.tokenizer

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
)
def count_decoder_latest_tokens(texts, model_name_or_path):
    tokenizer = get_tokenizer(model_name_or_path)
    return [
        len(tokenizer.encode(text, add_special_tokens=False))
        for text in texts
    ]
