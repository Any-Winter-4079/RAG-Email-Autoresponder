from config.general import modal_secret
from config.modal_apps import DECODER_LATEST_APP_NAME
from config.decoder import MIN_CONTAINERS, SCALEDOWN_WINDOW, EMAIL_WRITER_PROFILE
from config.decoder_latest import image, GPU, TIMEOUT
from helpers.decoder import run_local_lm_or_vlm
import modal

# Modal
app = modal.App(DECODER_LATEST_APP_NAME)

@app.function(
        image=image,
        gpu=GPU,
        secrets=[modal_secret],
        timeout=TIMEOUT,
        scaledown_window=SCALEDOWN_WINDOW,
        min_containers=MIN_CONTAINERS
        )
def run_local_lm_or_vlm_latest(
    context,
    current_turn_input_text,
    provider,
    model_name_or_path,
    is_vision_model,
    current_turn_image_in_bytes,
    system_prompt,
    max_new_tokens,
    temperature,
    top_p,
    top_k,
    use_flash_attention_2,
    enable_thinking,
    model_family=None,
    return_prompt_text=False,
    decoder_profile=EMAIL_WRITER_PROFILE
    ):
    return run_local_lm_or_vlm(
        context=context,
        current_turn_input_text=current_turn_input_text,
        provider=provider,
        model_name_or_path=model_name_or_path,
        is_vision_model=is_vision_model,
        current_turn_image_in_bytes=current_turn_image_in_bytes,
        system_prompt=system_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
        use_flash_attention_2=use_flash_attention_2,
        enable_thinking=enable_thinking,
        model_family=model_family,
        return_prompt_text=return_prompt_text,
        decoder_profile=decoder_profile,
    )
