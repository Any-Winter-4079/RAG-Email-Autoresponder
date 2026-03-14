from config.decoder import MODEL_PROFILES, LLM_JUDGE_PROFILE
from config.modal_apps import DECODER_APP_NAME
from config.modal_functions import RUN_QWEN3_LM_OR_VLM_FUNCTION_NAME
import modal

def judge_chunks(query, chunks):
    run_qwen3_lm_or_vlm = modal.Function.from_name(
        DECODER_APP_NAME,
        RUN_QWEN3_LM_OR_VLM_FUNCTION_NAME,
    )

    judge_profile_config = MODEL_PROFILES[LLM_JUDGE_PROFILE].copy()
    prompt_template = judge_profile_config.pop("prompt_template")
    formatted_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        formatted_chunks.append(f"-----\n[Chunk {idx}]\n{chunk}\n-----")
    prompt = prompt_template.format(
        query=query,
        chunks="\n".join(formatted_chunks),
    )
    scores, _ = run_qwen3_lm_or_vlm.remote(
        context=[],
        current_turn_input_text=prompt,
        current_turn_image_in_bytes=None,
        **judge_profile_config,
        decoder_profile=LLM_JUDGE_PROFILE
    )
    return scores
