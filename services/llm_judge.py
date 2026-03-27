from config.general import modal_secret
from config.modal_apps import LLM_JUDGE_APP_NAME, DECODER_APP_NAME
from config.modal_functions import RUN_QWEN3_LM_OR_VLM_FUNCTION_NAME
from config.decoder import MODEL_PROFILES, LLM_JUDGE_PROFILE
from config.llm_judge import image, MODAL_TIMEOUT, SCALEDOWN_WINDOW, MIN_CONTAINERS
import modal

app = modal.App(LLM_JUDGE_APP_NAME)
OPENAI_RATE_LIMIT_SLEEP_SECONDS = 0.0

@app.function(
    image=image,
    secrets=[modal_secret],
    timeout=MODAL_TIMEOUT,
    scaledown_window=SCALEDOWN_WINDOW,
    min_containers=MIN_CONTAINERS,
)
def run_llm_judge(query, chunks):
    import time
    from helpers.openai_decoder import run_openai_llm, extract_rate_limit_sleep_seconds

    global OPENAI_RATE_LIMIT_SLEEP_SECONDS

    judge_profile_config = MODEL_PROFILES[LLM_JUDGE_PROFILE].copy()
    prompt_template = judge_profile_config.pop("prompt_template")
    formatted_chunks = []
    for idx, chunk in enumerate(chunks, start=1):
        if isinstance(chunk, dict):
            chunk_id = chunk.get("id", idx)
            chunk_text = chunk.get("text", "")
        else:
            chunk_id = idx
            chunk_text = str(chunk)
        formatted_chunks.append(
            "-----\n"
            f"<chunk>\n<id>{chunk_id}</id>\n<text>\n{chunk_text}\n</text>\n</chunk>\n"
            "-----"
        )
    prompt = prompt_template.format(
        query=query,
        chunks="\n".join(formatted_chunks),
    )

    if judge_profile_config["provider"] == "openai":
        max_attempts = 2
        for attempt_index in range(max_attempts):
            if OPENAI_RATE_LIMIT_SLEEP_SECONDS > 0:
                print(
                    "run_llm_judge: honoring prior rate-limit cooldown before request:\n"
                    f"\tsleep seconds: {OPENAI_RATE_LIMIT_SLEEP_SECONDS:.2f}"
                )
                time.sleep(OPENAI_RATE_LIMIT_SLEEP_SECONDS)
            try:
                judge_result, _ = run_openai_llm(
                    current_turn_input_text=prompt,
                    system_prompt=judge_profile_config["system_prompt"],
                    model_name_or_path=judge_profile_config["model_name_or_path"],
                    max_new_tokens=judge_profile_config["max_new_tokens"],
                    enable_thinking=judge_profile_config["enable_thinking"],
                    reasoning_effort=judge_profile_config["reasoning_effort"],
                    decoder_profile=LLM_JUDGE_PROFILE,
                    return_prompt_text=judge_profile_config["return_prompt_text"],
                )
                return judge_result
            except Exception as e:
                error_message = str(e)
                is_last_attempt = attempt_index == max_attempts - 1
                if (
                    is_last_attempt
                    or "rate_limit_exceeded" not in error_message
                    and "Please try again in" not in error_message
                ):
                    print(f"run_llm_judge: OpenAI generation failed: {e}")
                    return None
                sleep_seconds = extract_rate_limit_sleep_seconds(error_message) + 1.0
                OPENAI_RATE_LIMIT_SLEEP_SECONDS = max(
                    OPENAI_RATE_LIMIT_SLEEP_SECONDS,
                    sleep_seconds,
                )
                print(
                    "run_llm_judge: rate-limited, sleeping before retry:\n"
                    f"\tattempt: {attempt_index + 1}/{max_attempts}\n"
                    f"\tsleep seconds: {sleep_seconds:.2f}"
                )
                time.sleep(sleep_seconds)
        return None

    judge_profile_config.pop("reasoning_effort", None)
    run_qwen3_lm_or_vlm = modal.Function.from_name(
        DECODER_APP_NAME,
        RUN_QWEN3_LM_OR_VLM_FUNCTION_NAME,
    )
    try:
        judge_result, _ = run_qwen3_lm_or_vlm.remote(
            context=[],
            current_turn_input_text=prompt,
            current_turn_image_in_bytes=None,
            **judge_profile_config,
            decoder_profile=LLM_JUDGE_PROFILE
        )
        return judge_result
    except Exception as e:
        print(f"run_llm_judge: local decoder generation failed: {e}")
        return None
