from config.decoder import (
    ABSTRACT_OPENING_TAG,
    ABSTRACT_CLOSING_TAG,
    SUMMARY_OPENING_TAG,
    SUMMARY_CLOSING_TAG,
    CLEANED_TEXT_OPENING_TAG,
    CLEANED_TEXT_CLOSING_TAG,
    QUESTION_OPENING_TAG,
    QUESTION_CLOSING_TAG,
    ANSWER_OPENING_TAG,
    ANSWER_CLOSING_TAG,
)
from helpers.decoder import remove_think_tokens, extract_lm_cleaned_content


def run_openai_data_cleaner(
    current_turn_input_text,
    system_prompt,
    model_name_or_path,
    max_new_tokens,
    # temperature,
    # top_p,
    enable_thinking,
    reasoning_effort,
    return_prompt_text=False,
):
    from openai import OpenAI

    client = OpenAI()
    reasoning = {
        "effort": reasoning_effort if enable_thinking else "minimal"
    }
    response = client.responses.create(
        model=model_name_or_path,
        instructions=system_prompt,
        input=current_turn_input_text,
        max_output_tokens=max_new_tokens,
        # temperature=temperature,
        # top_p=top_p,
        reasoning=reasoning,
    )
    output_text = response.output_text
    if enable_thinking:
        output_text = remove_think_tokens(output_text)
    parsed_output = extract_lm_cleaned_content(
        output_text,
        ABSTRACT_OPENING_TAG,
        ABSTRACT_CLOSING_TAG,
        SUMMARY_OPENING_TAG,
        SUMMARY_CLOSING_TAG,
        CLEANED_TEXT_OPENING_TAG,
        CLEANED_TEXT_CLOSING_TAG,
        QUESTION_OPENING_TAG,
        QUESTION_CLOSING_TAG,
        ANSWER_OPENING_TAG,
        ANSWER_CLOSING_TAG,
    )
    if return_prompt_text:
        return parsed_output, f"{system_prompt}\n\n{current_turn_input_text}"
    return parsed_output, None
