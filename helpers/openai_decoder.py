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
    ANSWERABILITY_OPENING_TAG,
    ANSWERABILITY_CLOSING_TAG,
    SUBQUERIES_OPENING_TAG,
    SUBQUERIES_CLOSING_TAG,
    SUBQUERY_OPENING_TAG,
    SUBQUERY_CLOSING_TAG,
    SUBQUERY_TEXT_OPENING_TAG,
    SUBQUERY_TEXT_CLOSING_TAG,
    SUBQUERY_ANSWERABILITY_OPENING_TAG,
    SUBQUERY_ANSWERABILITY_CLOSING_TAG,
    SUBQUERY_CONFIDENCE_OPENING_TAG,
    SUBQUERY_CONFIDENCE_CLOSING_TAG,
    SUBQUERY_SUPPORTING_CHUNK_IDS_OPENING_TAG,
    SUBQUERY_SUPPORTING_CHUNK_IDS_CLOSING_TAG,
    SUBQUERY_INSUFFICIENT_CHUNK_IDS_OPENING_TAG,
    SUBQUERY_INSUFFICIENT_CHUNK_IDS_CLOSING_TAG,
    SUBQUERY_RATIONALE_OPENING_TAG,
    SUBQUERY_RATIONALE_CLOSING_TAG,
    CHUNK_ID_OPENING_TAG,
    CHUNK_ID_CLOSING_TAG,
    DRAFT_ANSWER_OPENING_TAG,
    DRAFT_ANSWER_CLOSING_TAG,
    DATA_CLEANER_PROFILE,
    LLM_JUDGE_PROFILE,
)
from helpers.decoder import (
    remove_think_tokens,
    extract_lm_cleaned_content,
    extract_llm_judge_content,
)

#########################################################
# Helper 1: Extract retry sleep seconds from rate limit #
#########################################################
def extract_rate_limit_sleep_seconds(error_message):
    import re

    sleep_match = re.search(r"Please try again in ([0-9]+(?:\.[0-9]+)?)s", error_message)
    if sleep_match:
        return float(sleep_match.group(1)) + 1.0
    return 30.0

######################################
# Helper 2: Run OpenAI LLM and parse #
######################################
def run_openai_llm(
    current_turn_input_text,
    system_prompt,
    model_name_or_path,
    max_new_tokens,
    # temperature,
    # top_p,
    enable_thinking,
    reasoning_effort,
    decoder_profile,
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

    if decoder_profile == DATA_CLEANER_PROFILE:
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
    elif decoder_profile == LLM_JUDGE_PROFILE:
        parsed_output = extract_llm_judge_content(
            output_text,
            ANSWERABILITY_OPENING_TAG,
            ANSWERABILITY_CLOSING_TAG,
            SUBQUERIES_OPENING_TAG,
            SUBQUERIES_CLOSING_TAG,
            SUBQUERY_OPENING_TAG,
            SUBQUERY_CLOSING_TAG,
            SUBQUERY_TEXT_OPENING_TAG,
            SUBQUERY_TEXT_CLOSING_TAG,
            SUBQUERY_ANSWERABILITY_OPENING_TAG,
            SUBQUERY_ANSWERABILITY_CLOSING_TAG,
            SUBQUERY_CONFIDENCE_OPENING_TAG,
            SUBQUERY_CONFIDENCE_CLOSING_TAG,
            SUBQUERY_SUPPORTING_CHUNK_IDS_OPENING_TAG,
            SUBQUERY_SUPPORTING_CHUNK_IDS_CLOSING_TAG,
            SUBQUERY_INSUFFICIENT_CHUNK_IDS_OPENING_TAG,
            SUBQUERY_INSUFFICIENT_CHUNK_IDS_CLOSING_TAG,
            SUBQUERY_RATIONALE_OPENING_TAG,
            SUBQUERY_RATIONALE_CLOSING_TAG,
            CHUNK_ID_OPENING_TAG,
            CHUNK_ID_CLOSING_TAG,
            DRAFT_ANSWER_OPENING_TAG,
            DRAFT_ANSWER_CLOSING_TAG,
        )
    else:
        parsed_output = None

    if return_prompt_text:
        return parsed_output, f"{system_prompt}\n\n{current_turn_input_text}"
    return parsed_output, None
