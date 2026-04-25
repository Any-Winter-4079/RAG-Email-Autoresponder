##########################################
# Helper 1: Extract content between tags #
##########################################
def extract_matched_content(response, opening_tag, closing_tag):
    import re

    if response is None:
        return None

    # match text within tags (we use () to return content inside the match)
    # . -> any character
    # * -> zero or more times
    # ? -> stop as soon as possible (at the 1st closing tag match, vs at the last one)
    matches = re.findall(f"{opening_tag}(.*?){closing_tag}", response, flags=re.DOTALL)
    return [match.strip() for match in matches]

##############################################
# Helper 2: Remove <think>...</think> tokens #
##############################################
def remove_think_tokens(response):
    import re

    # if no response, do not reply
    if response is None:
        return None
    
    # if response, remove <think>...</think> tokens
    return re.sub(r"<think>.*?</think>", "", response, flags=re.DOTALL)

############################################################
# Helper 3: Remove model-specific reasoning wrapper tokens #
############################################################
def remove_reasoning_wrappers(response, model_family, enable_thinking):
    import re
    from config.decoder import GEMMA4_MODEL_FAMILY

    if response is None:
        return None

    if enable_thinking:
        response = remove_think_tokens(response)

    if model_family == GEMMA4_MODEL_FAMILY:
        # gemma 4 26b a4b can emit an empty thought wrapper even with thinking disabled:
        # https://huggingface.co/google/gemma-4-26B-A4B-it
        response = re.sub(r"<\|channel\|?>thought\s*.*?<channel\|>\s*", "", response, flags=re.DOTALL)

    return response

###################################################
# Helper 4: Extract <message>...</message> tokens #
###################################################
def extract_message_content(
        response,
        no_message_opening_tag,
        no_message_closing_tag,
        message_opening_tag,
        message_closing_tag
        ):
    # if no response, do not reply
    if response is None:
        return None

    # if LM thinks it does not have enough info to answer, we do not reply
    no_message = extract_matched_content(response, no_message_opening_tag, no_message_closing_tag)
    if no_message:
        return None
    
    # if LM thinks it has enough info to answer, use the message
    message = extract_matched_content(response, message_opening_tag, message_closing_tag)
    if message:
        # return message (there should be a single message per response)
        return message[0]

    # if neither tag, we do not reply (LM did not follow intructions or instructions were not correct)
    return None

############################################################################
# Helper 5: Extract <from>...</from>, <to>...</to>, <subject>...</subject> #
#                   <body>...</body>                                       # 
############################################################################
def extract_thread_content(
        response,
        thread_opening_tag,
        thread_closing_tag,
        message_opening_tag,
        message_closing_tag,
        from_opening_tag,
        from_closing_tag,
        to_opening_tag,
        to_closing_tag,
        subject_opening_tag,
        subject_closing_tag,
        body_opening_tag,
        body_closing_tag
        ):
    # if no response, do not return content
    if response is None:
        return None

    threads = extract_matched_content(response, thread_opening_tag, thread_closing_tag)
    if not threads:
        return None

    parsed_threads = []
    for thread_text in threads:
        messages = extract_matched_content(thread_text, message_opening_tag, message_closing_tag)
        parsed_messages = []
        for message_text in messages:
            from_value = extract_matched_content(message_text, from_opening_tag, from_closing_tag)
            to_value = extract_matched_content(message_text, to_opening_tag, to_closing_tag)
            subject_value = extract_matched_content(message_text, subject_opening_tag, subject_closing_tag)
            body_value = extract_matched_content(message_text, body_opening_tag, body_closing_tag)

            parsed_messages.append({
                "from": from_value[0] if from_value else None,
                "to": to_value[0] if to_value else None,
                "subject": subject_value[0] if subject_value else None,
                "body": body_value[0] if body_value else None
            })
        parsed_threads.append({"messages": parsed_messages})

    return parsed_threads

##############################################################################
# Helper 6: Extract <abstract>...</abstract>, <summary>...</summary>,        # 
#                   <cleanedtext>...</cleanedtext>, <question>...</question> #
#                   <answer>...</answer>                                     #
##############################################################################
def extract_lm_cleaned_content(
        response,
        abstract_opening_tag,
        abstract_closing_tag,
        summary_opening_tag,
        summary_closing_tag,
        cleanedtext_opening_tag,
        cleanedtext_closing_tag,
        question_opening_tag,
        question_closing_tag,
        answer_opening_tag,
        answer_closing_tag
        ):

    # if no response, do not return content
    if response is None:
        return None
    
    # extract abstract, summary, cleanedtext, questions, answers
    abstract = extract_matched_content(response, abstract_opening_tag, abstract_closing_tag)
    if abstract:
        # there should be a single abstract per response
        abstract = abstract[0]
    summary = extract_matched_content(response, summary_opening_tag, summary_closing_tag)
    if summary:
        # there should be a single summary per response
        summary = summary[0]
    cleanedtext = extract_matched_content(response, cleanedtext_opening_tag, cleanedtext_closing_tag)
    if cleanedtext:
        # there should be a single cleanedtext per response
        cleanedtext = cleanedtext[0]
    questions = extract_matched_content(response, question_opening_tag, question_closing_tag)
    answers = extract_matched_content(response, answer_opening_tag, answer_closing_tag)

    return {
        "abstract": abstract,
        "summary": summary,
        "cleanedtext": cleanedtext,
        "questions": questions,
        "answers": answers,
    }

#############################################
# Helper 7: Extract email-KB curator output #
#############################################
def extract_email_knowledge_base_curator_content(
        response,
        thread_opening_tag,
        thread_closing_tag,
        no_useful_information_opening_tag,
        no_useful_information_closing_tag,
        abstract_opening_tag,
        abstract_closing_tag,
        summary_opening_tag,
        summary_closing_tag,
        cleanedtext_opening_tag,
        cleanedtext_closing_tag,
        question_opening_tag,
        question_closing_tag,
        answer_opening_tag,
        answer_closing_tag,
        ):

    if response is None:
        return None

    threads = extract_matched_content(
        response,
        thread_opening_tag,
        thread_closing_tag,
    )
    if not threads:
        return None

    parsed_threads = []
    for thread_text in threads:
        no_useful_information = extract_matched_content(
            thread_text,
            no_useful_information_opening_tag,
            no_useful_information_closing_tag,
        )
        if no_useful_information:
            parsed_threads.append({
                "no_useful_information": True,
                "abstract": None,
                "summary": None,
                "cleanedtext": None,
                "questions": [],
                "answers": [],
            })
            continue

        parsed_output = extract_lm_cleaned_content(
            thread_text,
            abstract_opening_tag,
            abstract_closing_tag,
            summary_opening_tag,
            summary_closing_tag,
            cleanedtext_opening_tag,
            cleanedtext_closing_tag,
            question_opening_tag,
            question_closing_tag,
            answer_opening_tag,
            answer_closing_tag,
        )
        parsed_output["no_useful_information"] = False
        parsed_threads.append(parsed_output)

    return parsed_threads

#############################################
# Helper 8: Extract query rewriter sections #
#############################################
def extract_query_rewriter_content(
        response,
        keyword_queries_opening_tag,
        keyword_queries_closing_tag,
        natural_queries_opening_tag,
        natural_queries_closing_tag,
        hyde_queries_opening_tag,
        hyde_queries_closing_tag,
        question_queries_opening_tag,
        question_queries_closing_tag,
        reranker_query_opening_tag,
        reranker_query_closing_tag,
        query_opening_tag,
        query_closing_tag,
        no_request_opening_tag,
        no_request_closing_tag,
        ):

    # if no response, do not return content
    if response is None:
        return None

    no_request_sections = extract_matched_content(response, no_request_opening_tag, no_request_closing_tag)
    if no_request_sections:
        return {
            "no_request": True,
            "keyword_queries": [],
            "natural_queries": [],
            "hyde_queries": [],
            "question_queries": [],
            "reranker_query": None,
        }

    # extract keyword queries, natural queries, hyde queries, question queries, reranker query
    keyword_query_sections = extract_matched_content(response, keyword_queries_opening_tag, keyword_queries_closing_tag)
    if keyword_query_sections:
        # there should be a single keyword query section per response, so we extract the queries inside it
        keyword_queries = extract_matched_content(keyword_query_sections[0], query_opening_tag, query_closing_tag)
    else:
        keyword_queries = []

    natural_query_sections = extract_matched_content(response, natural_queries_opening_tag, natural_queries_closing_tag)
    if natural_query_sections:
        # there should be a single natural query section per response, so we extract the queries inside it
        natural_queries = extract_matched_content(natural_query_sections[0], query_opening_tag, query_closing_tag)
    else:
        natural_queries = []

    hyde_query_sections = extract_matched_content(response, hyde_queries_opening_tag, hyde_queries_closing_tag)
    if hyde_query_sections:
        # there should be a single hyde query section per response, so we extract the queries inside it
        hyde_queries = extract_matched_content(hyde_query_sections[0], query_opening_tag, query_closing_tag)
    else:
        hyde_queries = []

    question_query_sections = extract_matched_content(response, question_queries_opening_tag, question_queries_closing_tag)
    if question_query_sections:
        # there should be a single question query section per response, so we extract the queries inside it
        question_queries = extract_matched_content(question_query_sections[0], query_opening_tag, query_closing_tag)
    else:
        question_queries = []

    reranker_queries = extract_matched_content(response, reranker_query_opening_tag, reranker_query_closing_tag)
    if reranker_queries:
        # there should be a single reranker query per response
        reranker_query = reranker_queries[0]
    else:
        reranker_query = None

    return {
        "no_request": False,
        "keyword_queries": keyword_queries,
        "natural_queries": natural_queries,
        "hyde_queries": hyde_queries,
        "question_queries": question_queries,
        "reranker_query": reranker_query,
    }

##########################################
# Helper 9: Extract <score>...</score>'s #
##########################################
def extract_score_values(response, score_opening_tag, score_closing_tag):
    scores = extract_matched_content(response, score_opening_tag, score_closing_tag)
    if not scores:
        return None

    parsed_scores = []
    for score in scores:
        try:
            parsed_scores.append(float(score))
        except Exception:
            return None
    return parsed_scores

#######################################################
# Helper 10: Extract structured LLM judge output tags #
#######################################################
def extract_llm_judge_content(
        response,
        answerability_opening_tag,
        answerability_closing_tag,
        subqueries_opening_tag,
        subqueries_closing_tag,
        subquery_opening_tag,
        subquery_closing_tag,
        subquery_text_opening_tag,
        subquery_text_closing_tag,
        subquery_answerability_opening_tag,
        subquery_answerability_closing_tag,
        subquery_confidence_opening_tag,
        subquery_confidence_closing_tag,
        subquery_supporting_chunk_ids_opening_tag,
        subquery_supporting_chunk_ids_closing_tag,
        subquery_insufficient_chunk_ids_opening_tag,
        subquery_insufficient_chunk_ids_closing_tag,
        subquery_rationale_opening_tag,
        subquery_rationale_closing_tag,
        chunk_id_opening_tag,
        chunk_id_closing_tag,
        draft_answer_opening_tag,
        draft_answer_closing_tag,
        ):
    if response is None:
        return None

    answerability_values = extract_matched_content(
        response,
        answerability_opening_tag,
        answerability_closing_tag,
    )
    draft_answer_values = extract_matched_content(
        response,
        draft_answer_opening_tag,
        draft_answer_closing_tag,
    )
    subquery_sections = extract_matched_content(
        response,
        subqueries_opening_tag,
        subqueries_closing_tag,
    )

    if not answerability_values:
        return None

    if subquery_sections:
        raw_subqueries = extract_matched_content(
            subquery_sections[0],
            subquery_opening_tag,
            subquery_closing_tag,
        )
    else:
        raw_subqueries = []

    subqueries = []
    for raw_subquery in raw_subqueries:
        subquery_text_values = extract_matched_content(
            raw_subquery,
            subquery_text_opening_tag,
            subquery_text_closing_tag,
        )
        subquery_answerability_values = extract_matched_content(
            raw_subquery,
            subquery_answerability_opening_tag,
            subquery_answerability_closing_tag,
        )
        subquery_confidence_values = extract_matched_content(
            raw_subquery,
            subquery_confidence_opening_tag,
            subquery_confidence_closing_tag,
        )
        subquery_supporting_sections = extract_matched_content(
            raw_subquery,
            subquery_supporting_chunk_ids_opening_tag,
            subquery_supporting_chunk_ids_closing_tag,
        )
        subquery_insufficient_sections = extract_matched_content(
            raw_subquery,
            subquery_insufficient_chunk_ids_opening_tag,
            subquery_insufficient_chunk_ids_closing_tag,
        )
        subquery_rationale_values = extract_matched_content(
            raw_subquery,
            subquery_rationale_opening_tag,
            subquery_rationale_closing_tag,
        )
        if subquery_supporting_sections:
            subquery_supporting_chunk_ids = extract_matched_content(
                subquery_supporting_sections[0],
                chunk_id_opening_tag,
                chunk_id_closing_tag,
            )
        else:
            subquery_supporting_chunk_ids = []
        if subquery_insufficient_sections:
            subquery_insufficient_chunk_ids = extract_matched_content(
                subquery_insufficient_sections[0],
                chunk_id_opening_tag,
                chunk_id_closing_tag,
            )
        else:
            subquery_insufficient_chunk_ids = []
        try:
            subquery_confidence = round(float(subquery_confidence_values[0]), 1)
        except Exception:
            subquery_confidence = None
        subqueries.append({
            "text": subquery_text_values[0] if subquery_text_values else "",
            "answerability": subquery_answerability_values[0] if subquery_answerability_values else "",
            "confidence": subquery_confidence,
            "supporting_chunk_ids": subquery_supporting_chunk_ids,
            "insufficient_chunk_ids": subquery_insufficient_chunk_ids,
            "rationale": subquery_rationale_values[0] if subquery_rationale_values else "",
        })

    return {
        "answerability": answerability_values[0],
        "subqueries": subqueries,
        "draft_answer": draft_answer_values[0] if draft_answer_values else "",
    }

###########################
# Helper 11: Count tokens #
###########################
def count_tokens(tokenizer, text):
    try:
        return len(tokenizer.encode(text, add_special_tokens=False))
    except Exception as e:
        print(f"count_tokens: error counting tokens: {e}")
        return 0

#################################
# Helper 12: Truncate to tokens #
#################################
def truncate_to_tokens(tokenizer, text, max_tokens):
    try:
        token_ids = tokenizer.encode(text, add_special_tokens=False)
    except Exception as e:
        print(f"truncate_to_tokens: error encoding text: {e}")
        return None
    if len(token_ids) <= max_tokens:
        return text
    ellipsis = "..."
    ellipsis_tokens = len(tokenizer.encode(ellipsis, add_special_tokens=False))
    if max_tokens <= ellipsis_tokens:
        return ellipsis
    truncated = tokenizer.decode(
        token_ids[:max_tokens - ellipsis_tokens],
        skip_special_tokens=True
    ).strip()
    return f"{truncated}{ellipsis}" if truncated else ellipsis

##################################
# Helper 13: Run local LM or VLM #
##################################
def run_local_lm_or_vlm(
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
        decoder_profile=None,
        ):
    import io
    import torch
    from PIL import Image
    from peft import AutoPeftModelForCausalLM
    from transformers import AutoTokenizer, AutoModelForCausalLM, AutoProcessor, Qwen3VLForConditionalGeneration
    from config.decoder import (
        DATA_CLEANER_PROFILE,
        EMAIL_KNOWLEDGE_BASE_CURATOR_PROFILE,
        EMAIL_WRITER_PROFILE,
        THREAD_GROUPER_PROFILE,
        QUERY_REWRITER_PROFILE,
        LLM_JUDGE_PROFILE,
        GEMMA4_MODEL_FAMILY,
        MESSAGE_OPENING_TAG,
        MESSAGE_CLOSING_TAG,
        NO_MESSAGE_OPENING_TAG,
        NO_MESSAGE_CLOSING_TAG,
        SUMMARY_OPENING_TAG,
        SUMMARY_CLOSING_TAG,
        ABSTRACT_OPENING_TAG,
        ABSTRACT_CLOSING_TAG,
        CLEANED_TEXT_OPENING_TAG,
        CLEANED_TEXT_CLOSING_TAG,
        THREAD_OPENING_TAG,
        THREAD_CLOSING_TAG,
        THREAD_MESSAGE_OPENING_TAG,
        THREAD_MESSAGE_CLOSING_TAG,
        THREAD_FROM_OPENING_TAG,
        THREAD_FROM_CLOSING_TAG,
        THREAD_TO_OPENING_TAG,
        THREAD_TO_CLOSING_TAG,
        THREAD_SUBJECT_OPENING_TAG,
        THREAD_SUBJECT_CLOSING_TAG,
        THREAD_BODY_OPENING_TAG,
        THREAD_BODY_CLOSING_TAG,
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
        KEYWORD_QUERIES_OPENING_TAG,
        KEYWORD_QUERIES_CLOSING_TAG,
        NATURAL_QUERIES_OPENING_TAG,
        NATURAL_QUERIES_CLOSING_TAG,
        HYDE_QUERIES_OPENING_TAG,
        HYDE_QUERIES_CLOSING_TAG,
        QUESTION_QUERIES_OPENING_TAG,
        QUESTION_QUERIES_CLOSING_TAG,
        RERANKER_QUERY_OPENING_TAG,
        RERANKER_QUERY_CLOSING_TAG,
        NO_REQUEST_OPENING_TAG,
        NO_REQUEST_CLOSING_TAG,
        NO_USEFUL_INFORMATION_OPENING_TAG,
        NO_USEFUL_INFORMATION_CLOSING_TAG,
        QUERY_OPENING_TAG,
        QUERY_CLOSING_TAG,
    )

    if (not hasattr(run_local_lm_or_vlm, "model") or
        getattr(run_local_lm_or_vlm, "model_name_or_path", None) != model_name_or_path or
        getattr(run_local_lm_or_vlm, "model_family", None) != model_family or
        getattr(run_local_lm_or_vlm, "is_vision_model", None) != is_vision_model):

        print(f"run_local_lm_or_vlm: loading model from {model_name_or_path} (is_vision_model: {is_vision_model}, provider: {provider})...")

        try:
            run_local_lm_or_vlm.model = AutoPeftModelForCausalLM.from_pretrained(
                model_name_or_path,
                dtype="auto",
                device_map="auto",
                attn_implementation="sdpa" if not use_flash_attention_2 else "flash_attention_2"
            )
        except Exception:
            if is_vision_model:
                run_local_lm_or_vlm.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_name_or_path,
                    dtype="auto",
                    device_map="auto",
                    attn_implementation="sdpa" if not use_flash_attention_2 else "flash_attention_2"
                )
            else:
                run_local_lm_or_vlm.model = AutoModelForCausalLM.from_pretrained(
                    model_name_or_path,
                    dtype="auto",
                    device_map="auto",
                    attn_implementation="sdpa" if not use_flash_attention_2 else "flash_attention_2"
                )
        if is_vision_model or model_family == GEMMA4_MODEL_FAMILY:
            run_local_lm_or_vlm.processor = AutoProcessor.from_pretrained(model_name_or_path)
        else:
            run_local_lm_or_vlm.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        run_local_lm_or_vlm.model_name_or_path = model_name_or_path
        run_local_lm_or_vlm.model_family = model_family
        run_local_lm_or_vlm.is_vision_model = is_vision_model

        print("run_local_lm_or_vlm: model and processor loaded and cached")

    model = run_local_lm_or_vlm.model
    processor = (
        run_local_lm_or_vlm.processor
        if is_vision_model or model_family == GEMMA4_MODEL_FAMILY
        else run_local_lm_or_vlm.tokenizer
    )

    messages = []
    prompt_text = None

    def form_vlm_input_turn_content(input_text, input_image_in_bytes):
        content = [{"type": "text", "text": input_text}]
        if input_image_in_bytes:
            context_image_pil = Image.open(io.BytesIO(input_image_in_bytes))
        else:
            context_image_pil = None
        if context_image_pil:
            content.insert(0, {"type": "image", "image": context_image_pil})
        return content

    messages.append({
        "role": "system",
        "content": [{"type": "text", "text": system_prompt}] if is_vision_model else system_prompt
    })

    for context_turn in context:
        if is_vision_model:
            context_input_turn_content = form_vlm_input_turn_content(
                context_turn["input_text"],
                context_turn.get("input_image"),
            )
        messages.append({
            "role": "user",
            "content": context_input_turn_content if is_vision_model else context_turn["input_text"]
        })
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": context_turn["output_text"]}] if is_vision_model else context_turn["output_text"]
        })

    if is_vision_model:
        current_input_turn_content = form_vlm_input_turn_content(
            current_turn_input_text,
            current_turn_image_in_bytes,
        )
    messages.append({
        "role": "user",
        "content": current_input_turn_content if is_vision_model else current_turn_input_text
    })

    if return_prompt_text:
        if is_vision_model:
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=enable_thinking
            )

    if is_vision_model:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
    elif model_family == GEMMA4_MODEL_FAMILY:
        prompt_text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking
        )
        inputs = processor(text=prompt_text, return_tensors="pt")
    else:
        inputs = processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
            enable_thinking=enable_thinking
        )
    inputs = inputs.to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            use_cache=True,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=True,
        )

    generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=not (model_family == GEMMA4_MODEL_FAMILY and not is_vision_model),
        clean_up_tokenization_spaces=False
    )
    output_text = output_text[0] if output_text else None

    print(f"{output_text}\n\n")

    output_text = remove_reasoning_wrappers(
        output_text,
        model_family=model_family,
        enable_thinking=enable_thinking,
    )

    if decoder_profile == EMAIL_WRITER_PROFILE:
        output_text = extract_message_content(
            output_text,
            NO_MESSAGE_OPENING_TAG,
            NO_MESSAGE_CLOSING_TAG,
            MESSAGE_OPENING_TAG,
            MESSAGE_CLOSING_TAG
        )
    elif decoder_profile == THREAD_GROUPER_PROFILE:
        output_text = extract_thread_content(
            output_text,
            THREAD_OPENING_TAG,
            THREAD_CLOSING_TAG,
            THREAD_MESSAGE_OPENING_TAG,
            THREAD_MESSAGE_CLOSING_TAG,
            THREAD_FROM_OPENING_TAG,
            THREAD_FROM_CLOSING_TAG,
            THREAD_TO_OPENING_TAG,
            THREAD_TO_CLOSING_TAG,
            THREAD_SUBJECT_OPENING_TAG,
            THREAD_SUBJECT_CLOSING_TAG,
            THREAD_BODY_OPENING_TAG,
            THREAD_BODY_CLOSING_TAG
        )
    elif decoder_profile == DATA_CLEANER_PROFILE:
        output_text = extract_lm_cleaned_content(
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
            ANSWER_CLOSING_TAG
        )
    elif decoder_profile == EMAIL_KNOWLEDGE_BASE_CURATOR_PROFILE:
        output_text = extract_email_knowledge_base_curator_content(
            output_text,
            THREAD_OPENING_TAG,
            THREAD_CLOSING_TAG,
            NO_USEFUL_INFORMATION_OPENING_TAG,
            NO_USEFUL_INFORMATION_CLOSING_TAG,
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
    elif decoder_profile == QUERY_REWRITER_PROFILE:
        output_text = extract_query_rewriter_content(
            output_text,
            KEYWORD_QUERIES_OPENING_TAG,
            KEYWORD_QUERIES_CLOSING_TAG,
            NATURAL_QUERIES_OPENING_TAG,
            NATURAL_QUERIES_CLOSING_TAG,
            HYDE_QUERIES_OPENING_TAG,
            HYDE_QUERIES_CLOSING_TAG,
            QUESTION_QUERIES_OPENING_TAG,
            QUESTION_QUERIES_CLOSING_TAG,
            RERANKER_QUERY_OPENING_TAG,
            RERANKER_QUERY_CLOSING_TAG,
            QUERY_OPENING_TAG,
            QUERY_CLOSING_TAG,
            NO_REQUEST_OPENING_TAG,
            NO_REQUEST_CLOSING_TAG,
        )
    elif decoder_profile == LLM_JUDGE_PROFILE:
        output_text = extract_llm_judge_content(
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
        print(f"run_local_lm_or_vlm: unknown decoder_profile '{decoder_profile}'")
        output_text = None

    return (output_text, prompt_text) if return_prompt_text else (output_text, None)
