from config.general import modal_secret
from config.modal_apps import EMAIL_AGENT_APP_NAME
from config.email_agent import (
    image,
    MODAL_TIMEOUT,
    EMAIL_HOUR,
    EMAIL_MINUTE
)
import modal

# Modal
app = modal.App(EMAIL_AGENT_APP_NAME)

@app.function(
        image=image,
        # with Cron format "Minute Hour Day Month DayOfWeek":
        schedule=modal.Cron(f"{EMAIL_MINUTE} {EMAIL_HOUR} * * *", timezone="Europe/Madrid"),
        secrets=[modal_secret],
        timeout=MODAL_TIMEOUT,
        # https://modal.com/docs/guide/region-selection: price multiplier = 1.25x
        region="eu-south-2" # "spaincentral": AZR Madrid / "eu-south-2": AWS Spain
)
def run_email_agent():
    import os
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from helpers.decoder import count_tokens, get_tokenizer, truncate_to_tokens

    from helpers.data import (
        assign_thread_ids_by_subject_and_participant_overlap_for_production,
        get_unquoted_text
    )
    from config.decoder import MODEL_PROFILES, EMAIL_WRITER_PROFILE
    from config.crawler_agent import CRAWL_DAY, CRAWL_MONTH
    from config.email_agent import (
        MAX_EMAILS,
        MAX_UNQUOTED_TOKENS_PER_CONTEXT_EMAIL,
        MAX_QUOTED_TOKENS_PER_CONTEXT_EMAIL,
        EMAIL_WRITER_THREAD_CONTEXT_TOKEN_RATIO,
        EMAIL_WRITER_RAG_CONTEXT_TOKEN_RATIO,
        INBOX_FOLDER,
        SENT_FOLDER,
        UNREAD_ONLY,
        LEAVE_UNREAD,
        LAST_N_DAYS,
        SEND_TO_SELF,
        SAVE_AS_DRAFT,
        DRAFTS_FOLDER,
        TOP_K_PER_QUERY,
        TOP_K_AFTER_QUERY_FUSION,
        USE_MAX_SIMILARITY_QUERY_FUSION_BEFORE_RRF,
        TOP_K_AFTER_SOURCE_RRF,
        TOP_K_AFTER_RERANK,
        RERANKER_NAME,
        BASE_DATA_VARIANT,
        SOURCE_TO_ENCODER_SETTINGS
    )
    from helpers.email_agent import (
        transform_env_csv_into_list,
        read_latest_emails,
        format_response_quoting_original_body,
        format_rag_context_for_email_writer,
        build_formatted_thread_emails,
        join_formatted_thread_emails,
        truncate_formatted_thread_emails,
        build_email_writer_prompt,
        send_emails,
        save_drafts,
        mark_emails_as_read
    )
    from helpers.eval import build_base_data_variant_to_source_to_encoder_settings
    from helpers.retrieval_pipeline import run_retrieval_pipeline

    today = datetime.now(ZoneInfo("Europe/Madrid"))
    if today.day == CRAWL_DAY and today.month == CRAWL_MONTH:
        print("run_email_agent: today is crawling day, skipping email agent")
        return

    # required env vars
    imap_server = os.getenv("IMAP_SERVER")
    imap_port_str = os.getenv("IMAP_PORT")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port_str = os.getenv("SMTP_PORT")
    imap_email = os.getenv("IMAP_EMAIL")
    smtp_email = os.getenv("SMTP_EMAIL")
    password = os.getenv("PASSWORD")
    my_email_addresses = os.getenv("MY_EMAIL_ADDRESSES")
    my_name = os.getenv("MY_NAME")
    my_description = os.getenv("MY_DESCRIPTION")

    missing_env_vars = []
    if not imap_server: missing_env_vars.append("IMAP_SERVER")
    if not imap_port_str: missing_env_vars.append("IMAP_PORT")
    if not smtp_server: missing_env_vars.append("SMTP_SERVER")
    if not smtp_port_str: missing_env_vars.append("SMTP_PORT")
    if not imap_email: missing_env_vars.append("IMAP_EMAIL")
    if not smtp_email: missing_env_vars.append("SMTP_EMAIL")
    if not password: missing_env_vars.append("PASSWORD")
    if not my_email_addresses: missing_env_vars.append("MY_EMAIL_ADDRESSES")
    if not my_name: missing_env_vars.append("MY_NAME")
    if not my_description: missing_env_vars.append("MY_DESCRIPTION")

    if missing_env_vars:
        print(f"run_email_agent: missing required environment variables: {', '.join(missing_env_vars)}")
        return

    # optional env vars
    blacklisted_emails = transform_env_csv_into_list(os.getenv("BLACKLISTED_EMAILS", ""))
    blacklisted_emails.append(smtp_email.lower()) # add self
    blacklisted_emails = list(set(blacklisted_emails))
    blacklisted_domains = transform_env_csv_into_list(os.getenv("BLACKLISTED_DOMAINS", ""))
    my_email_addresses = transform_env_csv_into_list(my_email_addresses)
    if not my_email_addresses:
        print("run_email_agent: MY_EMAIL_ADDRESSES must include at least one email")
        return
    
    email_writer_profile_config = MODEL_PROFILES[EMAIL_WRITER_PROFILE].copy()

    # find decoder service
    try:
        decoder_app_name = email_writer_profile_config.pop("decoder_app_name")
        decoder_function_name = email_writer_profile_config.pop("decoder_function_name")
        run_local_lm_or_vlm = modal.Function.from_name(decoder_app_name, decoder_function_name)
    except Exception as e:
        print(f"run_email_agent: failed to find decoder service. Is it deployed? Error: {e}")
        return

    # read latest emails
    emails = read_latest_emails(
        max_emails=MAX_EMAILS,
        folder=INBOX_FOLDER,
        last_n_days=LAST_N_DAYS,
        imap_email=imap_email,
        password=password,
        unread_only=UNREAD_ONLY,
        imap_server=imap_server,
        imap_port=int(imap_port_str),
        blacklisted_emails=blacklisted_emails,
        blacklisted_domains=blacklisted_domains
    )
    if not emails:
        print("run_email_agent: no new emails to process")
        return
    else:
        print(f"run_email_agent: {len(emails)} new emails to process")

    # load additional context emails from inbox and sent folders
    context_inbox_emails = read_latest_emails(
        max_emails=None,
        folder=INBOX_FOLDER,
        last_n_days=LAST_N_DAYS,
        imap_email=imap_email,
        password=password,
        unread_only=False,
        imap_server=imap_server,
        imap_port=int(imap_port_str),
        blacklisted_emails=blacklisted_emails,
        blacklisted_domains=blacklisted_domains
    )
    # reverse to match config/decoder's oldest to newest
    context_inbox_emails = list(reversed(context_inbox_emails))
    context_sent_emails = read_latest_emails(
        max_emails=None,
        folder=SENT_FOLDER,
        last_n_days=LAST_N_DAYS,
        imap_email=imap_email,
        password=password,
        unread_only=False,
        imap_server=imap_server,
        imap_port=int(imap_port_str),
        blacklisted_emails=blacklisted_emails,
        blacklisted_domains=blacklisted_domains
    )
    # reverse to match config/decoder's oldest to newest
    context_sent_emails = list(reversed(context_sent_emails))
    print(
        "run_email_agent: loaded "
        f"{len(context_inbox_emails)} inbox context emails and "
        f"{len(context_sent_emails)} sent context emails"
    )

    # total context emails are the "set" of inbox emails, sent emails, and emails to answer
    # because the unread emails to answer must be present in the thread assignment window
    combined_context_emails = context_inbox_emails + context_sent_emails + list(reversed(emails))
    # we require inbox+sent+emails (to reply to) to form the thread ids, despite later
    # separating again 
    unique_context_emails = []
    seen_context_ids = set()
    for email in combined_context_emails:
        email_id = email.get("id")
        if email_id in seen_context_ids:
            continue
        seen_context_ids.add(email_id)
        unique_context_emails.append(email)

    # add thread ids (by normalized subject + participant overlap)
    combined_context_emails = assign_thread_ids_by_subject_and_participant_overlap_for_production(
        unique_context_emails,
        my_email_addresses
    )

    # select decoder configuration for email writing
    email_writer_profile_config = MODEL_PROFILES[EMAIL_WRITER_PROFILE].copy()

    # pop config keys run_local_lm_or_vlm would not expect
    email_writer_profile_config.pop("decoder_app_name")
    email_writer_profile_config.pop("decoder_function_name")
    prompt_template = email_writer_profile_config.pop("prompt_template")
    max_input_tokens = email_writer_profile_config.pop("max_input_tokens")

    # get decoder tokenizer 
    decoder_path = email_writer_profile_config["model_name_or_path"]
    decoder_tokenizer = get_tokenizer(decoder_path)

    # map email id to thread id and thread id to emails
    email_id_to_thread_id = {}
    thread_id_to_emails = {}
    for thread_email in combined_context_emails:
        thread_id = thread_email.get("threadID")
        if thread_id is None:
            continue
        email_id = thread_email.get("id")
        if email_id is not None:
            email_id_to_thread_id[email_id] = thread_id
        if thread_id not in thread_id_to_emails:
            thread_id_to_emails[thread_id] = []
        thread_id_to_emails[thread_id].append(thread_email)

    email_id_to_thread_context_emails = {}
    for email in emails:
        email_id = email.get("id")
        thread_id = email_id_to_thread_id.get(email_id)
        thread_context_emails = [
            context_email
            for context_email in thread_id_to_emails.get(thread_id, [])
            if context_email.get("id") != email_id
        ]
        # sort from oldest to newest
        email_id_to_thread_context_emails[email_id] = sorted(
            thread_context_emails,
            key=lambda context_email: context_email.get("date") or datetime.min,
        )

    retrieval_emails = [
        {
            **email,
            "context_emails": email_id_to_thread_context_emails.get(
                email.get("id"),
                [],
            ),
        }
        for email in emails
    ]
    try:
        retrieval_summary = run_retrieval_pipeline(
            emails=retrieval_emails,
            base_data_variant_to_source_to_encoder_settings=(
                build_base_data_variant_to_source_to_encoder_settings(
                    base_data_variant_to_settings={
                        BASE_DATA_VARIANT: {"encoders": {}},
                    },
                    data_sources=list(SOURCE_TO_ENCODER_SETTINGS),
                    source_to_encoder_settings=SOURCE_TO_ENCODER_SETTINGS,
                )
            ),
            top_k_per_query=TOP_K_PER_QUERY,
            top_k_after_query_fusion=TOP_K_AFTER_QUERY_FUSION,
            use_max_similarity_query_fusion_before_rrf=USE_MAX_SIMILARITY_QUERY_FUSION_BEFORE_RRF,
            result_record_metadata={},
            top_k_after_source_rrf=TOP_K_AFTER_SOURCE_RRF,
            top_k_after_rerank=TOP_K_AFTER_RERANK,
            reranker_name=RERANKER_NAME,
            log_prefix="run_email_agent",
            use_runtime_query_rewriter_tokenizer=True,
            max_concurrent_query_rewrites=1,
        )
    except Exception as e:
        print(f"run_email_agent: retrieval pipeline failed: {e}")
        return

    no_request_email_ids = {
        email.get("id")
        for email in retrieval_summary["no_request_emails"]
    }
    email_id_to_rag_context = {}
    for reranker_output in retrieval_summary["base_data_variant_to_reranker_output"].values():
        for retrieval_result in reranker_output["results"]:
            email_id = retrieval_result["email"].get("id")
            if retrieval_result["retrieval_failed"]:
                continue
            rag_context = format_rag_context_for_email_writer(
                retrieval_result["retrieval_results"]
            )
            if rag_context is not None:
                email_id_to_rag_context[email_id] = rag_context

    reply_bodies, original_subjects, recipient_emails, processed_email_ids = [], [], [], []

    # for each email:
    for email in emails:
        # get subject and original body and sender
        original_subject = email.get("subject")
        original_body = email.get("message_body")
        original_sender = email.get("from")

        # if message is incomplete, skip
        if not original_subject or not original_body or not original_sender:
            print(
                "run_email_agent: skipping email because missing data:\n"
                f"\tsubject: {original_subject!r}\n"
                f"\tbody: {original_body!r}\n"
                f"\tsender: {original_sender!r}"
            )
            continue

        email_id = email.get("id")
        if email_id in no_request_email_ids:
            print(f"run_email_agent: skipping '{original_subject}' because query rewriter returned no request")
            continue
        if email_id not in email_id_to_rag_context:
            print(f"run_email_agent: skipping '{original_subject}' because retrieval returned no context")
            continue
        print(f"run_email_agent: generating reply for '{original_subject}' from {original_sender}")

        # get thread context emails for this email
        thread_context_emails = email_id_to_thread_context_emails.get(email_id, [])
        rag_context = email_id_to_rag_context[email_id]
        original_body_for_prompt = original_body

        # first try the full prompt without removing quoted text or truncating anything
        formatted_thread_emails = build_formatted_thread_emails(
            thread_context_emails,
            remove_quoted_text_from_bodies=False,
            decoder_tokenizer=decoder_tokenizer,
            max_unquoted_tokens_per_context_email=MAX_UNQUOTED_TOKENS_PER_CONTEXT_EMAIL,
            max_quoted_tokens_per_context_email=MAX_QUOTED_TOKENS_PER_CONTEXT_EMAIL,
        )
        thread_context = join_formatted_thread_emails(formatted_thread_emails)
        try:
            prompt = build_email_writer_prompt(
                prompt_template=prompt_template,
                my_name=my_name,
                my_description=my_description,
                subject=original_subject,
                sender=original_sender,
                body=original_body_for_prompt,
                thread_context=thread_context,
                rag_context=rag_context,
            )
        except KeyError as e:
            print(f"run_email_agent: error formatting email writer prompt template: {e}")
            continue
        prompt_tokens = count_tokens(decoder_tokenizer, prompt)

        # if the full prompt is too large, remove quoted text from prior thread emails and retry
        if prompt_tokens > max_input_tokens:
            formatted_thread_emails_for_prompt = build_formatted_thread_emails(
                thread_context_emails,
                remove_quoted_text_from_bodies=True,
                decoder_tokenizer=decoder_tokenizer,
                max_unquoted_tokens_per_context_email=MAX_UNQUOTED_TOKENS_PER_CONTEXT_EMAIL,
                max_quoted_tokens_per_context_email=MAX_QUOTED_TOKENS_PER_CONTEXT_EMAIL,
            )
            thread_context = join_formatted_thread_emails(formatted_thread_emails_for_prompt)
            try:
                prompt = build_email_writer_prompt(
                    prompt_template=prompt_template,
                    my_name=my_name,
                    my_description=my_description,
                    subject=original_subject,
                    sender=original_sender,
                    body=original_body_for_prompt,
                    thread_context=thread_context,
                    rag_context=rag_context,
                )
            except KeyError as e:
                print(f"run_email_agent: error formatting email writer prompt template: {e}")
                continue
            prompt_tokens = count_tokens(decoder_tokenizer, prompt)
        else:
            formatted_thread_emails_for_prompt = formatted_thread_emails

        # if prior-thread quoted-text removal is still not enough, remove quoted text from the current email and retry
        if prompt_tokens > max_input_tokens:
            original_body_for_prompt = get_unquoted_text(original_body)
            try:
                prompt = build_email_writer_prompt(
                    prompt_template=prompt_template,
                    my_name=my_name,
                    my_description=my_description,
                    subject=original_subject,
                    sender=original_sender,
                    body=original_body_for_prompt,
                    thread_context=thread_context,
                    rag_context=rag_context,
                )
            except KeyError as e:
                print(f"run_email_agent: error formatting email writer prompt template: {e}")
                continue
            prompt_tokens = count_tokens(decoder_tokenizer, prompt)

        # if quoted-text removal is still not enough, split the available token budget
        # among the current email body, prior thread emails, and retrieved chunks
        if prompt_tokens > max_input_tokens:
            try:
                base_prompt = build_email_writer_prompt(
                    prompt_template=prompt_template,
                    my_name=my_name,
                    my_description=my_description,
                    subject=original_subject,
                    sender=original_sender,
                    body="",
                    thread_context="",
                    rag_context="",
                )
            except KeyError as e:
                print(f"run_email_agent: error formatting email writer prompt template: {e}")
                continue
            n_base_prompt_tokens = count_tokens(decoder_tokenizer, base_prompt)
            variable_token_budget = max_input_tokens - n_base_prompt_tokens
            if variable_token_budget <= 0:
                print("run_email_agent: skipping email because base prompt equals or exceeds input token budget")
                continue
            thread_context_token_budget = max(
                1,
                int(variable_token_budget * EMAIL_WRITER_THREAD_CONTEXT_TOKEN_RATIO),
            )
            rag_context_token_budget = max(
                1,
                int(variable_token_budget * EMAIL_WRITER_RAG_CONTEXT_TOKEN_RATIO),
            )
            body_token_budget = max(
                1,
                variable_token_budget - thread_context_token_budget - rag_context_token_budget,
            )
            print(
                "run_email_agent: truncating email writer prompt with token budgets: "
                f"input={max_input_tokens:,} | "
                f"prompt={n_base_prompt_tokens:,} | "
                f"body={body_token_budget:,} | "
                f"thread_context={thread_context_token_budget:,} | "
                f"rag_context={rag_context_token_budget:,}"
            )
            # truncating the thread separately to keep the conversation starter before newer messages
            if formatted_thread_emails_for_prompt:
                thread_context = truncate_formatted_thread_emails(
                    formatted_thread_emails_for_prompt,
                    thread_context_token_budget,
                    decoder_tokenizer=decoder_tokenizer,
                )
            else:
                thread_context = join_formatted_thread_emails(
                    formatted_thread_emails_for_prompt
                )
            if thread_context is None:
                print("run_email_agent: skipping email because thread context truncation failed")
                continue
            try:
                prompt = build_email_writer_prompt(
                    prompt_template=prompt_template,
                    my_name=my_name,
                    my_description=my_description,
                    subject=original_subject,
                    sender=original_sender,
                    body=original_body_for_prompt,
                    thread_context=thread_context,
                    rag_context=rag_context,
                )
            except KeyError as e:
                print(f"run_email_agent: error formatting email writer prompt template: {e}")
                continue
            prompt_tokens = count_tokens(decoder_tokenizer, prompt)

            # truncating the chunks if necessary
            if prompt_tokens > max_input_tokens:
                rag_context = truncate_to_tokens(
                    decoder_tokenizer,
                    rag_context,
                    rag_context_token_budget,
                )
                if rag_context is None:
                    print("run_email_agent: skipping email because RAG context truncation failed")
                    continue
                try:
                    prompt = build_email_writer_prompt(
                        prompt_template=prompt_template,
                        my_name=my_name,
                        my_description=my_description,
                        subject=original_subject,
                        sender=original_sender,
                        body=original_body_for_prompt,
                        thread_context=thread_context,
                        rag_context=rag_context,
                    )
                except KeyError as e:
                    print(f"run_email_agent: error formatting email writer prompt template: {e}")
                    continue
                prompt_tokens = count_tokens(decoder_tokenizer, prompt)

            # and finally the body if necessary
            if prompt_tokens > max_input_tokens:
                original_body_for_prompt = truncate_to_tokens(
                    decoder_tokenizer,
                    original_body_for_prompt,
                    body_token_budget,
                )
                if original_body_for_prompt is None:
                    print("run_email_agent: skipping email because body truncation failed")
                    continue
                try:
                    prompt = build_email_writer_prompt(
                        prompt_template=prompt_template,
                        my_name=my_name,
                        my_description=my_description,
                        subject=original_subject,
                        sender=original_sender,
                        body=original_body_for_prompt,
                        thread_context=thread_context,
                        rag_context=rag_context,
                    )
                except KeyError as e:
                    print(f"run_email_agent: error formatting email writer prompt template: {e}")
                    continue

        # then, after body, thread, chunks are ready to fit:
        # run decoder (without "template" in email_writer_profile_config)
        try:
            proposed_reply, prompt_text = run_local_lm_or_vlm.remote(
                context=[],
                current_turn_input_text=prompt,
                current_turn_image_in_bytes=None,
                **email_writer_profile_config
            )
        except Exception as e:
            print(f"run_email_agent: decoder generation failed: {e}")
            continue

        if MODEL_PROFILES[EMAIL_WRITER_PROFILE]["return_prompt_text"]:
            print(f"{prompt_text}\n\n")

        # if LM thinks it does not have enough info to answer or fails to use <message>...</message>, skip email reply
        if proposed_reply is None:
            continue

        # format reply quoting original inquiry and append it to reply bodies list
        reply_body = format_response_quoting_original_body(proposed_reply, original_body)
        reply_bodies.append(reply_body)
        processed_email_ids.append(email["id"])

        # append subject to subjects list
        original_subjects.append(original_subject)
        
        # set recipient email and append it to recipient emails list
        recipient_email = smtp_email if (SEND_TO_SELF and not SAVE_AS_DRAFT) else original_sender
        recipient_emails.append(recipient_email)

    # save drafts
    if SAVE_AS_DRAFT:
        success, error = save_drafts(
            reply_bodies=reply_bodies,
            original_subjects=original_subjects,
            imap_email=imap_email,
            smtp_email=smtp_email,
            recipient_emails=recipient_emails,
            password=password,
            imap_server=imap_server,
            imap_port=int(imap_port_str),
            drafts_folder=DRAFTS_FOLDER
        )
        action_performed = f"saved {len(original_subjects)} drafts"
    # or send replies
    else:
        success, error = send_emails(
            reply_bodies=reply_bodies,
            original_subjects=original_subjects,
            smtp_email=smtp_email,
            recipient_emails=recipient_emails,
            password=password,
            smtp_server=smtp_server,
            smtp_port=int(smtp_port_str)
        )
        action_performed = f"sent {len(original_subjects)} emails"

    if success:
        print(f"run_email_agent: {action_performed} successfully")
        if not LEAVE_UNREAD and processed_email_ids:
            mark_success, mark_error = mark_emails_as_read(
                email_ids=processed_email_ids,
                imap_email=imap_email,
                password=password,
                imap_server=imap_server,
                imap_port=int(imap_port_str)
            )
            if mark_success:
                print(f"run_email_agent: marked {len(processed_email_ids)} emails as read")
            else:
                print(f"run_email_agent: emails were replied to, but marking as read failed: {mark_error}")
    else:
        print(f"run_email_agent: error: {error}")
