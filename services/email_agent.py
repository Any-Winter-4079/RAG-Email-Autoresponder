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
    from transformers import AutoTokenizer
    from helpers.decoder import count_tokens, truncate_to_tokens

    from helpers.data import (
        assign_thread_ids_by_subject_and_participant_overlap_for_production
    )
    from config.decoder import MODEL_PROFILES, EMAIL_WRITER_PROFILE
    from config.crawler_agent import CRAWL_DAY, CRAWL_MONTH
    from config.email_agent import (
        MAX_EMAILS,
        EMAIL_WRITER_BODY_TOKEN_RATIO,
        EMAIL_WRITER_THREAD_CONTEXT_TOKEN_RATIO,
        EMAIL_WRITER_RAG_CONTEXT_TOKEN_RATIO,
        INBOX_FOLDER,
        SENT_FOLDER,
        UNREAD_ONLY,
        LEAVE_UNREAD,
        LAST_N_DAYS,
        SEND_TO_SELF,
        SAVE_AS_DRAFT,
        DRAFTS_FOLDER
    )
    from helpers.email_agent import (
        transform_env_csv_into_list,
        read_latest_emails,
        format_response_quoting_original_body,
        compact_email_body_for_decoder,
        send_emails,
        save_drafts,
        mark_emails_as_read
    )

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
    body_token_budget = max(1, int(max_input_tokens * EMAIL_WRITER_BODY_TOKEN_RATIO))
    thread_context_token_budget = max(
        1,
        int(max_input_tokens * EMAIL_WRITER_THREAD_CONTEXT_TOKEN_RATIO),
    )
    rag_context_token_budget = max(
        1,
        int(max_input_tokens * EMAIL_WRITER_RAG_CONTEXT_TOKEN_RATIO),
    )
    print(
        "run_email_agent: email writer token budgets: "
        f"input={max_input_tokens:,} | "
        f"body={body_token_budget:,} | "
        f"thread_context={thread_context_token_budget:,} | "
        f"rag_context={rag_context_token_budget:,}"
    )

    # get decoder tokenizer 
    decoder_path = email_writer_profile_config["model_name_or_path"]
    decoder_tokenizer = AutoTokenizer.from_pretrained(decoder_path, trust_remote_code=True)

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
        # if we are the author (message to self), skip
        # if any(email in original_sender.lower() for email in my_email_addresses):
        #     print(f"run_email_agent: skipping email because sender is one of {', '.join(my_email_addresses)}")
        #     continue
        print(f"run_email_agent: generating reply for '{original_subject}' from {original_sender}")

        # get thread context emails for this email
        email_id = email.get("id")
        thread_id = email_id_to_thread_id.get(email_id)
        thread_context_emails = [
            context_email
            for context_email in thread_id_to_emails.get(thread_id, [])
            if context_email.get("id") != email_id
        ]
        # sort from oldest to newest
        thread_context_emails = sorted(
            thread_context_emails,
            key=lambda context_email: context_email.get("date") or datetime.min,
        )
        has_prior_thread_context = bool(thread_context_emails)

        # get unquoted/quoted body text for the current email and truncate them
        original_body_compacted = compact_email_body_for_decoder(
            decoder_tokenizer,
            original_body,
            body_token_budget,
            0 if has_prior_thread_context else body_token_budget,
            "[text omitted: body missing]",
            unquoted_fail_placeholder=None,
            quoted_fail_placeholder="[quoted text omitted: tokenization failed]",
            log_prefix="run_email_agent: current email"
        )
        if original_body_compacted is None:
            continue

        # make sure base prompt fits, clamp email if needed
        try:
            base_prompt = prompt_template.format(
                my_name=my_name,
                my_description=my_description,
                subject=original_subject,
                sender=original_sender,
                body="",
                thread_context="",
                rag_context=""
            )
        except KeyError as e:
            print(f"run_email_agent: error formatting email writer prompt template (without body): {e}")
            continue
        n_base_prompt_tokens = count_tokens(decoder_tokenizer, base_prompt)
        if n_base_prompt_tokens > max_input_tokens:
            print("run_email_agent: skipping email because base prompt exceeds input token budget")
            continue
        n_body_tokens = count_tokens(decoder_tokenizer, original_body_compacted)
        if n_body_tokens > body_token_budget:
            original_body_compacted = truncate_to_tokens(
                decoder_tokenizer,
                original_body_compacted,
                body_token_budget
            )
            if original_body_compacted is None:
                print("run_email_agent: skipping email because body truncation failed")
                continue

        # build prompt with email body
        try:
            prompt = prompt_template.format(
                my_name=my_name,
                my_description=my_description,
                subject=original_subject,
                sender=original_sender,
                body=original_body_compacted,
                thread_context="",
                rag_context=""
            )
        except KeyError as e:
            print(f"run_email_agent: error formatting email writer prompt template (with body): {e}")
            continue
        prompt_tokens = count_tokens(decoder_tokenizer, prompt)
        if prompt_tokens > max_input_tokens:
            print("run_email_agent: skipping email because base prompt exceeds input token budget")
            continue

        # add reconstructed context emails if they fit; quoted content is stripped
        # to avoid recursively duplicating earlier messages
        thread_context = ""
        message_separator = "\n[END MESSAGE]\n"
        for context_email in thread_context_emails:
            context_email_body = context_email.get("message_body")
            if not context_email_body:
                print("run_email_agent: skipping context email because body is missing")
                continue
            context_email_body = compact_email_body_for_decoder(
                decoder_tokenizer,
                context_email_body,
                thread_context_token_budget,
                0,
                "[text omitted: body missing]",
                unquoted_fail_placeholder="[text omitted: tokenization failed]",
                quoted_fail_placeholder="[quoted text omitted: tokenization failed]",
                log_prefix="run_email_agent: context email"
            )
            if context_email_body is None:
                continue
            context_email_from = (context_email.get("from") or "").strip()
            context_email_to = (context_email.get("to") or "").strip()
            context_email_subject = (context_email.get("subject") or "").strip()
            context_email_date = context_email.get("date")
            context_email_date_text = str(context_email_date) if context_email_date else ""
            block_header = (
                "From: " + context_email_from + "\n"
                "To: " + context_email_to + "\n"
                "Date: " + context_email_date_text + "\n"
                "Subject: " + context_email_subject + "\n"
                "Body:\n"
            )
            candidate_context = f"{thread_context}\n{block_header}{context_email_body}{message_separator}".strip()
            candidate_tokens = count_tokens(decoder_tokenizer, candidate_context)
            if candidate_tokens > thread_context_token_budget:
                if not thread_context:
                    candidate_context = truncate_to_tokens(
                        decoder_tokenizer,
                        candidate_context,
                        thread_context_token_budget
                    )
                    if candidate_context is not None:
                        thread_context = candidate_context
                break
            thread_context = candidate_context

        if not thread_context:
            thread_context = "(no prior messages found)"

        # construct prompt
        try:
            prompt = prompt_template.format(
                my_name=my_name,
                my_description=my_description,
                subject=original_subject,
                sender=original_sender,
                body=original_body_compacted,
                thread_context=thread_context,
                rag_context=""
            )
        except KeyError as e:
            print(f"run_email_agent: error formatting email writer prompt template (with body and context): {e}")
            continue
        prompt_tokens = count_tokens(decoder_tokenizer, prompt)
        if prompt_tokens > max_input_tokens:
            print("run_email_agent: skipping email because prompt exceeds input token budget")
            continue

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
