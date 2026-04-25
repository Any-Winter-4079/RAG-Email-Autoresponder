#######################################################################
# Helper 1: Extract email addresses from one or two participant texts #
#######################################################################
def extract_emails_from_participant_raw_texts(
        participant_1_raw_text,
        participant_2_raw_text=None,
        ):
    import re

    participant_emails = set()
    for text in [participant_1_raw_text, participant_2_raw_text]:
        if not text:
            continue
        for email in re.findall(r'[\w\.-]+@[\w\.-]+', text.lower()):
            participant_emails.add(email)
    return participant_emails

#############################################################################
# Helper 2: Check whether all email participants are internal UPM personnel #
#############################################################################
def is_upm_internal(author, recipients, upm_domains):
    # if all participants (author and recipients) aren't students / external people (e.g., they are professors):
    # return True to remove from the dataset
    all_emails = list(extract_emails_from_participant_raw_texts(author, recipients))
    return all(any(f"@{domain}" in email for domain in upm_domains) for email in all_emails) and len(all_emails) > 0

#####################################
# Helper 3: Normalize email subject #
#####################################
def normalize_subject(subject):
    # convert to lowercase and remove reply/forward prefixes
    normalized = subject.strip().lower()
    prefixes = ["re:", "fw:", "fwd:"]
    prefix_removed = True
    while prefix_removed:
        prefix_removed = False
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):].strip()
                prefix_removed = True
    return normalized

#################################################
# Helper 4: Normalize email body for comparison #
#################################################
def normalize_email_body(body):
    return body.replace("\n", " ").replace("\r", " ").strip().lower()

##################################################################
# Helper 5: Split an email body into unquoted and quoted content #
##################################################################
def get_unquoted_text(body, return_quoted=False):
    import re
    # get 1st match
    match = re.search(
        r"\s*(en .*?\b\d{4}\b.* escribió:|en .*?<[^>]*@[^>]*>.* escribió:|on .*?\b\d{4}\b.* wrote[:：]|on .*?<[^>]*@[^>]*>.* wrote[:：]|de:.*<[^>]*@[^>]*>.*enviado.*para:.*<[^>]*@[^>]*>.*asunto:|de:.*enviado:.*para:.*asunto:)",
        body.lower(),
        flags=re.IGNORECASE | re.DOTALL
    )
    if match:
        unquoted = body[:match.start()]
        quoted = body[match.start():]
    else:
        unquoted = body
        quoted = ""
    unquoted = unquoted.strip()
    quoted = quoted.strip()
    return (unquoted, quoted) if return_quoted else unquoted

####################################################################
# Helper 6: Detect template text in the unquoted portion of a body #
####################################################################
def has_template_in_unquoted(body, templates):
    unquoted = normalize_email_body(get_unquoted_text(body))
    return any(template in unquoted for template in templates)

####################################################################
# Helper 7: Count folderURI values and print stage-by-stage deltas #
####################################################################
def get_and_print_folder_uri_counts(folder_uri_column, title, previous_counts=None):
    from collections import Counter

    folder_uri_counts = Counter(folder_uri_column)
    print(f"\nget_and_print_folder_uri_counts: {title}")
    # print Alumnos/Seminarios/... counts from folderURI
    for uri, count in folder_uri_counts.items():
        diff_text = ""
        if previous_counts is not None:
            diff = count - previous_counts.get(uri, 0)
            diff_text = " (==)" if diff == 0 else f" ({diff:+d})"
        print(
            "get_and_print_folder_uri_counts: "
            f"{uri.split('/')[-1]}: {count} messages{diff_text}"
        )
    total_count = sum(folder_uri_counts.values())
    total_diff_text = ""
    if previous_counts is not None:
        total_diff = total_count - sum(previous_counts.values())
        total_diff_text = " (==)" if total_diff == 0 else f" ({total_diff:+d})"
    print(
        "get_and_print_folder_uri_counts: "
        f"Total: {total_count} messages{total_diff_text}"
    )
    return folder_uri_counts

###########################################################################
# Helper 8: Assign dataset thread IDs via subject and participant overlap #
###########################################################################
def assign_thread_ids_by_subject_and_participant_overlap_for_dataset(rows, my_email_addresses, n_lookback_window_rows):
    folder_uri_index = rows[0].index("folderURI")
    subject_index = rows[0].index("c1subject")
    body_index = rows[0].index("c0body")
    author_index = rows[0].index("c3author")
    recipients_index = rows[0].index("c4recipients")

    my_email_addresses = set(
        email.lower()
        for email in my_email_addresses
    )

    thread_id_to_metadata = {}
    row_thread_ids = []
    next_thread_id = 1

    for row_index, row in enumerate(rows[1:]):
        folder_uri = row[folder_uri_index]
        subject = row[subject_index]
        normalized_subject = normalize_subject(subject)
        author = row[author_index]
        recipients = row[recipients_index]
        all_participants = extract_emails_from_participant_raw_texts(author, recipients)
        participants = {email for email in all_participants if email not in my_email_addresses}

        candidate_thread_ids = []
        for thread_id, metadata in thread_id_to_metadata.items():
            if metadata["folder_uri"] != folder_uri:
                continue
            if metadata["normalized_subject"] != normalized_subject:
                continue
            if row_index - metadata["last_row_index"] > n_lookback_window_rows:
                continue
            if participants and metadata["participants"] and not participants.intersection(metadata["participants"]):
                continue
            candidate_thread_ids.append(thread_id)

        if candidate_thread_ids:
            best_thread_id = max(
                candidate_thread_ids,
                key=lambda thread_id: (
                    len(participants.intersection(thread_id_to_metadata[thread_id]["participants"])),
                    thread_id_to_metadata[thread_id]["last_row_index"],
                ),
            )
            thread_id_to_metadata[best_thread_id]["participants"].update(participants)
            thread_id_to_metadata[best_thread_id]["last_row_index"] = row_index
            row_thread_ids.append(best_thread_id)
        else:
            thread_id = next_thread_id
            next_thread_id += 1
            thread_id_to_metadata[thread_id] = {
                "folder_uri": folder_uri,
                "normalized_subject": normalized_subject,
                "participants": set(participants),
                "last_row_index": row_index,
            }
            row_thread_ids.append(thread_id)

    ordered_thread_ids = []
    thread_id_to_rows = {}
    for row, thread_id in zip(rows[1:], row_thread_ids):
        if thread_id not in thread_id_to_rows:
            ordered_thread_ids.append(thread_id)
            thread_id_to_rows[thread_id] = []
        thread_id_to_rows[thread_id].append(row)

    threads = []
    for thread_id in ordered_thread_ids:
        thread_rows = thread_id_to_rows[thread_id]
        threads.append({
            "folder_uri": thread_id_to_metadata[thread_id]["folder_uri"],
            "thread_id": thread_id,
            "thread_size": len(thread_rows),
            "emails": [
                {
                    "subject": row[subject_index],
                    "body": row[body_index],
                    "author": row[author_index],
                    "recipients": row[recipients_index],
                }
                for row in thread_rows
            ],
        })

    return threads

#########################################################
# Helper 9: Reconstruct dataset thread IDs with decoder #
#########################################################
def assign_thread_ids_with_decoder_for_dataset(
        threads,
        run_thread_grouper,
        thread_grouper_model_config,
        task_description_start,
        example_message,
        prompt_template,
        max_emails_per_batch,
        max_rule_based_threads_per_batch,
        max_concurrent_batches,
        max_input_tokens,
        pre_decoder_statistics_plot_path=None,
        post_decoder_statistics_plot_path=None,
        ):
    import asyncio
    from transformers import AutoProcessor, AutoTokenizer
    from config.decoder import GEMMA4_MODEL_FAMILY
    from helpers.decoder import count_tokens

    def build_prompt_from_batch_emails(batch_emails):
        prompt = prompt_template.format(
            task_description_start=task_description_start.format(
                email_count=len(batch_emails)
            ),
            example_message=example_message,
            emails_section="\n".join(str(email) for email in batch_emails),
        )
        return prompt

    # get tokenizer
    if (
        thread_grouper_model_config.get("is_vision_model")
        or thread_grouper_model_config.get("model_family") == GEMMA4_MODEL_FAMILY
    ):
        thread_grouper_processor = AutoProcessor.from_pretrained(
            thread_grouper_model_config["model_name_or_path"],
            trust_remote_code=True,
        )
        thread_grouper_tokenizer = thread_grouper_processor.tokenizer
    else:
        thread_grouper_tokenizer = AutoTokenizer.from_pretrained(
            thread_grouper_model_config["model_name_or_path"],
            trust_remote_code=True,
        )

    post_decoder_threads = [] # the final threads
    bypass_threads = [] # part of the final threads
    batches = [] # batches for the decoder, the result of which will complete the final threads; internally, per batch, keeps track of batch threads, and prompt for batch
    batch = [] # list of one or several threads
    batch_statistics = []
    bypass_statistics = {
        "email_limit": {"n_threads": 0, "n_emails": 0, "email_counts": []},
        "token_limit": {"n_threads": 0, "n_emails": 0, "prompt_tokens": []},
    }

    for thread in threads:
        #add threadID to each
        # since decoder expects threadID
        # on each email, not per group of emails
        thread_emails = [
            {
                **email,
                "threadID": thread["thread_id"],
            }
            for email in thread["emails"]
        ]
        thread_email_count = len(thread_emails)
        batch_prompt = build_prompt_from_batch_emails(thread_emails)
        n_thread_prompt_tokens =  count_tokens(
            thread_grouper_tokenizer,
            batch_prompt
        )

        # the thread can:
        # 1, bypass the decoder, due to its size in emails
        # 2, bypass the decoder, due to its size in tokens
        # 3, be added to a (growing) candidate batch, until
        # it no longer fits, then the batch is finalized
        # and a new one starts

        # if the current thread is too large (email count-wise) even for a single decoder batch
        if thread_email_count > max_emails_per_batch:
            # add it as a bypass
            bypass_threads.append(thread)
            # update statistics
            bypass_statistics["email_limit"]["n_threads"] += 1
            bypass_statistics["email_limit"]["n_emails"] += thread_email_count
            bypass_statistics["email_limit"]["email_counts"].append(
                thread_email_count
            )
            # move on to another thread
            continue

        # if the current thread is too large (token-wise) even for a single decoder batch
        if n_thread_prompt_tokens > max_input_tokens:
            # add it as a bypass
            bypass_threads.append(thread)
            # update statistics
            bypass_statistics["token_limit"]["n_threads"] += 1
            bypass_statistics["token_limit"]["n_emails"] += thread_email_count
            bypass_statistics["token_limit"]["prompt_tokens"].append(
                n_thread_prompt_tokens
            )
            # move on to another thread
            continue

        # if the current thread could fit in a batch, check if it can fit in the *current* batch:
        # - if it can, add it
        # - if it can't, check if the current batch is empty:
        #   - if it is, add thread to the batch
        #   - if not, finalize existing batch and open a new one with the current thread
        batch_emails = [email for thread in batch for email in thread["emails"]]
        n_hypothetical_batch_threads = len(batch) + 1
        n_hypothetical_batch_emails = len(batch_emails) + len(thread_emails)
        hypothetical_batch_prompt = build_prompt_from_batch_emails(batch_emails + thread_emails)
        n_hypothetical_batch_prompt_tokens = count_tokens(thread_grouper_tokenizer, hypothetical_batch_prompt)
        fits = (
                (thread["folder_uri"] == batch[-1]["folder_uri"] if len(batch) > 0 else True)
                and n_hypothetical_batch_emails <= max_emails_per_batch
                and n_hypothetical_batch_threads <= max_rule_based_threads_per_batch
                and n_hypothetical_batch_prompt_tokens <= max_input_tokens
            )
        if fits:
            # ensure threadID is per email
            thread["emails"] = thread_emails
            # then append thread
            batch.append(thread)
        else:
            if len(batch) == 0:
                # ensure threadID is per email
                thread["emails"] = thread_emails
                # then append thread
                batch = [thread]
            else:
                # build prompt since it includes the emails themselves
                batch_emails = [email for thread in batch for email in thread["emails"]]
                batch_prompt = build_prompt_from_batch_emails(batch_emails)
                # then finalize
                batches.append((batch, batch_prompt))
                # update statistics
                n_batch_prompt_tokens =  count_tokens(
                    thread_grouper_tokenizer,
                    batch_prompt
                )
                batch_statistics.append({
                    "n_emails": len(batch_emails),
                    "n_threads": len(batch),
                    "n_tokens": n_batch_prompt_tokens
                })
                # reset batch starting with the current thread, ensuring threadID is per email
                thread["emails"] = thread_emails
                batch = [thread]
        
    # close the final batch (if open)
    if batch:
        batch_emails = [email for thread in batch for email in thread["emails"]]
        batch_prompt = build_prompt_from_batch_emails(batch_emails)
        batches.append((batch, batch_prompt))
        # update statistics
        n_batch_prompt_tokens =  count_tokens(
            thread_grouper_tokenizer,
            batch_prompt
        )
        batch_statistics.append({
            "n_emails": len(batch_emails),
            "n_threads": len(batch),
            "n_tokens": n_batch_prompt_tokens
        })

    # plot batch/bypass summary
    if pre_decoder_statistics_plot_path:
        save_pre_decoder_statistics_plot(
            batch_statistics,
            bypass_statistics,
            title="Decoder batch vs bypass statistics",
            output_path=pre_decoder_statistics_plot_path,
            max_input_tokens=max_input_tokens,
        )
    
    async def run_batch(prompt):
        return await run_thread_grouper.remote.aio(
            context=[],
            current_turn_input_text=prompt,
            current_turn_image_in_bytes=None,
            **thread_grouper_model_config,
        )

    async def run_all_batches():
        if not batches:
            return []
        semaphore = asyncio.Semaphore(max_concurrent_batches)

        async def run_batch_with_semaphore(prompt):
            async with semaphore:
                return await run_batch(prompt)

        return await asyncio.gather(
            *(run_batch_with_semaphore(batch_prompt) for _, batch_prompt in batches),
            return_exceptions=True,
        )

    batch_results = asyncio.run(run_all_batches())

    decoder_statistics = {
        "n_failed_exception_batches": 0,
        "n_failed_oom_batches": 0,
        "n_failed_timeout_batches": 0,
        "n_failed_empty_output_batches": 0,
        "n_failed_short_output_batches": 0,
        "n_kept_exact_output_batches": 0,
        "n_kept_expanded_output_batches": 0,
    }
    # after running the decoder, construct
    # final threads

    # 1- add bypassed threads
    post_decoder_threads += bypass_threads

    # 2- add the decoder threads and/or update stats
    for batch, batch_result in zip(batches, batch_results):
        batch_threads = batch[0]
        batch_email_count = sum(len(batch_thread["emails"]) for batch_thread in batch_threads)

        if isinstance(batch_result, Exception):
            decoder_statistics["n_failed_exception_batches"] += 1
            batch_result_text = str(batch_result).lower()
            if "out of memory" in batch_result_text or "oom" in batch_result_text:
                decoder_statistics["n_failed_oom_batches"] += 1
            if "timeout" in batch_result_text or "timed out" in batch_result_text:
                decoder_statistics["n_failed_timeout_batches"] += 1
            continue

        parsed_threads, prompt_text = batch_result
        if not parsed_threads:
            decoder_statistics["n_failed_empty_output_batches"] += 1
            continue

        n_output_messages = sum(
            len(thread["messages"])
            for thread in parsed_threads
        )
        if n_output_messages < batch_email_count:
            decoder_statistics["n_failed_short_output_batches"] += 1
            continue
        if n_output_messages == batch_email_count:
            decoder_statistics["n_kept_exact_output_batches"] += 1
        else:
            decoder_statistics["n_kept_expanded_output_batches"] += 1

        # since each batch has some common folder_uri, we
        # can take any of them, here the last thread's
        folder_uri = batch_threads[-1]["folder_uri"]
        for thread in parsed_threads:
            post_decoder_threads.append({
                "folder_uri": folder_uri,
                "thread_size": len(thread["messages"]),
                "emails": [
                    {
                        "subject": message.get("subject"),
                        "body": message.get("body"),
                        "author": message.get("from"),
                        "recipients": message.get("to"),
                    }
                    for message in thread["messages"]
                ],
            })

    # after we have them all, add a unique id
    for thread_id, thread in enumerate(post_decoder_threads):
        thread["thread_id"] = thread_id

    # finally, plot post-decoder summary
    if post_decoder_statistics_plot_path:
        save_post_decoder_statistics_plot(
            n_rule_based_thread_count=len(threads),
            n_rule_based_threads_sent_to_lm=sum([len(batch[0]) for batch in batches]),
            n_planned_batches=len(batches),
            bypass_statistics=bypass_statistics,
            **decoder_statistics,
            title="Thread grouping post-decoder statistics",
            output_path=post_decoder_statistics_plot_path
        )

    return post_decoder_threads

#############################################
# Helper 10: Lighten a hex color with white #
#############################################
def lighten_hex_color(hex_color, mix_with_white=0.45):
    hex_color = hex_color.lstrip("#")
    red = int(hex_color[0:2], 16)
    green = int(hex_color[2:4], 16)
    blue = int(hex_color[4:6], 16)
    red = int(red + (255 - red) * mix_with_white)
    green = int(green + (255 - green) * mix_with_white)
    blue = int(blue + (255 - blue) * mix_with_white)
    return f"#{red:02X}{green:02X}{blue:02X}"

########################################################################
# Helper 11: Save thread-grouping pre-decoder statistics plot as image #
########################################################################
def save_pre_decoder_statistics_plot(
        batch_statistics,
        bypass_statistics,
        title,
        output_path,
        max_input_tokens=None,
        mode="thread_grouper",
        ):
    from collections import Counter
    from pathlib import Path
    import matplotlib.pyplot as plt

    if not batch_statistics and not bypass_statistics:
        return

    batch_email_counts = [batch_stat["n_emails"] for batch_stat in batch_statistics]
    batch_thread_counts = [batch_stat["n_threads"] for batch_stat in batch_statistics]
    batch_prompt_tokens = [batch_stat["n_tokens"] for batch_stat in batch_statistics]
    email_limit_bypass_statistics = bypass_statistics["email_limit"]
    token_limit_bypass_statistics = bypass_statistics["token_limit"]

    email_count_to_n_batches = Counter(batch_email_counts)
    thread_count_to_n_batches = Counter(batch_thread_counts)
    sorted_prompt_tokens = sorted(batch_prompt_tokens)
    token_x_values = list(range(1, len(sorted_prompt_tokens) + 1))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=2,
        figsize=(14, 9.5),
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )
    fig.suptitle(title, fontsize=17)
    axes = axes.flatten()

    email_bar_color = lighten_hex_color("#00CBBF")
    thread_bar_color = lighten_hex_color("#F46920")
    token_line_color = lighten_hex_color("#F53255")
    token_limit_color = lighten_hex_color("#4C566A")
    email_limit_bypass_color = lighten_hex_color("#FFAF00")
    token_limit_bypass_color = lighten_hex_color("#9DCA1C")

    email_x_values = sorted(email_count_to_n_batches.keys())
    email_y_values = [email_count_to_n_batches[n_emails] for n_emails in email_x_values]
    email_bars = axes[0].bar(
        email_x_values,
        email_y_values,
        width=0.85,
        color=email_bar_color,
        edgecolor="white",
        linewidth=0.8,
    )
    axes[0].set_title("Batches by email count")
    axes[0].set_xlabel("Emails per batch", fontsize=13)
    axes[0].set_ylabel("Number of batches", fontsize=13)
    axes[0].set_xticks(email_x_values)
    axes[0].tick_params(axis="both", labelsize=12)
    axes[0].grid(axis="y", alpha=0.25)
    for bar in email_bars:
        axes[0].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    thread_x_values = sorted(thread_count_to_n_batches.keys())
    thread_y_values = [thread_count_to_n_batches[n_threads] for n_threads in thread_x_values]
    thread_bars = axes[1].bar(
        thread_x_values,
        thread_y_values,
        width=0.85,
        color=thread_bar_color,
        edgecolor="white",
        linewidth=0.8,
    )
    if mode == "thread_grouper":
        axes[1].set_title("Batches by thread count")
        axes[1].set_xlabel("Rule-based threads per batch", fontsize=13)
    else:
        axes[1].set_title("Batches by thread-chunk count")
        axes[1].set_xlabel("Thread chunks per batch", fontsize=13)
    axes[1].set_ylabel("Number of batches", fontsize=13)
    axes[1].set_xticks(thread_x_values)
    axes[1].tick_params(axis="both", labelsize=12)
    axes[1].grid(axis="y", alpha=0.25)
    for bar in thread_bars:
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{int(bar.get_height())}",
            ha="center",
            va="bottom",
            fontsize=11,
        )

    if sorted_prompt_tokens:
        axes[2].plot(
            token_x_values,
            sorted_prompt_tokens,
            color=token_line_color,
            marker="o",
            linewidth=2.0,
            markersize=4.2,
        )
    if max_input_tokens is not None:
        axes[2].axhline(
            max_input_tokens,
            color=token_limit_color,
            linestyle="--",
            linewidth=1.5,
            label=f"Token limit ({max_input_tokens})",
        )
    if sorted_prompt_tokens or max_input_tokens is not None:
        axes[2].legend(fontsize=12)
    axes[2].set_title("Sorted prompt tokens per batch")
    axes[2].set_xlabel("Batch rank (sorted by tokens)", fontsize=13)
    axes[2].set_ylabel("Prompt tokens", fontsize=13)
    axes[2].tick_params(axis="both", labelsize=12)
    axes[2].grid(axis="y", alpha=0.25)

    bypass_reason_labels = [
        "Email limit",
        "Token limit",
    ]
    if mode == "thread_grouper":
        bypass_reason_thread_counts = [
            email_limit_bypass_statistics["n_threads"],
            token_limit_bypass_statistics["n_threads"],
        ]
    else:
        bypass_reason_thread_counts = [
            email_limit_bypass_statistics["n_events"],
            token_limit_bypass_statistics["n_events"],
        ]
    bypass_bars = axes[3].bar(
        bypass_reason_labels,
        bypass_reason_thread_counts,
        color=[email_limit_bypass_color, token_limit_bypass_color],
        edgecolor="white",
        linewidth=0.8,
    )
    if mode == "thread_grouper":
        axes[3].set_title("Bypassed threads by reason")
        axes[3].set_ylabel("Number of bypassed threads", fontsize=13)
    else:
        axes[3].set_title("Split events by reason")
        axes[3].set_ylabel("Number of split events", fontsize=13)
    axes[3].tick_params(axis="both", labelsize=12)
    axes[3].grid(axis="y", alpha=0.25)
    if mode == "thread_grouper":
        bypass_bar_label_texts = [
            f"{email_limit_bypass_statistics['n_threads']} threads\n{email_limit_bypass_statistics['n_emails']} emails",
            f"{token_limit_bypass_statistics['n_threads']} threads\n{token_limit_bypass_statistics['n_emails']} emails",
        ]
    else:
        bypass_bar_label_texts = [
            f"{email_limit_bypass_statistics['n_events']} events",
            f"{token_limit_bypass_statistics['n_events']} events",
        ]
    for bar, label_text in zip(bypass_bars, bypass_bar_label_texts):
        axes[3].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            label_text,
            ha="center",
            va="bottom",
            fontsize=11,
        )

    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"save_pre_decoder_statistics_plot: saved plot to {output_path}")

###########################################################################
# Helper 12: Save 3D folderURI dropped-message distribution plot as image #
###########################################################################
def save_folder_uri_drop_3d_plot(
        folder_uri_count_history,
        title,
        output_path,
        excluded_phase_labels=None,
        ):
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
    # noqa: F401 tells Ruff/Pyflakes-style linters to ignore the unused-import warning here
    # Importing Axes3D registers Matplotlib's "3d" projection type, which makes:
    # ax = fig.add_subplot(111, projection="3d")
    # work below
    from mpl_toolkits.mplot3d import Axes3D # noqa: F401

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if len(folder_uri_count_history) < 2:
        return

    excluded_phase_labels = set(excluded_phase_labels or [])

    folder_uris = sorted({
        folder_uri
        for _, folder_uri_counts in folder_uri_count_history
        for folder_uri in folder_uri_counts.keys()
    })
    folder_labels = [folder_uri.split("/")[-1] for folder_uri in folder_uris]
    x_values = list(range(len(folder_labels)))
    phase_labels = []
    polygons = []
    line_series = []
    for previous, current in zip(folder_uri_count_history, folder_uri_count_history[1:]):
        previous_label, previous_counts = previous
        current_label, current_counts = current
        del previous_label
        if current_label in excluded_phase_labels:
            continue
        drops = [
            previous_counts.get(folder_uri, 0) - current_counts.get(folder_uri, 0)
            for folder_uri in folder_uris
        ]
        phase_labels.append(current_label)
        line_series.append(drops)
        polygons.append([(x_values[0], 0)] + list(zip(x_values, drops)) + [(x_values[-1], 0)])

    phase_to_color = {
        "empty_body": "#FFAF00",
        "pre_enrollment": "#F46920",
        "admission_rejection": "#F53255",
        "duplicates": "#F857C1",
        "internal_upm_threads": "#29BDFD",
        "decoder_failed_batches": "#00CBBF",
    }
    default_color = "#4C566A"
    colors = [phase_to_color.get(phase_label, default_color) for phase_label in phase_labels]
    poly_collection = PolyCollection(polygons, facecolors=colors, alpha=0.33)

    fig = plt.figure(figsize=(16, 9))
    ax = fig.add_subplot(111, projection="3d")
    ax.add_collection3d(poly_collection, zs=range(len(phase_labels)), zdir="y")

    for phase_index, (_, drops, color) in enumerate(zip(phase_labels, line_series, colors)):
        ax.plot(
            x_values,
            [phase_index] * len(x_values),
            drops,
            color=color,
            marker="o",
            linewidth=2.0,
            markersize=5.0,
        )
        for x_value, drop in zip(x_values, drops):
            if drop > 0:
                ax.text(
                    x_value,
                    phase_index,
                    drop,
                    str(drop),
                    ha="center",
                    va="bottom",
                    fontsize=10,
                )

    ax.set_title(title, fontsize=16)
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("Dropped emails", fontsize=13)
    ax.set_xticks(x_values)
    ax.set_xticklabels(folder_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(phase_labels)))
    ax.set_yticklabels(phase_labels)
    ax.tick_params(axis="x", labelsize=12)
    ax.tick_params(axis="y", labelsize=12)
    ax.tick_params(axis="z", labelsize=12)
    ax.set_zlim(bottom=0)
    pane_face_color = (1.0, 1.0, 1.0, 0.96)
    pane_edge_color = (0.93, 0.93, 0.93, 1.0)
    grid_color = (0.88, 0.88, 0.88, 0.7)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.set_pane_color(pane_face_color)
        if hasattr(axis, "pane"):
            axis.pane.set_facecolor(pane_face_color)
            axis.pane.set_edgecolor(pane_edge_color)
        if hasattr(axis, "_axinfo") and "grid" in axis._axinfo:
            axis._axinfo["grid"]["color"] = grid_color
    ax.view_init(elev=25, azim=-50)
    fig.subplots_adjust(left=0.02, right=0.95, bottom=0.18, top=0.92)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"save_folder_uri_drop_3d_plot: saved 3D folder drop plot to {output_path}")

#########################################################################
# Helper 13: Save thread-grouping post-decoder statistics plot as image #
#########################################################################
def save_post_decoder_statistics_plot(
        n_rule_based_thread_count,
        n_rule_based_threads_sent_to_lm,
        n_planned_batches,
        bypass_statistics,
        n_failed_exception_batches,
        n_failed_oom_batches,
        n_failed_timeout_batches,
        n_failed_empty_output_batches,
        n_failed_short_output_batches,
        n_kept_exact_output_batches,
        n_kept_expanded_output_batches,
        title,
        output_path,
        mode="thread_grouper",
        n_failed_long_output_batches=0,
        n_split_threads=0,
        n_thread_chunks=0,
        ):
    from pathlib import Path
    import matplotlib.pyplot as plt

    failed_other_exception_batches = (
        n_failed_exception_batches
        - n_failed_oom_batches
        - n_failed_timeout_batches
    )
    if failed_other_exception_batches < 0:
        raise ValueError(
            "save_post_decoder_statistics_plot: invalid failure counts:\n"
            f"\tn_failed_exception_batches={n_failed_exception_batches}\n"
            f"\tn_failed_oom_batches={n_failed_oom_batches}\n"
            f"\tn_failed_timeout_batches={n_failed_timeout_batches}"
        )

    if mode == "thread_grouper":
        kept_batches = n_kept_exact_output_batches + n_kept_expanded_output_batches
        failed_batches = (
            n_failed_exception_batches
            + n_failed_short_output_batches
            + n_failed_empty_output_batches
        )
    else:
        kept_batches = n_kept_exact_output_batches
        failed_batches = (
            n_failed_exception_batches
            + n_failed_empty_output_batches
            + n_failed_short_output_batches
            + n_failed_long_output_batches
        )
    if kept_batches + failed_batches != n_planned_batches:
        raise ValueError(
            "save_post_decoder_statistics_plot: planned LM batch count mismatch:\n"
            f"\tn_planned_batches={n_planned_batches}\n"
            f"\tkept_batches={kept_batches}\n"
            f"\tfailed_batches={failed_batches}"
        )
    email_limit_bypass_statistics = bypass_statistics["email_limit"]
    token_limit_bypass_statistics = bypass_statistics["token_limit"]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(13.5, 8.5))
    grid = fig.add_gridspec(
        nrows=2,
        ncols=2,
        height_ratios=[1.15, 1.0],
    )
    outcome_ax = fig.add_subplot(grid[0, :])
    exception_ax = fig.add_subplot(grid[1, 0])
    discarded_ax = fig.add_subplot(grid[1, 1])
    fig.suptitle(title, fontsize=17)

    if mode == "thread_grouper":
        top_bar_specs = [(
            "Planned LM batches",
            [
            ("Kept: exact output", n_kept_exact_output_batches, lighten_hex_color("#2E8B57")),
            ("Kept: expanded output", n_kept_expanded_output_batches, lighten_hex_color("#8BC34A")),
            ("Exception", n_failed_exception_batches, lighten_hex_color("#F46920")),
            ("Discarded: short output", n_failed_short_output_batches, lighten_hex_color("#F53255")),
            ("Discarded: empty output", n_failed_empty_output_batches, lighten_hex_color("#7E57C2")),
            ],
        )]
        outcome_title = (
            "LM batch outcomes\n"
            f"(out of {n_planned_batches} planned batches)"
        )
        outcome_xlabel = "Batch count"
        max_outcome_count = n_planned_batches
    else:
        top_bar_specs = [(
            "Planned LM batches",
            [
            ("Kept: exact output", n_kept_exact_output_batches, lighten_hex_color("#2E8B57")),
            ("Exception", n_failed_exception_batches, lighten_hex_color("#F46920")),
            ("Empty output", n_failed_empty_output_batches, lighten_hex_color("#7E57C2")),
            ("Count mismatch", n_failed_short_output_batches + n_failed_long_output_batches, lighten_hex_color("#F53255")),
            ],
        )]
        outcome_title = (
            "LM batch outcomes\n"
            f"(out of {n_planned_batches} planned batches)"
        )
        outcome_xlabel = "Batch count"
        max_outcome_count = n_planned_batches

    legend_labels_used = set()
    y_positions = list(reversed(range(len(top_bar_specs))))
    y_labels = []
    for y_position, (row_label, outcome_segments) in zip(y_positions, top_bar_specs):
        y_labels.append(row_label)
        left = 0
        for label, value, color in outcome_segments:
            legend_label = None
            if label not in legend_labels_used:
                legend_label = f"{label} ({value})"
                legend_labels_used.add(label)
            outcome_ax.barh(
                y_position,
                value,
                left=left,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                label=legend_label,
            )
            if value > 0:
                x_position = left + (value / 2)
                outcome_ax.text(
                    x_position,
                    y_position,
                    str(value),
                    ha="center",
                    va="center",
                    color="white" if value >= 10 else "black",
                    fontweight="bold" if value >= 10 else None,
                    fontsize=11,
                    clip_on=True,
                )
            left += value

    outcome_ax.set_xlim(0, max(1, max_outcome_count))
    outcome_ax.set_yticks(y_positions, y_labels)
    outcome_ax.set_title(outcome_title)
    outcome_ax.set_xlabel(outcome_xlabel, fontsize=13)
    outcome_ax.tick_params(axis="both", labelsize=12)
    outcome_ax.grid(axis="x", alpha=0.25)
    outcome_ax.legend(loc="lower right", fontsize=12)

    exception_labels = ["OOM", "Timeout"]
    exception_values = [n_failed_oom_batches, n_failed_timeout_batches]
    exception_colors = [
        lighten_hex_color("#FF6B6B"),
        lighten_hex_color("#6A5ACD"),
    ]

    exception_bars = exception_ax.bar(
        exception_labels,
        exception_values,
        color=exception_colors,
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, value in zip(exception_bars, exception_values):
        exception_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            str(value),
            ha="center",
            va="bottom",
            fontsize=11,
        )
    exception_ax.set_title(
        "Exception breakdown\n"
        f"({n_failed_exception_batches} batches)"
    )
    exception_ax.set_ylabel("Batch count", fontsize=13)
    exception_ax.tick_params(axis="both", labelsize=12)
    exception_ax.grid(axis="y", alpha=0.25)

    if mode == "thread_grouper":
        discarded_labels = ["Short output", "Empty output"]
        discarded_values = [n_failed_short_output_batches, n_failed_empty_output_batches]
        discarded_colors = [
            lighten_hex_color("#F53255"),
            lighten_hex_color("#7E57C2"),
        ]
        discarded_ax.set_title(
            "Discarded output breakdown\n"
            f"({n_failed_short_output_batches + n_failed_empty_output_batches} batches)"
        )
    else:
        discarded_labels = ["Fewer outputs", "More outputs"]
        discarded_values = [n_failed_short_output_batches, n_failed_long_output_batches]
        discarded_colors = [
            lighten_hex_color("#F53255"),
            lighten_hex_color("#8BC34A"),
        ]
        discarded_ax.set_title(
            "Count mismatch breakdown\n"
            f"({n_failed_short_output_batches + n_failed_long_output_batches} batches)"
        )
    discarded_bars = discarded_ax.bar(
        discarded_labels,
        discarded_values,
        color=discarded_colors,
        edgecolor="white",
        linewidth=0.8,
    )
    for bar, value in zip(discarded_bars, discarded_values):
        discarded_ax.text(
            bar.get_x() + bar.get_width() / 2,
            value,
            str(value),
            ha="center",
            va="bottom",
            fontsize=11,
        )
    discarded_ax.tick_params(axis="both", labelsize=12)
    discarded_ax.grid(axis="y", alpha=0.25)

    shared_bottom_ymax = max(55, max(exception_values + discarded_values + [0]) + 3)
    exception_ax.set_ylim(0, shared_bottom_ymax)
    discarded_ax.set_ylim(0, shared_bottom_ymax)

    if mode == "thread_grouper":
        total_bypassed_threads = (
            email_limit_bypass_statistics["n_threads"]
            + token_limit_bypass_statistics["n_threads"]
        )
        total_bypassed_emails = (
            email_limit_bypass_statistics["n_emails"]
            + token_limit_bypass_statistics["n_emails"]
        )
        summary_text = (
            f"Rule-based threads total: {n_rule_based_thread_count}\n"
            f"Rule-based threads sent to LM: {n_rule_based_threads_sent_to_lm}\n"
            f"Rule-based threads bypassed before LM: {total_bypassed_threads} "
            f"covering {total_bypassed_emails} emails\n"
            f"Kept LM batches: {kept_batches}\n"
            f"Skipped LM batches: {failed_batches}"
        )
    else:
        summary_text = (
            f"Input threads total: {n_rule_based_thread_count}\n"
            f"Split threads: {n_split_threads}\n"
            f"Thread chunks sent to LM: {n_thread_chunks}"
        )
    fig.text(
        0.02,
        0.02,
        summary_text,
        ha="left",
        va="bottom",
        fontsize=11,
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#F5F5F5",
            "edgecolor": "#D0D0D0",
            "alpha": 0.95,
        },
    )

    fig.tight_layout(rect=(0, 0.10, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"save_post_decoder_statistics_plot: saved plot to {output_path}")

###################################################
# Helper 14: Save pie-chart distribution as image #
###################################################
def save_pie_chart_distribution_plot(
        labels,
        sizes,
        title,
        output_path,
        preferred_label_order=None,
        label_to_color=None,
        ):
    from pathlib import Path
    import matplotlib.pyplot as plt

    preferred_label_order = list(preferred_label_order or [])
    label_to_color = dict(label_to_color or {})
    fallback_pie_colors = [
        "#FFAF00",
        "#F46920",
        "#F857C1",
        "#F53255",
        "#29BDFD",
        "#00CBBF",
        "#01C159",
        "#9DCA1C",
    ]
    label_to_size = dict(zip(labels, sizes))
    ordered_labels = [label for label in preferred_label_order if label in label_to_size]
    ordered_labels.extend(label for label in labels if label not in ordered_labels)
    labels = ordered_labels
    sizes = [label_to_size[label] for label in labels]
    used_colors = {
        label_to_color[label]
        for label in labels
        if label in label_to_color
    }
    remaining_fallback_colors = [
        color
        for color in fallback_pie_colors
        if color not in used_colors
    ]
    pie_colors = []
    fallback_color_index = 0
    for label in labels:
        if label in label_to_color:
            pie_colors.append(lighten_hex_color(label_to_color[label]))
            continue
        pie_colors.append(
            lighten_hex_color(
                remaining_fallback_colors[
                    fallback_color_index % len(remaining_fallback_colors)
                ]
            )
        )
        fallback_color_index += 1
    total_size = sum(sizes)
    if total_size > 0:
        startangle = 90 + 180 * (sizes[0] / total_size)
    else:
        startangle = 90
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, *_ = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%', # one decimal place
        colors=pie_colors,
        counterclock=False,
        startangle=startangle,
        labeldistance=1.03,
        pctdistance=0.62,
        textprops={"fontsize": 13},
    )
    ax.set_title(title, fontsize=16, pad=26)
    ax.axis("equal")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

###########################################################
# Helper 15: Save stacked size-distribution plot as image #
###########################################################
def save_stacked_size_distribution_plot(
        size_counts,
        inbound_email_counts,
        outbound_email_counts,
        title,
        x_label,
        y_label,
        output_path,
        ):
    from pathlib import Path
    import matplotlib.pyplot as plt

    if not size_counts:
        return

    x_values = sorted(size_counts.keys())
    count_values = [size_counts[size] for size in x_values]
    inbound_bar_heights = []
    outbound_bar_heights = []
    for size, count_value in zip(x_values, count_values):
        inbound_email_count = inbound_email_counts[size]
        outbound_email_count = outbound_email_counts[size]
        total_email_count = inbound_email_count + outbound_email_count
        inbound_ratio = (
            inbound_email_count / total_email_count
            if total_email_count
            else 0
        )
        outbound_ratio = (
            outbound_email_count / total_email_count
            if total_email_count
            else 0
        )
        inbound_bar_heights.append(count_value * inbound_ratio)
        outbound_bar_heights.append(count_value * outbound_ratio)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(14, 6.5))
    inbound_color = lighten_hex_color("#F46920")
    outbound_color = lighten_hex_color("#F53255")
    ax.bar(
        x_values,
        inbound_bar_heights,
        width=0.85,
        color=inbound_color,
        edgecolor="white",
        linewidth=0.6,
        label="Incoming share",
    )
    bars = ax.bar(
        x_values,
        outbound_bar_heights,
        width=0.85,
        bottom=inbound_bar_heights,
        color=outbound_color,
        edgecolor="white",
        linewidth=0.6,
        label="Outgoing share",
    )
    ax.set_title(title, fontsize=17, pad=14)
    ax.set_xlabel(x_label, fontsize=13)
    ax.set_ylabel(y_label, fontsize=13)
    ax.set_xticks(x_values)
    ax.tick_params(axis="both", labelsize=12)
    ax.legend(fontsize=12)
    for bar, count_value in zip(bars, count_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count_value,
            f"{count_value}",
            ha="center",
            va="bottom",
            fontsize=11,
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

################################################
# Helper 16: Remove fully internal UPM threads #
################################################
def remove_internal_upm_threads(threads, upm_domains):
    kept_threads = []
    discarded_threads = []

    for thread in threads:
        if thread["emails"] and all(
            is_upm_internal(
                email["author"],
                email["recipients"],
                upm_domains,
            )
            for email in thread["emails"]
        ):
            discarded_threads.append(thread)
            continue
        kept_threads.append(thread)

    return kept_threads, discarded_threads

##################################################################
# Helper 17: Build email samples grouped by thread and folderURI #
##################################################################
def build_samples_grouped_by_thread_and_folderURI(
        threads,
        my_email_addresses,
        ):
    my_email_addresses = set(
        email.lower()
        for email in (my_email_addresses or [])
        if email
    )

    folder_uri_to_thread_samples_groups = {}
    for thread in threads:
        folder_uri = thread["folder_uri"]
        thread_id = thread["thread_id"]
        thread_emails = thread["emails"]
        thread_size = thread["thread_size"]

        thread_samples = []
        for email_index, email in enumerate(thread_emails):
            email_author_emails = extract_emails_from_participant_raw_texts(
                email["author"]
            )
            email_recipient_emails = extract_emails_from_participant_raw_texts(
                email["recipients"]
            )
            is_message_to_self_from_director_pov = (
                bool(email_author_emails)
                and
                bool(email_recipient_emails)
                and email_author_emails.issubset(my_email_addresses)
                and email_recipient_emails.issubset(my_email_addresses)
            )
            if is_message_to_self_from_director_pov:
                continue

            context_emails = thread_emails[:email_index]
            later_emails = thread_emails[email_index + 1:]
            other_gold_reply_candidates = [
                later_email
                for later_email in later_emails
                if bool(
                    extract_emails_from_participant_raw_texts(
                        later_email["author"]
                    ).intersection(my_email_addresses)
                )
            ]

            gold_reply = None
            if other_gold_reply_candidates:
                for candidate in other_gold_reply_candidates:
                    candidate_recipient_emails = extract_emails_from_participant_raw_texts(
                        candidate["recipients"]
                    )
                    if email_author_emails.intersection(candidate_recipient_emails):
                        gold_reply = candidate
                        break
                if gold_reply is None:
                    gold_reply = other_gold_reply_candidates[0]

            thread_samples.append({
                "folder_uri": folder_uri,
                "thread_id": thread_id,
                "email": email,
                "context_emails": context_emails,
                "gold_reply": gold_reply,
                "other_gold_reply_candidates": [
                    candidate
                    for candidate in other_gold_reply_candidates
                    if candidate != gold_reply
                ],
                "thread_size": thread_size,
            })

        if not thread_samples:
            continue

        if folder_uri not in folder_uri_to_thread_samples_groups:
            folder_uri_to_thread_samples_groups[folder_uri] = []
        folder_uri_to_thread_samples_groups[folder_uri].append(thread_samples)

    return folder_uri_to_thread_samples_groups

###################################################################
# Helper 18: Split grouped samples into train, dev, and test sets #
###################################################################
def split_samples_by_split_name(folder_uri_to_thread_samples_groups, train_split_pct, dev_split_pct, seed):
    import random

    rng = random.Random(seed)
    split_names = ["train", "dev", "test"]
    split_name_to_samples = {
        split_name: []
        for split_name in split_names
    }

    for folder_uri, thread_samples_groups in folder_uri_to_thread_samples_groups.items():
        n_samples_in_folder = sum(
            len(thread_samples)
            for thread_samples in thread_samples_groups
        )
        split_name_to_n_samples_goal_in_folder = {
            "train": int(train_split_pct * n_samples_in_folder),
            "dev": int(dev_split_pct * n_samples_in_folder),
        }
        split_name_to_n_samples_goal_in_folder["test"] = (
            n_samples_in_folder
            - split_name_to_n_samples_goal_in_folder["train"]
            - split_name_to_n_samples_goal_in_folder["dev"]
        )
        split_name_to_n_samples_assigned_in_folder = {
            split_name: 0
            for split_name in split_names
        }

        thread_samples_groups = list(thread_samples_groups)
        rng.shuffle(thread_samples_groups)

        for thread_samples in thread_samples_groups:
            n_samples_in_thread = len(thread_samples)
            split_names_that_fit = [
                split_name
                for split_name in split_names
                if (
                    split_name_to_n_samples_assigned_in_folder[split_name]
                    + n_samples_in_thread
                ) <= split_name_to_n_samples_goal_in_folder[split_name]
            ]

            if split_names_that_fit:
                best_split_name = max(
                    split_names_that_fit,
                    key=lambda split_name: (
                        split_name_to_n_samples_goal_in_folder[split_name]
                        - split_name_to_n_samples_assigned_in_folder[split_name]
                    ),
                )
            else:
                best_split_name = min(
                    split_names,
                    key=lambda split_name: (
                        split_name_to_n_samples_assigned_in_folder[split_name]
                        + n_samples_in_thread
                        - split_name_to_n_samples_goal_in_folder[split_name]
                    ),
                )
            split_name_to_n_samples_assigned_in_folder[best_split_name] += n_samples_in_thread
            split_name_to_samples[best_split_name].extend(thread_samples)

    return split_name_to_samples

###############################################
# Helper 19: Save split-summary plot as image #
###############################################
def save_split_summary_plot(
        split_name_to_sample_count,
        split_name_to_thread_count,
        title,
        output_path,
        ):
    from pathlib import Path
    import matplotlib.pyplot as plt

    split_names = ["train", "dev", "test"]
    sample_counts = [split_name_to_sample_count[split_name] for split_name in split_names]
    thread_counts = [split_name_to_thread_count[split_name] for split_name in split_names]
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(
        ncols=2,
        figsize=(12.8, 5.0),
        sharey=False,
    )
    fig.suptitle(title, fontsize=16, y=0.93)

    x_values = list(range(len(split_names)))
    sample_color = lighten_hex_color("#00CBBF")
    thread_color = lighten_hex_color("#F46920")

    sample_bars = axes[0].bar(
        x_values,
        sample_counts,
        width=0.70,
        color=sample_color,
        edgecolor="white",
        linewidth=0.8,
    )
    thread_bars = axes[1].bar(
        x_values,
        thread_counts,
        width=0.70,
        color=thread_color,
        edgecolor="white",
        linewidth=0.8,
    )
    axes[0].set_title("Samples per split", fontsize=15, pad=10)
    axes[1].set_title("Threads per split", fontsize=15, pad=10)
    axes[0].set_ylabel("Count", fontsize=13)

    for ax in axes:
        ax.set_xticks(x_values)
        ax.set_xticklabels(split_names)
        ax.tick_params(axis="both", labelsize=12)
        ax.grid(axis="y", alpha=0.25)

    for bars, ax in [(sample_bars, axes[0]), (thread_bars, axes[1])]:
        for bar in bars:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{int(bar.get_height())}",
                ha="center",
                va="bottom",
                fontsize=11,
            )

    fig.tight_layout(rect=(0, 0, 1, 0.925), w_pad=1.25)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"save_split_summary_plot: saved plot to {output_path}")

######################################################
# Helper 20: Format email as single promptable block #
######################################################
def format_email_prompt_block(
        email,
        author_key="author",
        recipients_key="recipients",
        subject_key="subject",
        body_key="body",
        date_key=None,
        wrap_with_dashes=False,
        ):
    subject = (email.get(subject_key) or "").strip()
    body = (email.get(body_key) or "").strip()
    author = (email.get(author_key) or "").strip()
    recipients = (email.get(recipients_key) or "").strip()

    lines = [
        f"From: {author}",
        f"To: {recipients}",
    ]
    if date_key:
        date = email.get(date_key)
        date_text = str(date) if date else ""
        lines.append(f"Date: {date_text}")
    lines.extend([
        f"Subject: {subject}",
        "Body:",
        body,
    ])
    block = "\n".join(lines).strip()
    if wrap_with_dashes:
        block = f"-----\n{block}\n-----"
    return block

######################################################################
# Helper 21: Format grouped thread emails as single promptable block #
######################################################################
def format_email_thread_text(emails):
    return "\n".join(
        format_email_prompt_block(
            email,
            wrap_with_dashes=True,
        )
        for email in emails
    )

##########################################################
# Helper 22: Prepare encode batches for one data variant #
##########################################################
def prepare_batches_for_data_variant(
        variant,
        records,
        batch_size,
        encode_timestamp,
        ):
    base_variant = variant.removeprefix("email_")

    batches = []
    current_batch = {
        "texts": [],
        "payloads": [],
        "point_ids": [],
    }
    next_point_id = 0

    for record in records:
        if base_variant in ["lm_q_and_a_chunks", "lm_q_and_a_for_q_only_chunks"]:
            for pair_index, pair in enumerate(record["pairs"], start=1):
                if base_variant == "lm_q_and_a_chunks":
                    text = f"Q: {pair['question']}\nA: {pair['answer']}"
                else:
                    text = pair["question"]

                payload = {
                    **record,
                    "variant": variant,
                    "timestamp": encode_timestamp,
                }
                payload.pop("pairs", None)
                payload.update({
                    "pair_index": pair_index,
                    "question": pair["question"],
                    "answer": pair["answer"],
                    "decoder_token_count_q": pair["decoder_token_count_q"],
                    "encoder_token_count_q": pair["encoder_token_count_q"],
                    "decoder_token_count_a": pair["decoder_token_count_a"],
                    "encoder_token_count_a": pair["encoder_token_count_a"],
                })

                current_batch["texts"].append(text)
                current_batch["payloads"].append(payload)
                current_batch["point_ids"].append(next_point_id)
                next_point_id += 1

                if len(current_batch["texts"]) == batch_size:
                    batches.append(current_batch)
                    current_batch = {
                        "texts": [],
                        "payloads": [],
                        "point_ids": [],
                    }
        else:
            current_batch["texts"].append(record["text"])
            current_batch["payloads"].append({
                **record,
                "variant": variant,
                "timestamp": encode_timestamp,
            })
            current_batch["point_ids"].append(next_point_id)
            next_point_id += 1

            if len(current_batch["texts"]) == batch_size:
                batches.append(current_batch)
                current_batch = {
                    "texts": [],
                    "payloads": [],
                    "point_ids": [],
                }

    if current_batch["texts"]:
        batches.append(current_batch)

    return batches

#################################################################################
# Helper 23: Assign production thread IDs using subject and participant overlap #
#################################################################################
def assign_thread_ids_by_subject_and_participant_overlap_for_production(emails, my_email_addresses):
    if not emails:
        return []

    my_email_addresses = set(
        email.lower()
        for email in (my_email_addresses or [])
        if email
    )

    emails_with_threads = {}

    for email in emails:
        email_subject = email.get("subject") or ""
        email_normalized_subject = normalize_subject(email_subject)
        email_participants = extract_emails_from_participant_raw_texts(
            email.get("from"),
            email.get("to"),
        )
        email_participants = {email for email in email_participants if email not in my_email_addresses}
        if not email_participants:
            continue
        email_participants_key = tuple(sorted(email_participants))

        found_key_match = False
        # if it's the first email, add
        if len(emails_with_threads) == 0:
            emails_with_threads[(email_normalized_subject, email_participants_key)] = [email]
        # otherwise:
        else:
            # for what we've seen so far
            for key in list(emails_with_threads.keys()):
                thread_normalized_subject = key[0]
                thread_participants = key[1]
                # if our current email's normalized subject match and participants intersect:
                if email_normalized_subject == thread_normalized_subject and set(thread_participants).intersection(email_participants):
                    # calculate the new key (extending participants)
                    new_thread_participants = email_participants.union(thread_participants)
                    new_thread_participants_key = tuple(sorted(new_thread_participants))
                    # set as new emails for the (normalized subject, extended set) the old emails (popping the key) and new email
                    thread_emails = emails_with_threads.pop(key)
                    thread_emails.append(email)
                    emails_with_threads[(thread_normalized_subject, new_thread_participants_key)] = thread_emails
                    found_key_match = True
            # and if no match (with a normalized subject and participants), add as new key/value
            if not found_key_match:
                emails_with_threads[(email_normalized_subject, email_participants_key)] = [email]

    # add thread ids
    emails_with_threads_list = []
    for thread_id, thread_emails in enumerate(emails_with_threads.values(), start=1):
        for thread_email in thread_emails:
            email_with_thread = thread_email.copy()
            email_with_thread["threadID"] = thread_id
            emails_with_threads_list.append(email_with_thread)

    return emails_with_threads_list

################################################
# Helper 24: Build stable key per email sample #
################################################
def build_email_sample_key(email_sample):
    return (
        email_sample["folder_uri"],
        email_sample["thread_id"],
        email_sample["email"]["subject"],
        email_sample["email"]["body"],
    )

###################################################################
# Helper 25: Build intermediate and final rows for M3 fine-tuning #
###################################################################
def build_finetune_rows(
        data_variant_to_oracle_results,
        data_variant_to_rrf_results,
        query_types,
        ):
    from config.decoder import QUERY_TYPE_TO_N_MAX_QUERIES
    from helpers.eval import get_text_to_rerank_from_payload

    all_query_types = list(QUERY_TYPE_TO_N_MAX_QUERIES) + ["reranker", "original_email"]
    first_data_variant_oracle_results = next(iter(data_variant_to_oracle_results.values()))
    email_sample_key_to_intermediate_row = {}

    # figure out which emails to include in the fine-tuning dataset
    for oracle_result in first_data_variant_oracle_results:
        email_sample = oracle_result["sample"]
        email_sample_key = build_email_sample_key(email_sample)
        email_sample_key_to_intermediate_row[email_sample_key] = {
            "email_sample": email_sample,
            "queries": {
                query_type: set()
                for query_type in all_query_types
            },
            "positives": set(),
            "positives_by_source": {},
            "negatives": set(),
            "negatives_by_source": {},
        }

    # add oracle positives and negatives for each selected email
    for data_variant, data_variant_oracle_results in data_variant_to_oracle_results.items():
        for oracle_result in data_variant_oracle_results:
            if oracle_result["generation_failed"]:
                continue

            email_sample_key = build_email_sample_key(oracle_result["sample"])
            intermediate_row = email_sample_key_to_intermediate_row[email_sample_key]
            source = f"oracle:{data_variant}"

            # each reranker query can have *multiple* subqueries or subquestions we ask 
            # the model to generate (with positives and negatives on each)
            for subquery in oracle_result["discriminator_result"]["subqueries"]:
                if subquery["answerability"] not in {"0", "1"}:
                    continue
                for positive_chunk in subquery["supporting_chunks"]:
                    positive_text = get_text_to_rerank_from_payload(positive_chunk["payload"])
                    intermediate_row["positives"].add(positive_text)
                    if source not in intermediate_row["positives_by_source"]:
                        intermediate_row["positives_by_source"][source] = set()
                    intermediate_row["positives_by_source"][source].add(positive_text)
                for negative_chunk in subquery["insufficient_chunks"]:
                    negative_text = get_text_to_rerank_from_payload(negative_chunk["payload"])
                    if negative_text in intermediate_row["positives"]:
                        continue
                    intermediate_row["negatives"].add(negative_text)
                    if source not in intermediate_row["negatives_by_source"]:
                        intermediate_row["negatives_by_source"][source] = set()
                    intermediate_row["negatives_by_source"][source].add(negative_text)

    # add queries once from rrf and add rrf negatives for every data variant
    for data_variant, data_variant_rrf_results in data_variant_to_rrf_results.items():
        for rrf_result in data_variant_rrf_results:
            email_sample_key = build_email_sample_key(rrf_result["sample"])
            if email_sample_key not in email_sample_key_to_intermediate_row:
                continue
            intermediate_row = email_sample_key_to_intermediate_row[email_sample_key]

            if not any(intermediate_row["queries"][query_type] for query_type in all_query_types):
                if "reranker" in query_types:
                    intermediate_row["queries"]["reranker"].add(rrf_result["reranker_query"])
                if "original_email" in query_types:
                    intermediate_row["queries"]["original_email"].add(
                        f"Subject:\n{intermediate_row['email_sample']['email']['subject']}\n\n"
                        f"Body:\n{intermediate_row['email_sample']['email']['body']}"
                    )
                for rewritten_query in rrf_result["rewritten_queries"]:
                    query_type = rewritten_query["query_type"]
                    if query_type not in query_types:
                        continue
                    intermediate_row["queries"][query_type].add(rewritten_query["query"])

            source = f"rrf:{data_variant}"
            for retrieval_result in rrf_result["retrieval_results"]:
                negative_text = get_text_to_rerank_from_payload(retrieval_result["payload"])
                if negative_text in intermediate_row["positives"]:
                    continue
                intermediate_row["negatives"].add(negative_text)
                if source not in intermediate_row["negatives_by_source"]:
                    intermediate_row["negatives_by_source"][source] = set()
                intermediate_row["negatives_by_source"][source].add(negative_text)

    intermediate_rows = []

    # keep valid intermediate rows and flatten them into final fine-tune rows
    for intermediate_row in email_sample_key_to_intermediate_row.values():
        if not intermediate_row["positives"]:
            continue
        if not intermediate_row["negatives"]:
            continue
        intermediate_rows.append({
            "email_sample": intermediate_row["email_sample"],
            "queries": {
                query_type: sorted(intermediate_row["queries"][query_type])
                for query_type in all_query_types
            },
            "positives": sorted(intermediate_row["positives"]),
            "positives_by_source": {
                source: sorted(passage_texts)
                for source, passage_texts in intermediate_row["positives_by_source"].items()
            },
            "negatives": sorted(intermediate_row["negatives"]),
            "negatives_by_source": {
                source: sorted(passage_texts)
                for source, passage_texts in intermediate_row["negatives_by_source"].items()
            },
        })

    finetune_rows = []
    for intermediate_row in intermediate_rows:
        for query_type in all_query_types:
            for query_text in intermediate_row["queries"][query_type]:
                finetune_rows.append({
                    "query": query_text,
                    "pos": intermediate_row["positives"],
                    "neg": intermediate_row["negatives"],
                })

    return intermediate_rows, finetune_rows
