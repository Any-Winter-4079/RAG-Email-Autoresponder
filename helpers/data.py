#############################################################################
# Helper 1: Check whether all email participants are internal UPM personnel #
#############################################################################
def is_upm_internal(author, recipients, upm_domains):
    import re
    # if all participants (author and recipients) aren't students / external people (e.g., they are professors):
    # return True to remove from the dataset
    author_email = re.findall(r'[\w\.-]+@[\w\.-]+', author.lower()) # list
    recipient_emails = re.findall(r'[\w\.-]+@[\w\.-]+', recipients.lower()) # list
    all_emails = author_email + recipient_emails
    return all(any(f"@{domain}" in email for domain in upm_domains) for email in all_emails) and len(all_emails) > 0

###################################################################
# Helper 2: Normalize an email subject by removing reply prefixes #
###################################################################
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

####################################################
# Helper 3: Normalize an email body for comparison #
####################################################
def normalize_email_body(body):
    return body.replace("\n", " ").replace("\r", " ").strip().lower()

#######################################################################
# Helper 4: Extract participant email addresses from author/to fields #
#######################################################################
def extract_participant_emails(author_raw_text, recipients_raw_text):
    import re

    participants = set()
    for text in [author_raw_text, recipients_raw_text]:
        if not text:
            continue
        for email in re.findall(r'[\w\.-]+@[\w\.-]+', text.lower()):
            participants.add(email)
    return participants

############################################################
# Helper 5: Save a pie-chart distribution as an image file #
############################################################
def save_pie_chart_distribution(labels, sizes, title, output_path):
    from pathlib import Path
    import matplotlib.pyplot as plt

    pie_colors = [
        "#FFAF00",
        "#F46920",
        "#F857C1",
        "#F53255",
        "#29BDFD",
        "#00CBBF",
        "#01C159",
        "#9DCA1C",
    ]
    pie_alpha = 0.78
    fig, ax = plt.subplots(figsize=(8, 8))
    wedges, *_ = ax.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%', # one decimal place
        colors=pie_colors[:len(labels)],
        counterclock=False
    )
    for wedge in wedges:
        wedge.set_alpha(pie_alpha)
    ax.set_title(title)
    ax.axis("equal")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

####################################################################
# Helper 6: Save a stacked size-distribution plot as an image file #
####################################################################
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
    inbound_color = "#F46920"
    outbound_color = "#F53255"
    ax.bar(
        x_values,
        inbound_bar_heights,
        width=0.85,
        color=inbound_color,
        alpha=0.80,
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
        alpha=0.80,
        edgecolor="white",
        linewidth=0.6,
        label="Outgoing share",
    )
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_xticks(x_values)
    ax.legend()
    for bar, count_value in zip(bars, count_values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            count_value,
            f"{count_value}",
            ha="center",
            va="bottom",
        )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)

####################################################################################
# Helper 7: Save a 3D folderURI dropped-message distribution plot as an image file #
####################################################################################
def save_folder_uri_drop_3d_plot(folder_uri_count_history, output_path, excluded_phase_labels=None):
    from pathlib import Path
    import matplotlib.pyplot as plt
    from matplotlib.collections import PolyCollection
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
                ax.text(x_value, phase_index, drop, str(drop), ha="center", va="bottom")

    ax.set_title("3D Folder Drop Plot")
    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.set_zlabel("Dropped Emails")
    ax.set_xticks(x_values)
    ax.set_xticklabels(folder_labels, rotation=30, ha="right")
    ax.set_yticks(range(len(phase_labels)))
    ax.set_yticklabels(phase_labels)
    ax.set_zlim(bottom=0)
    ax.view_init(elev=25, azim=-50)
    fig.subplots_adjust(left=0.02, right=0.95, bottom=0.18, top=0.92)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"save_folder_uri_drop_3d_plot: saved 3D folder drop plot to {output_path}")

####################################################################
# Helper 8: Count folderURI values and print stage-by-stage deltas #
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

##########################################################################
# Helper 9: Assign thread IDs to contiguous blocks with the same subject #
##########################################################################
def assign_thread_ids_by_subject_blocks(emails):
    if not emails:
        return []

    emails_to_process = list(emails)

    thread_id = 0
    first_subject = emails_to_process[0].get("subject") or ""
    current_subject = normalize_subject(first_subject)
    current_block_emails = []
    emails_with_threads = []

    def flush_block():
        nonlocal thread_id, current_block_emails
        thread_id += 1
        for block_email in current_block_emails:
            email_with_thread = block_email.copy()
            email_with_thread["threadID"] = thread_id
            emails_with_threads.append(email_with_thread)
        current_block_emails = []

    for email in emails_to_process:
        subject = email.get("subject") or ""
        normalized_subject = normalize_subject(subject)
        if normalized_subject != current_subject:
            flush_block()
            current_subject = normalized_subject

        current_block_emails.append(email)

    if current_block_emails:
        flush_block()

    return emails_with_threads

##############################################################################
# Helper 10: Add dataset thread IDs for contiguous same-subject email blocks #
##############################################################################
def assign_thread_ids_by_subject_blocks_for_dataset(rows):
    # insert 'threadID' as 2nd column on 1st row
    rows_with_threads = [rows[0][:1] + ["threadID"] + rows[0][1:]]

    data_rows = rows[1:]
    email_subjects = [{"subject": row[1]} for row in data_rows]
    emails_with_threads = assign_thread_ids_by_subject_blocks(email_subjects)

    for row, email in zip(data_rows, emails_with_threads):
        rows_with_threads.append(row[:1] + [email["threadID"]] + row[1:])

    return rows_with_threads

##############################################################################
# Helper 11: Assign dataset thread IDs using subject and participant overlap #
##############################################################################
def assign_thread_ids_by_subject_and_participant_overlap_for_dataset(rows, my_email_addresses, lookback_window_rows):
    rows_with_threads = [rows[0][:1] + ["threadID"] + rows[0][1:]]

    folder_uri_index = rows[0].index("folderURI")
    subject_index = rows[0].index("c1subject")
    author_index = rows[0].index("c3author")
    recipients_index = rows[0].index("c4recipients")

    my_email_addresses = set(
        email.lower()
        for email in (my_email_addresses or [])
        if email
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
        all_participants = extract_participant_emails(author, recipients)
        participants = {email for email in all_participants if email not in my_email_addresses}
        if not participants:
            participants = all_participants

        candidate_thread_ids = []
        for thread_id, metadata in thread_id_to_metadata.items():
            if metadata["folder_uri"] != folder_uri:
                continue
            if metadata["normalized_subject"] != normalized_subject:
                continue
            if row_index - metadata["last_row_index"] > lookback_window_rows:
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

    for row, thread_id in zip(rows[1:], row_thread_ids):
        rows_with_threads.append(row[:1] + [thread_id] + row[1:])

    return rows_with_threads

############################################################################
# Helper 12: Reconstruct dataset thread IDs with decoder-based LM grouping #
############################################################################
def assign_thread_ids_with_decoder_for_dataset(
        rows,
        my_email_addresses,
        lookback_window_rows,
        run_thread_grouper,
        thread_grouper_model_config,
        task_description_start,
        example_message,
        prompt_template,
        max_emails_per_batch,
        max_input_tokens,
        weak_thread_size_plot_path,
        ):
    import asyncio
    import sys
    from collections import Counter
    from transformers import AutoProcessor, AutoTokenizer
    from config.decoder import GEMMA4_MODEL_FAMILY
    from helpers.decoder import count_tokens
    from statistics import median

    weak_rows_with_threads = assign_thread_ids_by_subject_and_participant_overlap_for_dataset(
        rows,
        my_email_addresses,
        lookback_window_rows,
    )
    weak_header = weak_rows_with_threads[0]
    final_header = weak_header[:2] + ["weakThreadID"] + weak_header[2:]

    rows_with_threads = [final_header]
    next_thread_id = 1
    batch_entries = []
    planned_batch_email_counts = []
    planned_batch_prompt_token_counts = []
    bypassed_email_limit_weak_group_count = 0
    bypassed_email_limit_email_count = 0
    bypassed_token_limit_weak_group_count = 0
    bypassed_token_limit_email_count = 0
    bypassed_token_limit_prompt_token_counts = []
    my_email_addresses = {
        email.lower()
        for email in (my_email_addresses or [])
        if email
    }
    weak_group_sizes = []
    weak_group_inbound_email_counts = Counter()
    weak_group_outbound_email_counts = Counter()

    folder_uri_to_weak_groups = build_weak_groups_by_folder_from_rows_with_threads(
        weak_rows_with_threads
    )

    thread_grouper_tokenizer = None
    if max_input_tokens is not None:
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

    def build_prompt_from_batch_rows(batch_rows):
        emails = [
            {
                "id": str(batch_row_index),
                "threadID": row["weak_thread_id"],
                "from": row["author"].strip(),
                "to": row["recipients"].strip(),
                "subject": row["subject"].strip(),
                "body": row["body"].strip(),
            }
            for batch_row_index, row in enumerate(batch_rows)
        ]
        prompt = prompt_template.format(
            task_description_start=task_description_start.format(
                email_count=len(emails)
            ),
            example_message=example_message,
            emails_section="\n".join(str(email) for email in emails),
        )
        return prompt

    def count_batch_prompt_tokens(batch_rows):
        if thread_grouper_tokenizer is None:
            return None
        return count_tokens(
            thread_grouper_tokenizer,
            build_prompt_from_batch_rows(batch_rows),
        )

    for folder_uri, weak_groups in folder_uri_to_weak_groups.items():
        for weak_group in weak_groups:
            weak_group_size = weak_group["thread_size"]
            weak_group_sizes.append(weak_group_size)
            outbound_email_count = sum(
                1
                for email in weak_group["emails"]
                if extract_participant_emails(email["author"], "").intersection(my_email_addresses)
            )
            inbound_email_count = weak_group_size - outbound_email_count
            weak_group_outbound_email_counts[weak_group_size] += outbound_email_count
            weak_group_inbound_email_counts[weak_group_size] += inbound_email_count

        batches = []
        current_batch = []
        current_batch_size = 0
        for weak_group in weak_groups:
            weak_group_size = weak_group["thread_size"]
            if weak_group_size > max_emails_per_batch:
                if current_batch:
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_size = 0
                bypassed_email_limit_weak_group_count += 1
                bypassed_email_limit_email_count += weak_group_size
                for message in weak_group["emails"]:
                    rows_with_threads.append([
                        folder_uri,
                        next_thread_id,
                        weak_group["weak_thread_id"],
                        message["subject"],
                        message["body"],
                        message["author"],
                        message["recipients"],
                    ])
                next_thread_id += 1
                continue

            if max_input_tokens is not None:
                weak_group_prompt_tokens = count_batch_prompt_tokens(weak_group["emails"])
                if weak_group_prompt_tokens is not None and weak_group_prompt_tokens > max_input_tokens:
                    if current_batch:
                        batches.append(current_batch)
                        current_batch = []
                        current_batch_size = 0
                    bypassed_token_limit_weak_group_count += 1
                    bypassed_token_limit_email_count += weak_group_size
                    bypassed_token_limit_prompt_token_counts.append(weak_group_prompt_tokens)
                    for message in weak_group["emails"]:
                        rows_with_threads.append([
                            folder_uri,
                            next_thread_id,
                            weak_group["weak_thread_id"],
                            message["subject"],
                            message["body"],
                            message["author"],
                            message["recipients"],
                        ])
                    next_thread_id += 1
                    continue

            if (
                current_batch
                and current_batch_size + weak_group_size > max_emails_per_batch
            ):
                batches.append(current_batch)
                current_batch = []
                current_batch_size = 0

            if max_input_tokens is not None and current_batch:
                candidate_batch = current_batch + weak_group["emails"]
                candidate_prompt_tokens = count_batch_prompt_tokens(candidate_batch)
                if (
                    candidate_prompt_tokens is not None
                    and candidate_prompt_tokens > max_input_tokens
                ):
                    batches.append(current_batch)
                    current_batch = []
                    current_batch_size = 0

            current_batch.extend(weak_group["emails"])
            current_batch_size += weak_group_size
        if current_batch:
            batches.append(current_batch)

        for batch_rows in batches:
            batch_weak_thread_ids = sorted(
                {
                    row["weak_thread_id"]
                    for row in batch_rows
                }
            )
            batch_weak_thread_id_hint = "|".join(str(thread_id) for thread_id in batch_weak_thread_ids)
            prompt = build_prompt_from_batch_rows(batch_rows)
            batch_prompt_tokens = count_batch_prompt_tokens(batch_rows)
            planned_batch_email_counts.append(len(batch_rows))
            if batch_prompt_tokens is not None:
                planned_batch_prompt_token_counts.append(batch_prompt_tokens)
            batch_entries.append((folder_uri, batch_rows, batch_weak_thread_id_hint, prompt))

    if weak_group_sizes:
        weak_group_size_counts = Counter(weak_group_sizes)
        save_stacked_size_distribution_plot(
            weak_group_size_counts,
            weak_group_inbound_email_counts,
            weak_group_outbound_email_counts,
            title="Weak Thread Size Distribution",
            x_label="Emails per weak thread",
            y_label="Weak thread count",
            output_path=weak_thread_size_plot_path,
        )
        print(
            "assign_thread_ids_with_decoder_for_dataset: "
            f"saved weak-thread-size plot to {weak_thread_size_plot_path}"
        )

    async def run_batch(prompt):
        return await run_thread_grouper.remote.aio(
            context=[],
            current_turn_input_text=prompt,
            current_turn_image_in_bytes=None,
            **thread_grouper_model_config,
        )

    async def run_all_batches():
        if not batch_entries:
            return []
        return await asyncio.gather(
            *(run_batch(prompt) for _, _, _, prompt in batch_entries),
            return_exceptions=True,
        )

    total_weak_group_count = len(weak_group_sizes)
    total_batched_weak_group_count = (
        total_weak_group_count
        - bypassed_email_limit_weak_group_count
        - bypassed_token_limit_weak_group_count
    )
    print(
        "assign_thread_ids_with_decoder_for_dataset: "
        f"planning summary: {total_weak_group_count} weak groups total, "
        f"{total_batched_weak_group_count} weak groups sent to LM, "
        f"{len(batch_entries)} LM batches planned"
    )
    if bypassed_email_limit_weak_group_count:
        print(
            "assign_thread_ids_with_decoder_for_dataset: "
            "planning summary: "
            f"{bypassed_email_limit_weak_group_count} weak groups bypassed by email count "
            f"covering {bypassed_email_limit_email_count} input emails"
        )
    if bypassed_token_limit_weak_group_count:
        print(
            "assign_thread_ids_with_decoder_for_dataset: "
            "planning summary: "
            f"{bypassed_token_limit_weak_group_count} weak groups bypassed by input tokens "
            f"covering {bypassed_token_limit_email_count} input emails"
        )
        print(
            "assign_thread_ids_with_decoder_for_dataset: "
            "planning summary: bypassed token-limit prompt tokens "
            f"(min/median/max) = "
            f"{min(bypassed_token_limit_prompt_token_counts)}/"
            f"{int(median(bypassed_token_limit_prompt_token_counts))}/"
            f"{max(bypassed_token_limit_prompt_token_counts)}"
        )
    if planned_batch_email_counts:
        print(
            "assign_thread_ids_with_decoder_for_dataset: "
            "planning summary: planned batch input emails "
            f"(min/median/max) = "
            f"{min(planned_batch_email_counts)}/"
            f"{int(median(planned_batch_email_counts))}/"
            f"{max(planned_batch_email_counts)}"
        )
    if planned_batch_prompt_token_counts:
        print(
            "assign_thread_ids_with_decoder_for_dataset: "
            "planning summary: planned batch prompt tokens "
            f"(min/median/max) = "
            f"{min(planned_batch_prompt_token_counts)}/"
            f"{int(median(planned_batch_prompt_token_counts))}/"
            f"{max(planned_batch_prompt_token_counts)}"
        )

    sys.exit(0)

    batch_results = asyncio.run(run_all_batches())
    failed_batch_count = 0
    failed_email_count = 0

    for (folder_uri, batch_rows, batch_weak_thread_id_hint, _), batch_result in zip(batch_entries, batch_results):
        if isinstance(batch_result, Exception):
            failed_batch_count += 1
            failed_email_count += len(batch_rows)
            print(
                "assign_thread_ids_with_decoder_for_dataset: "
                f"batch detail: skipping failed batch for folderURI={folder_uri} "
                f"with {len(batch_rows)} input emails ({batch_result})"
            )
            continue

        parsed_threads, _ = batch_result
        if not parsed_threads:
            failed_batch_count += 1
            failed_email_count += len(batch_rows)
            print(
                "assign_thread_ids_with_decoder_for_dataset: "
                f"batch detail: skipping empty-output batch for folderURI={folder_uri} "
                f"with {len(batch_rows)} input emails and 0 output emails"
            )
            continue

        n_output_messages = sum(
            len(thread["messages"])
            for thread in parsed_threads
        )

        if n_output_messages < len(batch_rows):
            failed_batch_count += 1
            failed_email_count += len(batch_rows)
            print(
                "assign_thread_ids_with_decoder_for_dataset: "
                f"batch detail: error: skipping short-output batch for folderURI={folder_uri} "
                f"with {len(batch_rows)} input emails and {n_output_messages} output emails"
            )
            continue
        if n_output_messages > len(batch_rows):
            print(
                "assign_thread_ids_with_decoder_for_dataset: "
                f"batch detail: warning: keeping expanded-output batch for folderURI={folder_uri} "
                f"with {len(batch_rows)} input emails and {n_output_messages} output emails"
            )

        for thread in parsed_threads:
            for message in thread["messages"]:
                rows_with_threads.append([
                    folder_uri,
                    next_thread_id,
                    batch_weak_thread_id_hint,
                    message.get("subject"),
                    message.get("body"),
                    message.get("from"),
                    message.get("to"),
                ])
            next_thread_id += 1

    if failed_batch_count:
        print(
            "assign_thread_ids_with_decoder_for_dataset: "
            f"summary: skipped {failed_batch_count} failed batches covering {failed_email_count} input emails"
        )
    if bypassed_email_limit_weak_group_count:
        print(
            "assign_thread_ids_with_decoder_for_dataset: "
            f"summary: bypassed {bypassed_email_limit_weak_group_count} oversized weak groups by email count "
            f"covering {bypassed_email_limit_email_count} input emails"
        )
    if bypassed_token_limit_weak_group_count:
        print(
            "assign_thread_ids_with_decoder_for_dataset: "
            f"summary: bypassed {bypassed_token_limit_weak_group_count} oversized weak groups by input tokens "
            f"covering {bypassed_token_limit_email_count} input emails"
        )

    return rows_with_threads

#################################################################################
# Helper 13: Assign production thread IDs using subject and participant overlap #
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
        email_participants = extract_participant_emails(email.get("from"), email.get("to"))
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

###################################################################
# Helper 14: Split an email body into unquoted and quoted content #
###################################################################
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

#######################################################################
# Helper 15: Build weak-thread groups by folderURI from threaded rows #
#######################################################################
def build_weak_groups_by_folder_from_rows_with_threads(rows_with_threads):
    if not rows_with_threads:
        return {}

    header = rows_with_threads[0]
    folder_uri_index = header.index("folderURI")
    weak_thread_id_index = header.index("threadID")
    subject_index = header.index("c1subject")
    body_index = header.index("c0body")
    author_index = header.index("c3author")
    recipients_index = header.index("c4recipients")

    folder_uri_to_rows = {}
    for row in rows_with_threads[1:]:
        folder_uri = row[folder_uri_index]
        folder_uri_to_rows.setdefault(folder_uri, []).append(row)

    folder_uri_to_weak_groups = {}
    for folder_uri, folder_rows in folder_uri_to_rows.items():
        weak_groups = []
        current_group_rows = []
        current_weak_thread_id = None

        for row in folder_rows:
            weak_thread_id = row[weak_thread_id_index]
            if current_group_rows and weak_thread_id != current_weak_thread_id:
                weak_groups.append({
                    "folder_uri": folder_uri,
                    "weak_thread_id": current_weak_thread_id,
                    "thread_size": len(current_group_rows),
                    "emails": [
                        {
                            "subject": group_row[subject_index],
                            "body": group_row[body_index],
                            "author": group_row[author_index],
                            "recipients": group_row[recipients_index],
                            "weak_thread_id": current_weak_thread_id,
                        }
                        for group_row in current_group_rows
                    ],
                })
                current_group_rows = []
            current_group_rows.append(row)
            current_weak_thread_id = weak_thread_id

        if current_group_rows:
            weak_groups.append({
                "folder_uri": folder_uri,
                "weak_thread_id": current_weak_thread_id,
                "thread_size": len(current_group_rows),
                "emails": [
                    {
                        "subject": group_row[subject_index],
                        "body": group_row[body_index],
                        "author": group_row[author_index],
                        "recipients": group_row[recipients_index],
                        "weak_thread_id": current_weak_thread_id,
                    }
                    for group_row in current_group_rows
                ],
            })

        folder_uri_to_weak_groups[folder_uri] = weak_groups

    return folder_uri_to_weak_groups

######################################################################
# Helper 16: Save weak-thread groups bucketed by size for inspection #
######################################################################
def save_weak_threads_by_size_for_dataset(
        rows,
        my_email_addresses,
        lookback_window_rows,
        output_path,
        ):
    import json
    from pathlib import Path

    weak_rows_with_threads = assign_thread_ids_by_subject_and_participant_overlap_for_dataset(
        rows,
        my_email_addresses,
        lookback_window_rows,
    )
    folder_uri_to_weak_groups = build_weak_groups_by_folder_from_rows_with_threads(
        weak_rows_with_threads
    )

    thread_size_to_weak_groups = {}
    for weak_groups in folder_uri_to_weak_groups.values():
        for weak_group in weak_groups:
            thread_size = weak_group["thread_size"]
            thread_size_to_weak_groups.setdefault(thread_size, []).append(weak_group)

    sorted_thread_size_to_weak_groups = {
        str(thread_size): thread_size_to_weak_groups[thread_size]
        for thread_size in sorted(thread_size_to_weak_groups, reverse=True)
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", encoding="utf-8") as json_file:
        json.dump(sorted_thread_size_to_weak_groups, json_file, ensure_ascii=False, indent=2)
    print(f"save_weak_threads_by_size_for_dataset: saved dataset to {output_path}")

######################################################################
# Helper 17: Build final post-LM threads grouped by folderURI and ID #
######################################################################
def build_grouped_threads_from_messages_with_threads_rows(messages_with_threads_rows):
    ordered_thread_keys = []
    thread_key_to_thread = {}

    for row in messages_with_threads_rows:
        folder_uri = row["folderURI"]
        thread_id = str(row["threadID"])
        thread_key = (folder_uri, thread_id)

        if thread_key not in thread_key_to_thread:
            thread_key_to_thread[thread_key] = {
                "folder_uri": folder_uri,
                "thread_id": thread_id,
                "emails": [],
            }
            if "weakThreadID" in row:
                thread_key_to_thread[thread_key]["weak_thread_id_hints"] = set()
            ordered_thread_keys.append(thread_key)

        if row.get("weakThreadID"):
            for weak_thread_id_hint in str(row["weakThreadID"]).split("|"):
                if weak_thread_id_hint:
                    thread_key_to_thread[thread_key]["weak_thread_id_hints"].add(
                        weak_thread_id_hint
                    )

        thread_key_to_thread[thread_key]["emails"].append({
            "subject": row["c1subject"],
            "body": row["c0body"],
            "author": row["c3author"],
            "recipients": row["c4recipients"],
        })

    for thread in thread_key_to_thread.values():
        thread["thread_size"] = len(thread["emails"])
        if "weak_thread_id_hints" in thread:
            thread["weak_thread_id_hints"] = sorted(thread["weak_thread_id_hints"])

    return ordered_thread_keys, thread_key_to_thread

#####################################################################
# Helper 18: Build final post-LM threads bucketed by LM thread size #
#####################################################################
def build_lm_threads_by_size_for_dataset(
        rows_with_threads,
        my_email_addresses,
        ):
    from collections import Counter

    if not rows_with_threads:
        return {}, Counter(), Counter(), Counter()

    header = rows_with_threads[0]
    messages_with_threads_rows = [
        dict(zip(header, row))
        for row in rows_with_threads[1:]
    ]
    _, thread_key_to_thread = build_grouped_threads_from_messages_with_threads_rows(
        messages_with_threads_rows
    )

    my_email_addresses = {
        email.lower()
        for email in (my_email_addresses or [])
        if email
    }

    thread_size_to_threads = {}
    thread_size_counts = Counter()
    thread_inbound_email_counts = Counter()
    thread_outbound_email_counts = Counter()

    for thread in thread_key_to_thread.values():
        thread_size = thread["thread_size"]
        thread_size_counts[thread_size] += 1

        outbound_email_count = sum(
            1
            for email in thread["emails"]
            if extract_participant_emails(email["author"], "").intersection(my_email_addresses)
        )
        inbound_email_count = thread_size - outbound_email_count
        thread_outbound_email_counts[thread_size] += outbound_email_count
        thread_inbound_email_counts[thread_size] += inbound_email_count

        thread_size_to_threads.setdefault(thread_size, []).append(thread)

    sorted_thread_size_to_threads = {
        str(thread_size): thread_size_to_threads[thread_size]
        for thread_size in sorted(thread_size_to_threads, reverse=True)
    }

    return (
        sorted_thread_size_to_threads,
        thread_size_counts,
        thread_inbound_email_counts,
        thread_outbound_email_counts,
    )

##################################################################
# Helper 19: Save the final post-LM LM-threads-by-size JSON file #
##################################################################
def save_lm_threads_by_size_json(thread_size_to_threads, output_path):
    import json
    from pathlib import Path

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, mode="w", encoding="utf-8") as json_file:
        json.dump(thread_size_to_threads, json_file, ensure_ascii=False, indent=2)
    print(f"save_lm_threads_by_size_json: saved dataset to {output_path}")

#####################################################################
# Helper 20: Detect template text in the unquoted portion of a body #
#####################################################################
def has_template_in_unquoted(body, templates):
    unquoted = normalize_email_body(get_unquoted_text(body))
    return any(template in unquoted for template in templates)

#################################################
# Helper 21: Save tabular dataset rows as a CSV #
#################################################
def save_dataset(rows, output_path, delimiter=";"):
    import csv
    with open(output_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=delimiter)
        writer.writerows(rows)
    print(f"save_dataset: saved dataset to {output_path}")

################################################
# Helper 22: Save tabular dataset rows as JSON #
################################################
def save_dataset_as_json(rows, output_path):
    import json

    if not rows:
        payload = []
    else:
        header = rows[0]
        payload = [
            dict(zip(header, row))
            for row in rows[1:]
        ]

    with open(output_path, mode="w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, ensure_ascii=False, indent=2)
    print(f"save_dataset_as_json: saved dataset to {output_path}")

####################################################################
# Helper 23: Build email samples grouped by folderURI and threadID #
####################################################################
def build_samples_by_folder_uri_and_thread_id(
        messages_with_threads_header,
        messages_with_threads_data,
        my_email_addresses,
        upm_domains,
        remove_internal_upm_messages,
        ):
    import re

    folder_uri_index = messages_with_threads_header.index("folderURI")
    thread_id_index = messages_with_threads_header.index("threadID")
    subject_index = messages_with_threads_header.index("c1subject")
    body_index = messages_with_threads_header.index("c0body")
    author_index = messages_with_threads_header.index("c3author")
    recipients_index = messages_with_threads_header.index("c4recipients")

    my_email_addresses = set(
        email.lower()
        for email in (my_email_addresses or [])
        if email
    )

    def extract_emails(text):
        if not text:
            return set()
        return set(re.findall(r'[\w\.-]+@[\w\.-]+', text.lower()))

    folder_uri_to_thread_id_to_rows = {}
    for row in messages_with_threads_data:
        folder_uri = row[folder_uri_index]
        thread_id = row[thread_id_index]
        if folder_uri not in folder_uri_to_thread_id_to_rows:
            folder_uri_to_thread_id_to_rows[folder_uri] = {}
        if thread_id not in folder_uri_to_thread_id_to_rows[folder_uri]:
            folder_uri_to_thread_id_to_rows[folder_uri][thread_id] = []
        folder_uri_to_thread_id_to_rows[folder_uri][thread_id].append(row)

    folder_uri_to_thread_id_to_samples = {}
    discarded_internal_threads = []
    for folder_uri, thread_id_to_rows in folder_uri_to_thread_id_to_rows.items():
        thread_id_to_samples = {}

        for thread_id, thread_rows in thread_id_to_rows.items():
            thread_emails = []
            for row in thread_rows:
                thread_emails.append({
                    "subject": row[subject_index],
                    "body": row[body_index],
                    "author": row[author_index],
                    "recipients": row[recipients_index],
                })

            if (
                remove_internal_upm_messages
                and
                thread_emails
                and all(
                    is_upm_internal(
                        email["author"],
                        email["recipients"],
                        upm_domains,
                    )
                    for email in thread_emails
                )
            ):
                discarded_internal_threads.append({
                    "folder_uri": folder_uri,
                    "thread_id": thread_id,
                    "thread_size": len(thread_emails),
                    "emails": thread_emails,
                })
                continue

            thread_samples = []
            thread_size = len(thread_emails)
            for email_index, email in enumerate(thread_emails):
                email_recipient_emails = extract_emails(email["recipients"])
                is_inbound_to_director = bool(email_recipient_emails.intersection(my_email_addresses))
                if not is_inbound_to_director:
                    continue

                context_emails = thread_emails[:email_index]
                later_emails = thread_emails[email_index + 1:]
                other_gold_reply_candidates = [
                    later_email
                    for later_email in later_emails
                    if bool(extract_emails(later_email["author"]).intersection(my_email_addresses))
                ]

                email_author_emails = extract_emails(email["author"])
                gold_reply = None
                if other_gold_reply_candidates:
                    for candidate in other_gold_reply_candidates:
                        candidate_recipient_emails = extract_emails(candidate["recipients"])
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

            if thread_samples:
                thread_id_to_samples[thread_id] = thread_samples

        if thread_id_to_samples:
            folder_uri_to_thread_id_to_samples[folder_uri] = thread_id_to_samples

    return folder_uri_to_thread_id_to_samples, discarded_internal_threads

#######################################################
# Helper 24: Count built samples grouped by folderURI #
#######################################################
def get_sample_counts_by_folder_uri(folder_uri_to_thread_id_to_samples):
    folder_uri_to_sample_count = {}
    for folder_uri, thread_id_to_samples in folder_uri_to_thread_id_to_samples.items():
        sample_count = get_nested_values_length_sum(thread_id_to_samples, depth=1)
        folder_uri_to_sample_count[folder_uri] = sample_count

    return folder_uri_to_sample_count

############################################################
# Helper 25: Sum nested value lengths to a requested depth #
############################################################
def get_nested_values_length_sum(dictionary, depth):
    if depth == 1:
        return sum(len(value) for value in dictionary.values())
    return sum(
        get_nested_values_length_sum(value, depth - 1)
        for value in dictionary.values()
    )

###################################################################
# Helper 26: Split grouped samples into train, dev, and test sets #
###################################################################
def split_samples_by_split_name(folder_uri_to_thread_id_to_samples, train_split_pct, dev_split_pct, seed):
    import random

    rng = random.Random(seed)
    split_names = ["train", "dev", "test"]
    split_name_to_samples = {
        split_name: []
        for split_name in split_names
    }

    for folder_uri, thread_id_to_samples in folder_uri_to_thread_id_to_samples.items():
        n_samples_in_folder = get_nested_values_length_sum(thread_id_to_samples, depth=1)
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

        thread_id_to_samples_items = list(thread_id_to_samples.items())
        rng.shuffle(thread_id_to_samples_items)

        for thread_id, samples in thread_id_to_samples_items:
            n_samples_in_thread = len(samples)
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
            split_name_to_samples[best_split_name].extend(samples)

    return split_name_to_samples

#####################################################
# Helper 27: Build a stable key for an email sample #
#####################################################
def build_email_sample_key(email_sample):
    return (
        email_sample["folder_uri"],
        email_sample["thread_id"],
        email_sample["email"]["subject"],
        email_sample["email"]["body"],
    )

###################################################################
# Helper 28: Build intermediate and final rows for M3 fine-tuning #
###################################################################
def build_finetune_rows(
        data_variant_to_oracle_results,
        data_variant_to_rrf_results,
        query_types,
        ):
    from config.decoder import QUERY_REWRITER_SECTION_TO_MAX_QUERIES
    from helpers.eval import get_text_to_rerank_from_payload

    all_query_types = list(QUERY_REWRITER_SECTION_TO_MAX_QUERIES) + ["reranker", "original_email"]
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

########################################################################
# Helper 29: Format grouped thread emails as a single promptable block #
########################################################################
def format_email_thread_text(emails):
    formatted_messages = []
    for email in emails:
        subject = (email.get("subject") or "").strip()
        body = (email.get("body") or "").strip()
        author = (email.get("author") or "").strip()
        recipients = (email.get("recipients") or "").strip()

        formatted_messages.append(
            "-----\n"
            f"From: {author}\n"
            f"To: {recipients}\n"
            f"Subject: {subject}\n\n"
            f"{body}\n"
            "-----"
        )
    return "\n".join(formatted_messages)

#########################################################################
# Helper 30: Build email-thread curator candidates from grouped threads #
#########################################################################
def build_email_thread_candidates_from_grouped_threads(grouped_threads):
    thread_candidates = []
    for grouped_thread in grouped_threads:
        emails = grouped_thread.get("emails") or []
        thread_candidates.append({
            "folder_uri": grouped_thread["folder_uri"],
            "thread_id": str(grouped_thread["thread_id"]),
            "weak_thread_id_hints": list(grouped_thread.get("weak_thread_id_hints") or []),
            "thread_size": grouped_thread.get("thread_size", len(emails)),
            "emails": emails,
            "thread_text": format_email_thread_text(emails),
        })
    return thread_candidates

####################################################################
# Helper 31: Build email-KB variant chunks from one curated thread #
####################################################################
def build_email_thread_knowledge_base_chunks(candidate, curator_output):
    abstract_chunks = []
    summary_chunks = []
    cleaned_text_chunks = []
    q_and_a_chunks = []

    abstract = curator_output.get("abstract")
    if abstract:
        abstract_chunks.append({
            "folder_uri": candidate["folder_uri"],
            "thread_id": candidate["thread_id"],
            "weak_thread_id_hints": candidate["weak_thread_id_hints"],
            "thread_size": candidate["thread_size"],
            "text": abstract,
        })

    summary = curator_output.get("summary")
    if summary:
        summary_chunks.append({
            "folder_uri": candidate["folder_uri"],
            "thread_id": candidate["thread_id"],
            "weak_thread_id_hints": candidate["weak_thread_id_hints"],
            "thread_size": candidate["thread_size"],
            "text": summary,
        })

    cleanedtext = curator_output.get("cleanedtext")
    if cleanedtext:
        cleaned_text_chunks.append({
            "folder_uri": candidate["folder_uri"],
            "thread_id": candidate["thread_id"],
            "weak_thread_id_hints": candidate["weak_thread_id_hints"],
            "thread_size": candidate["thread_size"],
            "text": cleanedtext,
        })

    pairs = [
        {
            "question": question,
            "answer": answer,
        }
        for question, answer in zip(
            curator_output.get("questions") or [],
            curator_output.get("answers") or [],
        )
    ]
    if pairs:
        q_and_a_chunks.append({
            "folder_uri": candidate["folder_uri"],
            "thread_id": candidate["thread_id"],
            "weak_thread_id_hints": candidate["weak_thread_id_hints"],
            "thread_size": candidate["thread_size"],
            "pairs": pairs,
        })

    return {
        "email_lm_abstract_chunks": abstract_chunks,
        "email_lm_summary_chunks": summary_chunks,
        "email_lm_cleaned_text_chunks": cleaned_text_chunks,
        "email_lm_q_and_a_chunks": q_and_a_chunks,
    }

######################################################################
# Helper 32: Prepare knowledge-base records as encoder-ready batches #
######################################################################
def prepare_batches_for_knowledge_base_variant(
        variant,
        records,
        batch_size,
        encode_timestamp,
        ):
    q_and_a_variants = {
        "lm_q_and_a_chunks",
        "email_lm_q_and_a_chunks",
    }
    q_only_variants = {
        "lm_q_and_a_for_q_only_chunks",
        "email_lm_q_and_a_for_q_only_chunks",
    }

    batches = []
    current_batch = {
        "texts": [],
        "payloads": [],
        "point_ids": [],
    }
    next_point_id = 0

    for record in records:
        if variant in q_and_a_variants or variant in q_only_variants:
            for pair_index, pair in enumerate(record["pairs"], start=1):
                if variant in q_and_a_variants:
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
                })
                if "decoder_token_count" in pair:
                    payload["decoder_token_count"] = pair["decoder_token_count"]
                if "encoder_token_count" in pair:
                    payload["encoder_token_count"] = pair["encoder_token_count"]

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
