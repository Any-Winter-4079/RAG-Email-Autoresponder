###################################################
# Helper 1: Check if email is from/to UPM domains #
###################################################
def is_upm_internal(author, recipients, upm_domains):
    import re
    # if all participants (author and recipients) aren't students / external people (e.g., they are professors):
    # return True to remove from the dataset
    author_email = re.findall(r'[\w\.-]+@[\w\.-]+', author.lower()) # list
    recipient_emails = re.findall(r'[\w\.-]+@[\w\.-]+', recipients.lower()) # list
    all_emails = author_email + recipient_emails
    return all(any(f"@{domain}" in email for domain in upm_domains) for email in all_emails) and len(all_emails) > 0

##########################################
# Helper 2: Normalize email subject text #
##########################################
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

#######################################
# Helper 3: Normalize email body text #
#######################################
def normalize_email_body(body):
    return body.replace("\n", " ").replace("\r", " ").strip().lower()

######################################################
# Helper 4: Extract participant emails from raw text #
######################################################
def extract_participant_emails(author_raw_text, recipients_raw_text):
    import re

    participants = set()
    for text in [author_raw_text, recipients_raw_text]:
        if not text:
            continue
        for email in re.findall(r'[\w\.-]+@[\w\.-]+', text.lower()):
            participants.add(email)
    return participants

#####################################
# Helper 5: Plot distribution chart #
#####################################
def plot_distribution(labels, sizes, title):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(8, 8))
    plt.pie(
        sizes,
        labels=labels,
        autopct='%1.1f%%', # one decimal place
        counterclock=False
    )
    plt.title(title)
    plt.axis("equal")
    plt.show()

################################################
# Helper 6: Print and return folder URI counts #
################################################
def get_and_print_folder_uri_counts(folder_uri_column, title, previous_counts=None):
    from collections import Counter

    folder_uri_counts = Counter(folder_uri_column)
    print(f"\n{title}")
    # print Alumnos/Seminarios/... counts from folderURI
    for uri, count in folder_uri_counts.items():
        diff_text = ""
        if previous_counts is not None:
            diff = count - previous_counts.get(uri, 0)
            diff_text = " (==)" if diff == 0 else f" ({diff:+d})"
        print(f"\t{uri.split('/')[-1]}: {count} messages{diff_text}")
    total_count = sum(folder_uri_counts.values())
    total_diff_text = ""
    if previous_counts is not None:
        total_diff = total_count - sum(previous_counts.values())
        total_diff_text = " (==)" if total_diff == 0 else f" ({total_diff:+d})"
    print(f"\tTotal: {total_count} messages{total_diff_text}")
    return folder_uri_counts

###########################################################
# Helper 7: Assign thread ids by contiguous same-subjects #
###########################################################
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

#########################################################################
# Helper 8: Assign thread ids by contiguous same-subjects (for dataset) #
#########################################################################
def assign_thread_ids_by_subject_blocks_for_dataset(rows):
    # insert 'threadID' as 2nd column on 1st row
    rows_with_threads = [rows[0][:1] + ["threadID"] + rows[0][1:]]

    data_rows = rows[1:]
    email_subjects = [{"subject": row[1]} for row in data_rows]
    emails_with_threads = assign_thread_ids_by_subject_blocks(email_subjects)

    for row, email in zip(data_rows, emails_with_threads):
        rows_with_threads.append(row[:1] + [email["threadID"]] + row[1:])

    return rows_with_threads

################################################################################
# Helper 9: Assign thread ids by subject and participant overlap (for dataset) #
################################################################################
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

####################################################################################
# Helper 10: Assign thread ids by subject and participant overlap (for production) #
####################################################################################
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

##############################################
# Helper 11: Split quoted messages from body #
##############################################
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

#########################################################
# Helper 12: Check template matches in unquoted content #
#########################################################
def has_template_in_unquoted(body, templates):
    unquoted = normalize_email_body(get_unquoted_text(body))
    return any(template in unquoted for template in templates)

###########################
# Helper 13: Save dataset #
###########################
def save_dataset(rows, output_path, delimiter=";"):
    import csv
    with open(output_path, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=delimiter)
        writer.writerows(rows)
    print(f"Saved dataset to {output_path}")

##############################################################
# Helper 14: Build samples grouped by folderURI and threadID #
##############################################################
def build_samples_by_folder_uri_and_thread_id(messages_with_threads_header, messages_with_threads_data, my_email_addresses):
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

            thread_samples = []
            thread_size = len(thread_emails)
            for email_index, email in enumerate(thread_emails):
                email_recipient_emails = extract_emails(email["recipients"])
                is_inbound_to_director = bool(email_recipient_emails.intersection(my_email_addresses))
                if not is_inbound_to_director:
                    continue

                context_emails = thread_emails[:email_index]
                later_emails = thread_emails[email_index + 1:]
                gold_reply_candidates = [
                    later_email
                    for later_email in later_emails
                    if bool(extract_emails(later_email["author"]).intersection(my_email_addresses))
                ]

                email_author_emails = extract_emails(email["author"])
                gold_reply = None
                if gold_reply_candidates:
                    for candidate in gold_reply_candidates:
                        candidate_recipient_emails = extract_emails(candidate["recipients"])
                        if email_author_emails.intersection(candidate_recipient_emails):
                            gold_reply = candidate
                            break
                    if gold_reply is None:
                        gold_reply = gold_reply_candidates[0]

                thread_samples.append({
                    "folder_uri": folder_uri,
                    "thread_id": thread_id,
                    "email": email,
                    "context_emails": context_emails,
                    "gold_reply": gold_reply,
                    "gold_reply_candidates": gold_reply_candidates,
                    "thread_size": thread_size,
                })

            if thread_samples:
                thread_id_to_samples[thread_id] = thread_samples

        if thread_id_to_samples:
            folder_uri_to_thread_id_to_samples[folder_uri] = thread_id_to_samples

    return folder_uri_to_thread_id_to_samples

#####################################################
# Helper 15: Get sample counts grouped by folderURI #
#####################################################
def get_sample_counts_by_folder_uri(folder_uri_to_thread_id_to_samples):
    folder_uri_to_sample_count = {}
    for folder_uri, thread_id_to_samples in folder_uri_to_thread_id_to_samples.items():
        sample_count = get_nested_values_length_sum(thread_id_to_samples, depth=1)
        folder_uri_to_sample_count[folder_uri] = sample_count

    return folder_uri_to_sample_count

################################################
# Helper 16: Sum nested value lengths by depth #
################################################
def get_nested_values_length_sum(dictionary, depth):
    if depth == 1:
        return sum(len(value) for value in dictionary.values())
    return sum(
        get_nested_values_length_sum(value, depth - 1)
        for value in dictionary.values()
    )

##########################################################
# Helper 17: Split samples into train/dev/val/test lists #
##########################################################
def split_samples_by_split_name(folder_uri_to_thread_id_to_samples, train_split_pct, dev_split_pct, val_split_pct):
    import random

    split_names = ["train", "dev", "val", "test"]
    split_name_to_samples = {
        split_name: []
        for split_name in split_names
    }

    for folder_uri, thread_id_to_samples in folder_uri_to_thread_id_to_samples.items():
        n_samples_in_folder = get_nested_values_length_sum(thread_id_to_samples, depth=1)
        split_name_to_n_samples_goal_in_folder = {
            "train": int(train_split_pct * n_samples_in_folder),
            "dev": int(dev_split_pct * n_samples_in_folder),
            "val": int(val_split_pct * n_samples_in_folder),
        }
        split_name_to_n_samples_goal_in_folder["test"] = (
            n_samples_in_folder
            - split_name_to_n_samples_goal_in_folder["train"]
            - split_name_to_n_samples_goal_in_folder["dev"]
            - split_name_to_n_samples_goal_in_folder["val"]
        )
        split_name_to_n_samples_assigned_in_folder = {
            split_name: 0
            for split_name in split_names
        }

        thread_id_to_samples_items = list(thread_id_to_samples.items())
        random.shuffle(thread_id_to_samples_items)

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
