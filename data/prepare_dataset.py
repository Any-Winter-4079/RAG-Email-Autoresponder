import sys
from os.path import dirname, abspath
from collections import Counter
project_root = abspath(dirname(dirname(__file__)))
sys.path.insert(0, project_root)

from config.data import (
    DATASET_PATH,
    KNOWLEDGE_BASE_PATH,
    THREADS_PATH,
    THREAD_GROUPER_PRE_DECODER_STATISTICS_PLOT_PATH,
    THREAD_GROUPER_POST_DECODER_STATISTICS_PLOT_PATH,
    DISCARDED_THREADS_PATH,
    AUTOMATED_OUTBOUND_TEMPLATES,
    PRE_ENROLLMENT_TEMPLATES,
    THREAD_GROUPING_STRATEGY,
    VALID_THREAD_GROUPING_STRATEGIES,
    REMOVE_INTERNAL_UPM_MESSAGES,
    UPM_DOMAINS,
    THREAD_GROUPER_MAX_EMAILS,
    THREAD_GROUPER_MAX_RULE_BASED_THREADS,
)
from config.decoder import (
    MAX_CONCURRENT_BATCHES,
    MODEL_PROFILES,
    THREAD_GROUPER_PROFILE,
)
from config.email_agent import N_CONTEXT_EMAILS_PER_FOLDER
from helpers.email_agent import transform_env_csv_into_list
from helpers.data import (
    normalize_subject,
    normalize_email_body,
    save_pie_chart_distribution_plot,
    save_stacked_size_distribution_plot,
    save_folder_uri_drop_3d_plot,
    get_and_print_folder_uri_counts,
    assign_thread_ids_by_subject_and_participant_overlap_for_dataset,
    assign_thread_ids_with_decoder_for_dataset,
    remove_internal_upm_threads,
    extract_emails_from_participant_raw_texts,
    get_unquoted_text,
    has_template_in_unquoted,
)
import os
import csv
import json
from pathlib import Path
import modal
from cycler import cycler
from dotenv import load_dotenv
import matplotlib.pyplot as plt

load_dotenv()
my_email_addresses = transform_env_csv_into_list(os.getenv("MY_EMAIL_ADDRESSES", ""))

custom_colors = [
    "#FFAF00", "#F46920", "#F53255", "#F857C1",
     "#29BDFD", "#00CBBF", "#01C159", "#9DCA1C"
]
plt.rcParams["axes.prop_cycle"] = cycler(color=custom_colors)

def main():
    if THREAD_GROUPING_STRATEGY not in VALID_THREAD_GROUPING_STRATEGIES:
        raise ValueError(
            "prepare_dataset: invalid thread grouping strategy:\n"
            f"\t{THREAD_GROUPING_STRATEGY}\n"
            f"\tvalid values: {sorted(VALID_THREAD_GROUPING_STRATEGIES)}"
        )

    with open(DATASET_PATH, newline='') as csv_file:
        rows = list(csv.reader(csv_file, delimiter=';'))
        header_row = rows[0]
        previous_folder_uri_counts = None

        print("\n"+"="*50)
        print("Step 0")
        print("="*50)
        print(f"\nTotal data columns: {len(header_row)}")
        print(f"Column names: {header_row}")
        print(f"Total data rows (excluding header with column names): {len(rows[1:])}")
        total_count_history = [("start", len(rows[1:]))]
        outbound_count = 0
        inbound_to_director_count = 0
        for row in rows[1:]:
            author = row[5].lower()
            recipients = row[6].lower()
            is_outbound = any(email in author for email in my_email_addresses)
            is_inbound_to_director = any(email in recipients for email in my_email_addresses)
            if is_outbound:
                outbound_count += 1
            if is_inbound_to_director:
                inbound_to_director_count += 1
        print(f"\nUsing director emails: {my_email_addresses}")
        print("Author distribution:")
        print(f"\tOutbound (author is director): {outbound_count} messages")
        print(f"\tInbound to director (recipient includes director): {inbound_to_director_count} messages")

        ########################################################
        # Remove id column (due to being unique, not relevant) #
        ########################################################
        print("\n"+"="*50)
        print("Step 1: removing id column")
        print("="*50)
        # obtain the column id values by getting the 1st value in each row
        id_column = [row[0] for row in rows[1:]] # skipping header
        # drop id if values they are unique
        if len(set(id_column)) == len(id_column):
            rows_no_id = [row[1:] for row in rows]
        else:
            raise ValueError("Duplicate IDs found")
        print(f"\nColumn names ({len(rows_no_id[0])} total): {rows_no_id[0]}")

        ###################################################################################
        # Remove folderID column (due to matching folderURI while being less descriptive) #
        ###################################################################################
        print("\n"+"="*50)
        print("Step 2: removing folderID column")
        print("="*50)
        # obtain the folderID values by getting the (now) 1st value in each row
        folder_id_column = [row[0] for row in rows_no_id[1:]] # skipping header
        unique_folder_ids = set(folder_id_column)
        print(f"\nUnique folderID values: {len(unique_folder_ids)}")

        # obtain the folderURI values by getting the (now) 2st value in each row
        folder_uri_column = [row[1] for row in rows_no_id[1:]] # skipping header
        unique_folder_uris = set(folder_uri_column)
        print(f"Unique folderURI values: {len(unique_folder_uris)}")

        # drop folderID column (first column) from each row
        rows_no_id_and_no_folder_id = [row[1:] for row in rows_no_id]
        print(f"Column names ({len(rows_no_id_and_no_folder_id[0])} total): {rows_no_id_and_no_folder_id[0]}")

        # show distribution
        previous_folder_uri_counts = get_and_print_folder_uri_counts(
            [row[0] for row in rows_no_id_and_no_folder_id[1:]], # folderURI is the 1st value and we skip the header
            title="Distribution of messages by folderURI:"
        )
        folder_uri_count_history = [("initial", dict(previous_folder_uri_counts))]
        
        #################################
        # Remove rows with empty c0body #
        #################################
        print("\n"+"="*50)
        print("Step 3: removing rows with empty c0body")
        print("="*50)
        non_empty_c0body_rows = [rows_no_id_and_no_folder_id[0]] # keep header
        for row in rows_no_id_and_no_folder_id[1:]:
            if row[2].strip(): # check that the body is non-empty
                non_empty_c0body_rows.append(row)

        # show updated distribution
        previous_folder_uri_counts = get_and_print_folder_uri_counts(
            [row[0] for row in non_empty_c0body_rows[1:]], # folderURI is the 1st value and we skip the header
            title="Distribution of messages by folderURI (post empty body removal):",
            previous_counts=previous_folder_uri_counts
        )
        folder_uri_count_history.append(("empty_body", dict(previous_folder_uri_counts)))
        total_count_history.append(("empty_body", len(non_empty_c0body_rows) - 1))

        #####################################################################################################
        # Remove pre-enrollment template messages (due to being messages not suited for automated response) #
        #####################################################################################################
        print("\n"+"="*50)
        print("Step 4: removing pre-enrollment template messages")
        print("="*50)
        normalized_pre_enrollment_templates = [template.lower() for template in PRE_ENROLLMENT_TEMPLATES]
        pre_enrollment_counts = {template: 0 for template in PRE_ENROLLMENT_TEMPLATES}
        for template, normalized_template in zip(PRE_ENROLLMENT_TEMPLATES, normalized_pre_enrollment_templates):
            pre_enrollment_counts[template] = sum(
                1 for row in non_empty_c0body_rows[1:]
                if normalized_template in normalize_email_body(get_unquoted_text(row[2]))
            )
        print("\nPre-enrollment template instance occurrences:")
        for template in PRE_ENROLLMENT_TEMPLATES:
            print(f"\ttemplate '{template}': {pre_enrollment_counts[template]} messages")

        pre_enrollment_filtered_rows = [non_empty_c0body_rows[0]] # keep header
        for row in non_empty_c0body_rows[1:]: # skipping header
            if not has_template_in_unquoted(row[2], normalized_pre_enrollment_templates):
                pre_enrollment_filtered_rows.append(row)

        pre_enrollment_match_counts = {matches: 0 for matches in range(0, len(normalized_pre_enrollment_templates) + 1)}
        for row in non_empty_c0body_rows[1:]:
            matches = sum(1 for template in normalized_pre_enrollment_templates if template in normalize_email_body(get_unquoted_text(row[2])))
            pre_enrollment_match_counts[matches] += 1
        print("\n\trows by pre-enrollment template match count:")
        for matches in range(0, len(normalized_pre_enrollment_templates) + 1):
            print(f"\t\t{matches} {'template: ' if matches == 1 else 'templates:'} {pre_enrollment_match_counts[matches]} rows")

        # show updated distribution
        previous_folder_uri_counts = get_and_print_folder_uri_counts(
            [row[0] for row in pre_enrollment_filtered_rows[1:]], # folderURI is the 1st value and we skip the header
            title="Distribution of messages by folderURI (post pre-enrollment filtering):",
            previous_counts=previous_folder_uri_counts
        )
        folder_uri_count_history.append(("pre_enrollment", dict(previous_folder_uri_counts)))
        total_count_history.append(("pre_enrollment", len(pre_enrollment_filtered_rows) - 1))

        #############################################################################################
        # Remove (admission/rejection) template messages (due to being automated outbound messages) #
        #############################################################################################
        # NOTE: keeping an instance per template for knowledge base messages
        print("\n"+"="*50)
        print("Step 5: removing admission/rejection template messages")
        print("="*50)
        normalized_templates = [message.lower() for message in AUTOMATED_OUTBOUND_TEMPLATES]
        template_sample_rows = [pre_enrollment_filtered_rows[0]]
        collected_templates = set()
        # count ocurrences of messages that are templates and not handwritten
        print("\nAdmission/rejection template instance occurrences:")
        for message in AUTOMATED_OUTBOUND_TEMPLATES:
            count = sum(
                1 for row in pre_enrollment_filtered_rows[1:]
                if message.lower() in normalize_email_body(get_unquoted_text(row[2]))
            )
            print(f"\ttemplate '{message}': {count} messages")
        multi_template_rows = sum(
            1 for row in pre_enrollment_filtered_rows[1:]
            if sum(1 for template in normalized_templates if template in normalize_email_body(get_unquoted_text(row[2]))) > 1
        )
        print(f"\n\trows with >1 template match: {multi_template_rows}")
        match_counts = {matches: 0 for matches in range(0, len(normalized_templates) + 1)}
        for row in pre_enrollment_filtered_rows[1:]:
            matches = sum(1 for template in normalized_templates if template in normalize_email_body(get_unquoted_text(row[2])))
            match_counts[matches] += 1
        print("\n\trows by template match count:")
        for matches in range(0, len(normalized_templates) + 1):
            print(f"\t\t{matches} {'template: ' if matches == 1 else 'templates:'} {match_counts[matches]} rows")

        # keep one instance per admission/rejection template for knowledge base messages
        for row in pre_enrollment_filtered_rows[1:]:
            normalized_body = normalize_email_body(get_unquoted_text(row[2]))
            for template in normalized_templates:
                if template in normalized_body and template not in collected_templates:
                    template_sample_rows.append(row)
                    collected_templates.add(template)
                    break
        if template_sample_rows[1:]:
            print(f"\nCollected {len(template_sample_rows) - 1} template samples for knowledge base messages")

        knowledge_base_messages = []
        for row in template_sample_rows[1:]:
            knowledge_base_messages.append({
                "folder_uri": row[0],
                "subject": row[1],
                "body": row[2],
                "author": row[3],
                "recipients": row[4],
            })
        Path(KNOWLEDGE_BASE_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(KNOWLEDGE_BASE_PATH, mode="w", encoding="utf-8") as json_file:
            json.dump(knowledge_base_messages, json_file, ensure_ascii=False, indent=2)
        print(f"Saved knowledge base to {KNOWLEDGE_BASE_PATH}")

        # filter out rows with admission / rejection to MUIA messages
        filtered_rows = [pre_enrollment_filtered_rows[0]] # keep header
        for row in pre_enrollment_filtered_rows[1:]: # skipping header
            if not has_template_in_unquoted(row[2], normalized_templates):
                filtered_rows.append(row)

        # show updated distribution
        previous_folder_uri_counts = get_and_print_folder_uri_counts(
            [row[0] for row in filtered_rows[1:]], # folderURI is the 1st value and we skip the header
            title="Distribution of messages by folderURI (post admission/rejection filtering):",
            previous_counts=previous_folder_uri_counts
        )
        folder_uri_count_history.append(("admission_rejection", dict(previous_folder_uri_counts)))
        total_count_history.append(("admission_rejection", len(filtered_rows) - 1))

        ##################################################################
        # Remove duplicate rows (same body, subject, author, recipients) #
        ##################################################################
        print("\n"+"="*50)
        print("Step 6: removing duplicate rows (spam, etc.)")
        print("="*50)
        seen_messages = set()
        dedup_filtered_rows = [filtered_rows[0]] # keep header
        for row in filtered_rows[1:]:
            normalized_subject = normalize_subject(row[1]) # c1subject is the 2nd column after id and folderID removal
            normalized_body = normalize_email_body(row[2]) # c0body is the 3rd column after id and folderID removal
            author = row[3].strip().lower()
            recipients = row[4].strip().lower()
            message_key = (normalized_body, normalized_subject, author, recipients)
            if message_key not in seen_messages:
                seen_messages.add(message_key)
                dedup_filtered_rows.append(row)
        
        # show updated distribution
        previous_folder_uri_counts = get_and_print_folder_uri_counts(
            [row[0] for row in dedup_filtered_rows[1:]], # folderURI is the 1st value and we skip the header
            title="Distribution of messages by folderURI (post deduplication):",
            previous_counts=previous_folder_uri_counts
        )
        folder_uri_count_history.append(("duplicates", dict(previous_folder_uri_counts)))
        total_count_history.append(("duplicates", len(dedup_filtered_rows) - 1))

        save_folder_uri_drop_3d_plot(
            folder_uri_count_history,
            "Dropped emails per folderURI (by dropping reason)",
            Path(THREADS_PATH).with_name("folder_uri_drop_3d.png"),
        )
        save_folder_uri_drop_3d_plot(
            folder_uri_count_history,
            "Dropped emails per folderURI (by dropping reason)",
            Path(THREADS_PATH).with_name("folder_uri_drop_3d_without_templates.png"),
            excluded_phase_labels={"pre_enrollment", "admission_rejection"},
        )

        ##########################
        # Group emails by thread #
        ##########################
        # NOTE: adding extra column with unique id per thread
        print("\n"+"="*50)
        print("Step 7: grouping emails by thread")
        print("="*50)

        n_dataset_thread_lookback_window_rows = 2 * N_CONTEXT_EMAILS_PER_FOLDER
        print(f"\nLookback window rows: {n_dataset_thread_lookback_window_rows}")
        threads = assign_thread_ids_by_subject_and_participant_overlap_for_dataset(
            dedup_filtered_rows,
            my_email_addresses,
            n_dataset_thread_lookback_window_rows,
        )

        if THREAD_GROUPING_STRATEGY == "decoder_based":
            thread_grouper_profile_config = MODEL_PROFILES[THREAD_GROUPER_PROFILE].copy()
            decoder_app_name = thread_grouper_profile_config.pop("decoder_app_name")
            decoder_function_name = thread_grouper_profile_config.pop("decoder_function_name")
            task_description_start = thread_grouper_profile_config.pop("dataset_task_description_start")
            example_message = thread_grouper_profile_config.pop("dataset_example")
            prompt_template = thread_grouper_profile_config.pop("prompt_template")
            max_input_tokens = thread_grouper_profile_config.pop("max_input_tokens", None)
            thread_grouper_profile_config.pop("production_task_description_start")
            thread_grouper_profile_config.pop("production_example")
            thread_grouper_profile_config["decoder_profile"] = THREAD_GROUPER_PROFILE

            run_thread_grouper = modal.Function.from_name(
                decoder_app_name,
                decoder_function_name,
            )
            threads = assign_thread_ids_with_decoder_for_dataset(
                threads,
                run_thread_grouper,
                thread_grouper_profile_config,
                task_description_start,
                example_message,
                prompt_template,
                THREAD_GROUPER_MAX_EMAILS,
                THREAD_GROUPER_MAX_RULE_BASED_THREADS,
                MAX_CONCURRENT_BATCHES,
                max_input_tokens,
                pre_decoder_statistics_plot_path=Path(
                    THREAD_GROUPER_PRE_DECODER_STATISTICS_PLOT_PATH
                ),
                post_decoder_statistics_plot_path=Path(
                    THREAD_GROUPER_POST_DECODER_STATISTICS_PLOT_PATH
                ),
            )

        previous_folder_uri_counts = get_and_print_folder_uri_counts(
            [
                thread["folder_uri"]
                for thread in threads
                for _ in thread["emails"]
            ],
            title="Distribution of messages by folderURI (post thread grouping):",
            previous_counts=previous_folder_uri_counts,
        )

        ###########################
        # Remove internal threads #
        ###########################
        print("\n"+"="*50)
        print("Step 8: removing fully internal UPM threads")
        print("="*50)

        if REMOVE_INTERNAL_UPM_MESSAGES:
            threads, discarded_internal_threads = remove_internal_upm_threads(
                threads,
                UPM_DOMAINS,
            )
            Path(DISCARDED_THREADS_PATH).parent.mkdir(parents=True, exist_ok=True)
            with open(DISCARDED_THREADS_PATH, mode="w", encoding="utf-8") as json_file:
                json.dump(discarded_internal_threads, json_file, ensure_ascii=False, indent=2)
            print(f"\nRemoved fully internal UPM threads: {len(discarded_internal_threads)}")
            print(f"Saved discarded internal threads to {DISCARDED_THREADS_PATH}")
            previous_folder_uri_counts = get_and_print_folder_uri_counts(
                [
                    thread["folder_uri"]
                    for thread in threads
                    for _ in thread["emails"]
                ],
                title="Distribution of messages by folderURI (post internal thread removal):",
                previous_counts=previous_folder_uri_counts,
            )
        else:
            print("\nSkipping removal of fully internal UPM threads")

        #################
        # Store threads #
        #################
        print("\n"+"="*50)
        print("Step 9: storing threads")
        print("="*50)

        Path(THREADS_PATH).parent.mkdir(parents=True, exist_ok=True)
        with open(THREADS_PATH, mode="w", encoding="utf-8") as json_file:
            json.dump(threads, json_file, ensure_ascii=False, indent=2)
        print(f"Saved threads to {THREADS_PATH}")

        threads_by_size = {}
        for thread in threads:
            thread_size = str(thread["thread_size"])
            threads_by_size.setdefault(thread_size, []).append(thread)

        sorted_threads_by_size = {
            thread_size: threads_by_size[thread_size]
            for thread_size in sorted(threads_by_size, key=int, reverse=True)
        }
        threads_by_size_path = Path(THREADS_PATH).with_name("threads_by_size.json")
        with open(threads_by_size_path, mode="w", encoding="utf-8") as json_file:
            json.dump(sorted_threads_by_size, json_file, ensure_ascii=False, indent=2)
        print(f"Saved threads by size to {threads_by_size_path}")

        normalized_my_email_addresses = {
            email.lower()
            for email in my_email_addresses
            if email
        }
        thread_size_counts = Counter()
        thread_inbound_email_counts = Counter()
        thread_outbound_email_counts = Counter()
        for thread in threads:
            thread_size = thread["thread_size"]
            thread_size_counts[thread_size] += 1

            outbound_email_count = sum(
                1
                for email in thread["emails"]
                if extract_emails_from_participant_raw_texts(
                    email["author"]
                ).intersection(normalized_my_email_addresses)
            )
            inbound_email_count = thread_size - outbound_email_count
            thread_outbound_email_counts[thread_size] += outbound_email_count
            thread_inbound_email_counts[thread_size] += inbound_email_count

        save_stacked_size_distribution_plot(
            thread_size_counts,
            thread_inbound_email_counts,
            thread_outbound_email_counts,
            title="Thread size distribution",
            x_label="Emails per thread",
            y_label="Thread count",
            output_path=Path(THREADS_PATH).with_name(
                "thread_size_distribution.png"
            ),
        )

        folder_label_order = [
            "Admisi&APM-n_Consultas",
            "Seminarios",
            "TFM_solicitudTitulo",
            "MH_PremioTFM",
            "Alumnos",
            "Erasmus_Movilidad",
        ]
        folder_label_to_color = {
            "Admisi&APM-n_Consultas": "#00CBBF",
            "Seminarios": "#FFAF00",
            "TFM_solicitudTitulo": "#F46920",
            "MH_PremioTFM": "#F857C1",
            "Alumnos": "#F53255",
            "Erasmus_Movilidad": "#29BDFD",
        }
        save_pie_chart_distribution_plot(
            [uri.split('/')[-1] for uri in previous_folder_uri_counts.keys()],
            list(previous_folder_uri_counts.values()),
            title="Distribution of messages by folderURI",
            output_path=Path(THREADS_PATH).with_name(
                "folder_uri_distribution.png"
            ),
            preferred_label_order=folder_label_order,
            label_to_color=folder_label_to_color,
        )
        authored_by_director_count = sum(thread_outbound_email_counts.values())
        authored_by_another_count = sum(thread_inbound_email_counts.values())
        save_pie_chart_distribution_plot(
            ["Authored by director", "Authored by another"],
            [
                authored_by_director_count,
                authored_by_another_count,
            ],
            title="Distribution of messages by author",
            output_path=Path(THREADS_PATH).with_name(
                "author_distribution.png"
            ),
        )

if __name__ == "__main__":
    main()
