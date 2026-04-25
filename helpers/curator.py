##############################################
# Helper 1: Build email knowledge-base input #
##############################################
def build_email_knowledge_base_threads(threads_path, knowledge_base_path):
    import json

    with open(threads_path, mode="r", encoding="utf-8") as json_file:
        threads = json.load(json_file)

    next_thread_id = max(
        int(thread.get("thread_id", 0))
        for thread in threads
    ) + 1

    if knowledge_base_path.exists():
        with open(knowledge_base_path, mode="r", encoding="utf-8") as json_file:
            templates = json.load(json_file)
        for template in templates:
            threads.append({
                "folder_uri": template["folder_uri"],
                "thread_id": str(next_thread_id),
                "thread_size": 1,
                "emails": [{
                    "subject": template.get("subject", ""),
                    "body": template.get("body", ""),
                    "author": template.get("author", ""),
                    "recipients": template.get("recipients", ""),
                }],
            })
            next_thread_id += 1

    return threads

#########################################################
# Helper 2: Run email knowledge base curator on threads #
#########################################################
def run_email_knowledge_base_curator_on_threads(
        threads,
        run_email_knowledge_base_curator,
        run_decoder_latest_tokenizer,
        email_knowledge_base_curator_profile_config,
        prompt_template,
        max_emails_per_batch,
        max_threads_per_batch,
        max_concurrent_batches,
        max_input_tokens,
        pre_decoder_statistics_plot_path=None,
        post_decoder_statistics_plot_path=None,
        return_run_data=False,
        ):
    import asyncio
    from config.decoder import (
        THREAD_CLOSING_TAG,
        THREAD_OPENING_TAG,
    )
    from helpers.data import (
        format_email_thread_text,
        save_pre_decoder_statistics_plot,
        save_post_decoder_statistics_plot,
    )

    def build_prompt_from_batch_threads(batch_threads):
        threads_text = "\n".join(
            (
                f"{THREAD_OPENING_TAG}\n"
                f"{format_email_thread_text(thread['emails'])}\n"
                f"{THREAD_CLOSING_TAG}"
            )
            for thread in batch_threads
        )
        prompt = prompt_template.format(threads_text=threads_text)
        return prompt

    def build_thread_chunk(thread, chunk_emails):
        return {
            "folder_uri": thread.get("folder_uri"),
            "thread_id": thread.get("thread_id"),
            "emails": chunk_emails,
        }

    model_name_or_path = email_knowledge_base_curator_profile_config["model_name_or_path"]

    def split_thread_into_chunks(thread):
        thread_chunks = []
        chunk_emails = []

        for email in thread["emails"]:
            hypothetical_chunk_emails = chunk_emails + [email]
            hypothetical_chunk = build_thread_chunk(
                thread,
                hypothetical_chunk_emails,
            )
            hypothetical_chunk_prompt = build_prompt_from_batch_threads(
                [hypothetical_chunk]
            )
            n_hypothetical_chunk_prompt_tokens = run_decoder_latest_tokenizer.remote(
                [hypothetical_chunk_prompt],
                model_name_or_path,
            )[0]

            fits = (
                (
                    max_emails_per_batch is None
                    or len(hypothetical_chunk_emails) <= max_emails_per_batch
                )
                and n_hypothetical_chunk_prompt_tokens <= max_input_tokens
            )
            if fits:
                chunk_emails = hypothetical_chunk_emails
            else:
                if chunk_emails:
                    split_reason = (
                        "email_limit"
                        if (
                            max_emails_per_batch is not None
                            and len(hypothetical_chunk_emails) > max_emails_per_batch
                        )
                        else "token_limit"
                    )
                    split_statistics[split_reason]["n_events"] += 1
                    thread_chunks.append(
                        build_thread_chunk(thread, chunk_emails)
                    )
                    chunk_emails = [email]
                else:
                    thread_chunks.append(
                        build_thread_chunk(thread, [email])
                    )
                    chunk_emails = []

        if chunk_emails:
            thread_chunks.append(
                build_thread_chunk(thread, chunk_emails)
            )

        return thread_chunks

    batches = []
    batch = []
    batch_statistics = []
    split_statistics = {
        "email_limit": {"n_events": 0},
        "token_limit": {"n_events": 0},
    }
    n_split_threads = 0
    n_thread_chunks = 0

    for thread in threads:
        thread_chunks = split_thread_into_chunks(thread)
        if len(thread_chunks) > 1:
            n_split_threads += 1
        n_thread_chunks += len(thread_chunks)

        for thread_chunk in thread_chunks:
            batch_emails = [
                email
                for batch_thread in batch
                for email in batch_thread["emails"]
            ]
            n_hypothetical_batch_threads = len(batch) + 1
            n_hypothetical_batch_emails = (
                len(batch_emails)
                + len(thread_chunk["emails"])
            )
            hypothetical_batch_prompt = build_prompt_from_batch_threads(
                batch + [thread_chunk]
            )
            n_hypothetical_batch_prompt_tokens = run_decoder_latest_tokenizer.remote(
                [hypothetical_batch_prompt],
                model_name_or_path,
            )[0]
            fits = (
                (
                    thread_chunk["folder_uri"] == batch[-1]["folder_uri"]
                    if len(batch) > 0 else True
                )
                and (
                    max_emails_per_batch is None
                    or n_hypothetical_batch_emails <= max_emails_per_batch
                )
                and (
                    max_threads_per_batch is None
                    or n_hypothetical_batch_threads <= max_threads_per_batch
                )
                and n_hypothetical_batch_prompt_tokens <= max_input_tokens
            )
            if fits:
                batch.append(thread_chunk)
            else:
                if len(batch) == 0:
                    batch = [thread_chunk]
                else:
                    batch_emails = [email for thread in batch for email in thread["emails"]]
                    batch_prompt = build_prompt_from_batch_threads(batch)
                    batches.append((batch, batch_prompt))
                    batch_statistics.append({
                        "n_emails": len(batch_emails),
                        "n_threads": len(batch),
                        "n_tokens": run_decoder_latest_tokenizer.remote(
                            [batch_prompt],
                            model_name_or_path,
                        )[0],
                    })
                    batch = [thread_chunk]

    if batch:
        batch_emails = [email for thread in batch for email in thread["emails"]]
        batch_prompt = build_prompt_from_batch_threads(batch)
        batches.append((batch, batch_prompt))
        batch_statistics.append({
            "n_emails": len(batch_emails),
            "n_threads": len(batch),
            "n_tokens": run_decoder_latest_tokenizer.remote(
                [batch_prompt],
                model_name_or_path,
            )[0],
        })

    if pre_decoder_statistics_plot_path:
        save_pre_decoder_statistics_plot(
            batch_statistics,
            split_statistics,
            title="Email curator batch vs split statistics",
            output_path=pre_decoder_statistics_plot_path,
            max_input_tokens=max_input_tokens,
            mode="email_curator",
        )

    async def run_batch(prompt):
        return await run_email_knowledge_base_curator.remote.aio(
            context=[],
            current_turn_input_text=prompt,
            current_turn_image_in_bytes=None,
            **email_knowledge_base_curator_profile_config,
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

    curator_statistics = {
        "n_input_threads": len(threads),
        "n_split_threads": n_split_threads,
        "n_thread_chunks": n_thread_chunks,
        "n_planned_batches": len(batches),
        "n_failed_exception_batches": 0,
        "n_failed_oom_batches": 0,
        "n_failed_timeout_batches": 0,
        "n_failed_empty_output_batches": 0,
        "n_short_count_mismatch_batches": 0,
        "n_long_count_mismatch_batches": 0,
        "n_no_useful_information_outputs": 0,
    }
    curated_thread_outputs = []

    for batch_index in range(len(batches)):
        batch_threads = batches[batch_index][0]
        batch_result = batch_results[batch_index]

        if isinstance(batch_result, Exception):
            curator_statistics["n_failed_exception_batches"] += 1
            batch_result_text = str(batch_result).lower()
            if "out of memory" in batch_result_text or "oom" in batch_result_text:
                curator_statistics["n_failed_oom_batches"] += 1
            if "timeout" in batch_result_text or "timed out" in batch_result_text:
                curator_statistics["n_failed_timeout_batches"] += 1
            continue

        curator_outputs, prompt_text = batch_result
        if curator_outputs is None:
            curator_statistics["n_failed_empty_output_batches"] += 1
            continue
        if len(curator_outputs) != len(batch_threads):
            if len(curator_outputs) < len(batch_threads):
                curator_statistics["n_short_count_mismatch_batches"] += 1
            else:
                curator_statistics["n_long_count_mismatch_batches"] += 1

        for i in range(min(len(curator_outputs), len(batch_threads))):
            thread_id = batch_threads[i]["thread_id"]
            curator_output = curator_outputs[i]
            curator_output["thread_id"] = thread_id
            if curator_output.get("no_useful_information"):
                curator_statistics["n_no_useful_information_outputs"] += 1
            curated_thread_outputs.append(curator_output)

    if post_decoder_statistics_plot_path:
        from helpers.curator import save_email_curator_usefulness_plot

        save_post_decoder_statistics_plot(
            n_rule_based_thread_count=len(threads),
            n_rule_based_threads_sent_to_lm=n_thread_chunks,
            n_planned_batches=len(batches),
            bypass_statistics=split_statistics,
            n_failed_exception_batches=curator_statistics["n_failed_exception_batches"],
            n_failed_oom_batches=curator_statistics["n_failed_oom_batches"],
            n_failed_timeout_batches=curator_statistics["n_failed_timeout_batches"],
            n_failed_empty_output_batches=curator_statistics["n_failed_empty_output_batches"],
            n_failed_short_output_batches=curator_statistics["n_short_count_mismatch_batches"],
            n_kept_exact_output_batches=(
                len(batches)
                - curator_statistics["n_failed_exception_batches"]
                - curator_statistics["n_failed_empty_output_batches"]
                - curator_statistics["n_short_count_mismatch_batches"]
                - curator_statistics["n_long_count_mismatch_batches"]
            ),
            n_kept_expanded_output_batches=0,
            title="Email curator post-decoder statistics",
            output_path=post_decoder_statistics_plot_path,
            mode="email_curator",
            n_failed_long_output_batches=curator_statistics["n_long_count_mismatch_batches"],
            n_split_threads=n_split_threads,
            n_thread_chunks=n_thread_chunks,
        )
        usefulness_plot_path = post_decoder_statistics_plot_path.with_name(
            f"{post_decoder_statistics_plot_path.stem}_usefulness"
            f"{post_decoder_statistics_plot_path.suffix}"
        )
        save_email_curator_usefulness_plot(
            n_input_threads=len(threads),
            n_split_threads=n_split_threads,
            n_thread_chunks=n_thread_chunks,
            n_curated_thread_chunks=len(curated_thread_outputs),
            n_no_useful_information_outputs=(
                curator_statistics["n_no_useful_information_outputs"]
            ),
            output_path=usefulness_plot_path,
        )

    if return_run_data:
        return {
            "batch_statistics": batch_statistics,
            "split_statistics": split_statistics,
            "curator_statistics": curator_statistics,
            "curated_thread_outputs": curated_thread_outputs,
        }

    return curated_thread_outputs

######################################################
# Helper 3: Build email knowledge base data variants #
######################################################
def build_email_thread_knowledge_base_chunks(curator_output):
    abstract_chunks = []
    summary_chunks = []
    cleaned_text_chunks = []
    q_and_a_chunks = []

    thread_id = curator_output.get("thread_id")

    abstract = curator_output.get("abstract")
    if abstract:
        abstract_chunks.append({
            "thread_id": thread_id,
            "text": abstract,
        })

    summary = curator_output.get("summary")
    if summary:
        summary_chunks.append({
            "thread_id": thread_id,
            "text": summary,
        })

    cleanedtext = curator_output.get("cleanedtext")
    if cleanedtext:
        cleaned_text_chunks.append({
            "thread_id": thread_id,
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
            "thread_id": thread_id,
            "pairs": pairs,
        })

    return {
        "lm_abstract_chunks": abstract_chunks,
        "lm_summary_chunks": summary_chunks,
        "lm_cleaned_text_chunks": cleaned_text_chunks,
        "lm_q_and_a_chunks": q_and_a_chunks,
    }

#####################################################
# Helper 4: Save email knowledge-base curator plots #
#####################################################
def save_email_knowledge_base_curator_plots(
    curator_run_data,
    max_input_tokens,
    pre_curator_statistics_plot_path,
    post_curator_statistics_plot_path,
):
    from helpers.data import (
        save_post_decoder_statistics_plot,
        save_pre_decoder_statistics_plot,
    )

    batch_statistics = curator_run_data["batch_statistics"]
    split_statistics = curator_run_data["split_statistics"]
    curator_statistics = curator_run_data["curator_statistics"]

    save_pre_decoder_statistics_plot(
        batch_statistics,
        split_statistics,
        title="Email curator batch vs split statistics",
        output_path=pre_curator_statistics_plot_path,
        max_input_tokens=max_input_tokens,
        mode="email_curator",
    )
    save_post_decoder_statistics_plot(
        n_rule_based_thread_count=curator_statistics["n_input_threads"],
        n_rule_based_threads_sent_to_lm=curator_statistics["n_thread_chunks"],
        n_planned_batches=curator_statistics["n_planned_batches"],
        bypass_statistics=split_statistics,
        n_failed_exception_batches=curator_statistics["n_failed_exception_batches"],
        n_failed_oom_batches=curator_statistics["n_failed_oom_batches"],
        n_failed_timeout_batches=curator_statistics["n_failed_timeout_batches"],
        n_failed_empty_output_batches=curator_statistics["n_failed_empty_output_batches"],
        n_failed_short_output_batches=curator_statistics["n_short_count_mismatch_batches"],
        n_kept_exact_output_batches=(
            curator_statistics["n_planned_batches"]
            - curator_statistics["n_failed_exception_batches"]
            - curator_statistics["n_failed_empty_output_batches"]
            - curator_statistics["n_short_count_mismatch_batches"]
            - curator_statistics["n_long_count_mismatch_batches"]
        ),
        n_kept_expanded_output_batches=0,
        title="Email curator post-decoder statistics",
        output_path=post_curator_statistics_plot_path,
        mode="email_curator",
        n_failed_long_output_batches=curator_statistics["n_long_count_mismatch_batches"],
        n_split_threads=curator_statistics["n_split_threads"],
        n_thread_chunks=curator_statistics["n_thread_chunks"],
    )
    usefulness_plot_path = post_curator_statistics_plot_path.with_name(
        f"{post_curator_statistics_plot_path.stem}_usefulness"
        f"{post_curator_statistics_plot_path.suffix}"
    )
    save_email_curator_usefulness_plot(
        n_input_threads=curator_statistics["n_input_threads"],
        n_split_threads=curator_statistics["n_split_threads"],
        n_thread_chunks=curator_statistics["n_thread_chunks"],
        n_curated_thread_chunks=curator_statistics["n_curated_thread_outputs"],
        n_no_useful_information_outputs=(
            curator_statistics["n_no_useful_information_outputs"]
        ),
        output_path=usefulness_plot_path,
    )

########################################################################
# Helper 5: Save email-curator post-decoder info/no-info plot as image #
########################################################################
def save_email_curator_usefulness_plot(
        n_input_threads,
        n_split_threads,
        n_thread_chunks,
        n_curated_thread_chunks,
        n_no_useful_information_outputs,
        output_path,
        ):
    from pathlib import Path
    import matplotlib.pyplot as plt

    from helpers.data import lighten_hex_color

    if n_no_useful_information_outputs > n_curated_thread_chunks:
        raise ValueError(
            "save_email_curator_usefulness_plot: invalid no-useful-information counts:\n"
            f"\tn_no_useful_information_outputs={n_no_useful_information_outputs}\n"
            f"\tn_curated_thread_chunks={n_curated_thread_chunks}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_useful_thread_chunks = (
        n_curated_thread_chunks
        - n_no_useful_information_outputs
    )
    fig, ax = plt.subplots(figsize=(14.5, 4.8))
    fig.suptitle(
        "Email curator thread-chunk usefulness\n"
        f"({n_curated_thread_chunks} curated thread chunks)",
        fontsize=17,
    )

    segments = [
        ("Useful", n_useful_thread_chunks, lighten_hex_color("#8BC34A")),
        ("No useful-info", n_no_useful_information_outputs, lighten_hex_color("#607D8B")),
    ]
    left = 0
    for label, value, color in segments:
        ax.barh(
            ["Curated thread chunks"],
            [value],
            left=left,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            label=f"{label} ({value})",
        )
        if value > 0:
            x_position = left + (value / 2)
            ax.text(
                x_position,
                0,
                str(value),
                ha="center",
                va="center",
                color="white" if value >= 10 else "black",
                fontweight="bold" if value >= 10 else None,
                fontsize=11,
                clip_on=True,
            )
        left += value

    ax.set_xlim(0, max(1, n_curated_thread_chunks))
    ax.set_xlabel("Thread-chunk count", fontsize=13)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 0.86),
        bbox_transform=fig.transFigure,
        ncol=2,
        fontsize=12,
        frameon=False,
    )

    summary_text = (
        f"Input threads total: {n_input_threads}\n"
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

    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.23, top=0.70)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    print(f"save_email_curator_usefulness_plot: saved plot to {output_path}")
