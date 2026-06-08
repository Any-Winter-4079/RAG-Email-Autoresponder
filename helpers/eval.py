import json
import statistics
from pathlib import Path

from config.data import (
    EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX,
    SPLIT_DATASETS_DIR,
)
from config.eval import (
    DATA_VARIANT_EVAL_SOURCES,
    DATA_VARIANT_TEST_EVAL_VARIANTS,
)
from helpers.data import lighten_hex_color
from helpers.general import (
    get_context_emails_mode_suffix,
    get_n_eval_samples_per_folder_uri_suffix,
    resolve_data_variant_eval_output_path,
    resolve_dumped_collection_payloads_path,
)

#######################################
# Helper 1: Write eval output to file #
#######################################
def write_eval_output_to_file(data_variant_results_dir, output_name, eval_output, data_variant):
    json_path = data_variant_results_dir / f"{output_name}.json"
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(eval_output, json_file, ensure_ascii=False, indent=2)
    print(f"\twrote {data_variant}/{json_path.name}")
    print()

###########################################
# Helper 2: Build source query from email #
###########################################
def build_source_query(sample):
    source_subject = (sample["email"].get("subject") or "").strip()
    source_body = (sample["email"].get("body") or "").strip()
    return f"Subject:\n{source_subject}\n\nBody:\n{source_body}"

###################################################
# Helper 3: Get rerank text from collection point #
###################################################
def get_text_from_payload(payload):
    if "question" in payload and "answer" not in payload:
        return payload["question"]
    if "question" in payload and "answer" in payload:
        return f"Q: {payload['question']}\nA: {payload['answer']}"
    return payload["text"]

######################################################
# Helper 4: Load selected split samples for eval run #
######################################################
def load_selected_split_samples(
        project_root,
        split_name,
        context_emails_mode,
        n_eval_samples_per_folder_uri,
        ):
    split_samples_path = Path(project_root) / SPLIT_DATASETS_DIR / f"{split_name}.json"
    with open(split_samples_path, "r", encoding="utf-8") as split_samples_file:
        all_split_samples = json.load(split_samples_file)

    if context_emails_mode == "without_context":
        all_split_samples = [
            sample
            for sample in all_split_samples
            if not sample["context_emails"]
        ]
    elif context_emails_mode == "with_context":
        all_split_samples = [
            sample
            for sample in all_split_samples
            if sample["context_emails"]
        ]

    all_folder_uris = sorted({sample["folder_uri"] for sample in all_split_samples})
    folder_uri_to_n_selected = {
        folder_uri: 0
        for folder_uri in all_folder_uris
    }
    if n_eval_samples_per_folder_uri is None:
        for sample in all_split_samples:
            folder_uri_to_n_selected[sample["folder_uri"]] += 1
        return all_split_samples, folder_uri_to_n_selected

    split_samples = []
    for sample in all_split_samples:
        folder_uri = sample["folder_uri"]
        if folder_uri_to_n_selected[folder_uri] >= n_eval_samples_per_folder_uri:
            continue
        split_samples.append(sample)
        folder_uri_to_n_selected[folder_uri] += 1
        if all(
            n_selected >= n_eval_samples_per_folder_uri
            for n_selected in folder_uri_to_n_selected.values()
        ):
            break

    return split_samples, folder_uri_to_n_selected

##################################################################
# Helper 5: Save dumped collection size-comparison plot as image #
##################################################################
def save_collection_dump_size_comparison_plot(
        collection_name_to_n_points,
        output_path,
        title="Dumped collection sizes",
        ):
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_variants = []
    base_variant_to_counts = {}
    for collection_name, n_points in collection_name_to_n_points.items():
        if collection_name.startswith(EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX):
            source = "email"
            base_variant = collection_name.removeprefix(
                EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX
            )
        else:
            source = "web"
            base_variant = collection_name

        if base_variant not in base_variant_to_counts:
            base_variant_to_counts[base_variant] = {
                "web": 0,
                "email": 0,
            }
            base_variants.append(base_variant)
        base_variant_to_counts[base_variant][source] = n_points

    if not base_variants:
        raise ValueError(
            "save_collection_dump_size_comparison_plot: no collection sizes provided"
        )

    web_color = lighten_hex_color("#00CBBF")
    email_color = lighten_hex_color("#9DCA1C")
    y_positions = list(range(len(base_variants)))
    fig_height = max(3.8, 1.15 * len(base_variants) + 1.6)
    fig, ax = plt.subplots(figsize=(10.8, fig_height))

    for y_position, base_variant in zip(y_positions, base_variants):
        web_count = base_variant_to_counts[base_variant]["web"]
        email_count = base_variant_to_counts[base_variant]["email"]
        total_count = web_count + email_count
        if total_count == 0:
            continue

        left = 0
        for source_label, count, color in [
            ("Web", web_count, web_color),
            ("Email", email_count, email_color),
        ]:
            if count == 0:
                continue
            percentage = 100.0 * count / total_count
            ax.barh(
                [y_position],
                [percentage],
                left=left,
                color=color,
                edgecolor="white",
                linewidth=0.8,
            )
            ax.text(
                left + percentage / 2,
                y_position,
                f"{source_label}\n{count} ({percentage:.1f}%)",
                ha="center",
                va="center",
                color="white",
                fontsize=11,
                fontweight="bold",
            )
            left += percentage

    ax.set_xlim(0, 100)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(base_variants, fontsize=11)
    ax.invert_yaxis()
    ax.set_xticks(range(0, 101, 10))
    ax.set_xlabel("Percentage of points", fontsize=12, labelpad=12)
    ax.set_title(title, fontsize=15, pad=16)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="x", labelsize=11, pad=6)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=web_color),
        plt.Rectangle((0, 0), 1, 1, color=email_color),
    ]
    ax.legend(
        legend_handles,
        ["Web", "Email"],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.72),
        ncol=2,
        fontsize=11,
        frameon=False,
    )

    fig.subplots_adjust(bottom=0.42, top=0.80)
    fig.tight_layout(rect=(0, 0.16, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

################################################################
# Helper 6: Get token count from one dumped collection payload #
################################################################
def get_collection_point_token_count(payload, token_type):
    if token_type not in {"decoder", "encoder"}:
        raise ValueError(
            "get_collection_point_token_count: unsupported token type:\n"
            f"\t{token_type}"
        )

    direct_key = f"{token_type}_token_count"
    if direct_key in payload:
        return payload[direct_key]

    q_key = f"{token_type}_token_count_q"
    a_key = f"{token_type}_token_count_a"
    if q_key not in payload:
        raise ValueError(
            "get_collection_point_token_count: payload does not contain token counts:\n"
            f"\ttoken type: {token_type}\n"
            f"\tpayload keys: {sorted(payload.keys())}"
        )

    variant = str(payload.get("variant") or "")
    base_variant = variant.removeprefix(EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX)
    if base_variant == "lm_q_and_a_for_q_only_chunks":
        return payload[q_key]

    if a_key in payload:
        return payload[q_key] + payload[a_key]

    if "answer" not in payload or not str(payload.get("answer") or "").strip():
        return payload[q_key]

    raise ValueError(
        "get_collection_point_token_count: Q&A payload is missing answer token count:\n"
        f"\tvariant: {variant}\n"
        f"\tpayload keys: {sorted(payload.keys())}"
    )

###################################################################
# Helper 7: Build summary statistics for dumped collection tokens #
###################################################################
def build_collection_dump_token_statistics(collection_name_to_token_counts):
    collection_name_to_stats = {}
    for collection_name, token_counts in collection_name_to_token_counts.items():
        if not token_counts:
            collection_name_to_stats[collection_name] = {
                "count": 0,
                "min": None,
                "q1": None,
                "median": None,
                "mean": None,
                "q3": None,
                "max": None,
            }
            continue

        if len(token_counts) == 1:
            q1 = median = q3 = token_counts[0]
        else:
            q1, _, q3 = statistics.quantiles(
                token_counts,
                n=4,
                method="inclusive",
            )
            median = statistics.median(token_counts)

        collection_name_to_stats[collection_name] = {
            "count": len(token_counts),
            "min": min(token_counts),
            "q1": q1,
            "median": median,
            "mean": statistics.fmean(token_counts),
            "q3": q3,
            "max": max(token_counts),
        }

    return collection_name_to_stats

#####################################################################
# Helper 8: Save dumped collection token-distribution plot as image #
#####################################################################
def save_collection_dump_token_distribution_plot(
        collection_name_to_token_counts,
        output_path,
        token_type,
        title=None,
        x_axis_max=None,
        ):
    import matplotlib.pyplot as plt

    if token_type not in {"decoder", "encoder"}:
        raise ValueError(
            "save_collection_dump_token_distribution_plot: unsupported token type:\n"
            f"\t{token_type}"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    base_variants = []
    base_variant_to_counts = {}
    for collection_name, token_counts in collection_name_to_token_counts.items():
        if collection_name.startswith(EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX):
            source = "email"
            base_variant = collection_name.removeprefix(
                EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX
            )
        else:
            source = "web"
            base_variant = collection_name

        if base_variant not in base_variant_to_counts:
            base_variant_to_counts[base_variant] = {
                "web": [],
                "email": [],
            }
            base_variants.append(base_variant)
        base_variant_to_counts[base_variant][source] = token_counts

    if not base_variants:
        raise ValueError(
            "save_collection_dump_token_distribution_plot: no token counts provided"
        )

    collection_name_to_stats = build_collection_dump_token_statistics(
        collection_name_to_token_counts=collection_name_to_token_counts
    )

    web_color = lighten_hex_color("#00CBBF")
    email_color = lighten_hex_color("#9DCA1C")
    y_base_positions = list(range(1, len(base_variants) + 1))
    position_offset = 0.22
    box_width = 0.34
    all_token_counts = [
        token_count
        for token_counts in collection_name_to_token_counts.values()
        for token_count in token_counts
    ]
    if not all_token_counts:
        raise ValueError(
            "save_collection_dump_token_distribution_plot: all token-count lists are empty"
        )

    data_max = max(all_token_counts)
    if x_axis_max is None:
        x_axis_max = data_max
    else:
        x_axis_max = max(x_axis_max, data_max)
    left_label_pad = max(110, round(x_axis_max * 0.08))
    x_margin = max(8, round(x_axis_max * 0.07))
    fig_height = max(4.2, 1.1 * len(base_variants) + 1.9)
    fig, ax = plt.subplots(figsize=(11.8, fig_height))

    def draw_source_boxplot(token_counts, y_position, color):
        boxplot = ax.boxplot(
            [token_counts],
            positions=[y_position],
            widths=box_width,
            vert=False,
            patch_artist=True,
            whis=(0, 100),
            showmeans=True,
            showfliers=False,
            manage_ticks=False,
            medianprops={
                "color": "#1F1F1F",
                "linewidth": 1.2,
            },
            whiskerprops={
                "color": "#6B6B6B",
                "linewidth": 1.0,
            },
            capprops={
                "color": "#6B6B6B",
                "linewidth": 1.0,
            },
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": "#1F1F1F",
                "markersize": 5,
            },
        )
        for patch in boxplot["boxes"]:
            patch.set_facecolor(color)
            patch.set_edgecolor("white")
            patch.set_linewidth(0.9)

    for y_base_position, base_variant in zip(y_base_positions, base_variants):
        for source, color, y_position in [
            ("web", web_color, y_base_position - position_offset),
            ("email", email_color, y_base_position + position_offset),
        ]:
            token_counts = base_variant_to_counts[base_variant][source]
            if not token_counts:
                continue

            draw_source_boxplot(token_counts, y_position, color)

            collection_name = (
                base_variant
                if source == "web"
                else f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}{base_variant}"
            )
            stats = collection_name_to_stats[collection_name]
            ax.text(
                -left_label_pad * 0.84,
                y_position,
                source.title(),
                ha="left",
                va="center",
                fontsize=11,
                fontweight="bold",
                color="#4B4B4B",
                clip_on=False,
            )
            ax.text(
                stats["max"] + x_margin * 0.18,
                y_position,
                f"n={stats['count']:,} | mean={stats['mean']:.1f}",
                ha="left",
                va="center",
                fontsize=9.5,
                color="#4B4B4B",
            )

    if title is None:
        title = (
            f"Dumped collection {token_type} token distributions\n"
            "(box = IQR, whiskers = min/max, diamond = mean)"
        )

    ax.set_xlim(-left_label_pad, x_axis_max + x_margin)
    ax.set_yticks(y_base_positions)
    ax.set_yticklabels(base_variants, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel("Tokens per chunk", fontsize=12, labelpad=12)
    ax.set_title(title, fontsize=15, pad=16)
    ax.grid(axis="x", alpha=0.25)
    ax.tick_params(axis="x", labelsize=11, pad=6)

    fig.subplots_adjust(bottom=0.20, top=0.84)
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

######################################################
# Helper 9: Save query-rewrite summary plot as image #
######################################################
def save_query_rewrite_summary_plot(
        n_emails,
        n_query_rewriter_exceptions,
        n_empty_query_rewrite_outputs,
        n_no_usable_query_rewrite_outputs,
        n_no_reranker_query_outputs,
        n_no_request_emails,
        n_rewritten_emails,
        n_duplicate_queries_removed,
        n_query_cap_hits,
        n_capped_queries_removed,
        output_path,
        title="Query rewrite summary",
        ):
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    email_labels = [
        "input emails",
        "request emails",
        "no-request emails",
    ]
    email_counts = [
        n_emails,
        n_rewritten_emails,
        n_no_request_emails,
        n_query_rewriter_exceptions,
        n_empty_query_rewrite_outputs,
        n_no_usable_query_rewrite_outputs,
        n_no_reranker_query_outputs,
    ]
    email_labels.extend([
        "rewrite exceptions",
        "empty rewrite outputs",
        "no-usable rewrite outputs",
        "missing reranker queries",
    ])
    query_labels = [
        "duplicate queries removed",
        "query cap hits",
        "queries removed for cap",
    ]
    query_counts = [
        n_duplicate_queries_removed,
        n_query_cap_hits,
        n_capped_queries_removed,
    ]
    n_failed_query_rewrites = (
        n_query_rewriter_exceptions
        + n_empty_query_rewrite_outputs
        + n_no_usable_query_rewrite_outputs
        + n_no_reranker_query_outputs
    )
    error_rate = 0.0 if not n_emails else n_failed_query_rewrites / n_emails

    fig, axes = plt.subplots(1, 2, figsize=(12.2, 5.0))
    email_color = lighten_hex_color("#00CBBF")
    query_color = lighten_hex_color("#9DCA1C")

    def draw_summary_bars(ax, labels, counts, color, xlabel):
        max_count = max(counts) if counts else 0
        ax.barh(labels, counts, color=color, edgecolor="white", linewidth=0.8)
        for label, count in zip(labels, counts):
            ax.text(
                count + max(0.02 * max(max_count, 1), 0.15),
                label,
                f"{count:,}",
                va="center",
                ha="left",
                fontsize=10.5,
                color="#3F3F3F",
            )
        ax.set_xlim(0, max(1, max_count * 1.18))
        ax.set_xlabel(xlabel, fontsize=11, labelpad=10)
        ax.grid(axis="x", alpha=0.22)
        ax.tick_params(axis="y", labelsize=10.5)
        ax.tick_params(axis="x", labelsize=10.5, pad=5)
        ax.invert_yaxis()

    draw_summary_bars(axes[0], email_labels, email_counts, email_color, "Email count")
    axes[0].set_title("Email-level counts", fontsize=12.5, pad=12)

    draw_summary_bars(axes[1], query_labels, query_counts, query_color, "Query count")
    axes[1].set_title("Query-level counts", fontsize=12.5, pad=12)

    fig.suptitle(
        f"{title}\nerror rate {error_rate:.2%}",
        fontsize=15,
        y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

###################################################
# Helper 10: Save retrieval summary plot as image #
###################################################
def save_retrieval_summary_plot(
        label_to_n_failed_emails,
        label_to_n_empty_result_emails,
        output_path,
        title="Retrieval summary",
        ):
    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = list(label_to_n_failed_emails)
    failed_counts = [
        label_to_n_failed_emails[label]
        for label in labels
    ]
    empty_counts = [
        label_to_n_empty_result_emails.get(label, 0)
        for label in labels
    ]

    fig_height = max(4.8, 0.42 * max(len(labels), 1) + 1.8)
    fig, axes = plt.subplots(1, 2, figsize=(14.2, fig_height))
    failed_color = lighten_hex_color("#FF7F50")
    empty_color = lighten_hex_color("#00CBBF")

    def draw_summary_bars(ax, counts, color, subtitle):
        max_count = max(counts) if counts else 0
        ax.barh(labels, counts, color=color, edgecolor="white", linewidth=0.8)
        for label, count in zip(labels, counts):
            ax.text(
                count + max(0.02 * max(max_count, 1), 0.15),
                label,
                f"{count:,}",
                va="center",
                ha="left",
                fontsize=9.8,
                color="#3F3F3F",
            )
        ax.set_xlim(0, max(1, max_count * 1.18))
        ax.set_title(subtitle, fontsize=12.5, pad=12)
        ax.grid(axis="x", alpha=0.22)
        ax.tick_params(axis="y", labelsize=9.6)
        ax.tick_params(axis="x", labelsize=10.2, pad=5)
        ax.invert_yaxis()

    draw_summary_bars(axes[0], failed_counts, failed_color, "Failed emails")
    draw_summary_bars(axes[1], empty_counts, empty_color, "Empty result emails")

    fig.suptitle(title, fontsize=15, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)

######################################################
# Helper 11: Build retrieval email from split sample #
######################################################
def build_retrieval_email_from_split_sample(split_sample):
    return {
        "from": split_sample["email"].get("author"),
        "to": split_sample["email"].get("recipients"),
        "date": None,
        "subject": split_sample["email"].get("subject"),
        "message_body": split_sample["email"].get("body"),
        "context_emails": split_sample.get("context_emails", []),
    }

####################################################################
# Helper 12: Build base data variant to source to encoder settings #
####################################################################
def build_base_data_variant_to_source_to_encoder_settings():
    base_data_variant_to_source_to_encoder_settings = {}
    for base_data_variant, eval_variant_settings in DATA_VARIANT_TEST_EVAL_VARIANTS.items():
        base_data_variant_to_source_to_encoder_settings[base_data_variant] = {}
        for source_name in DATA_VARIANT_EVAL_SOURCES:
            if source_name == "web":
                concrete_data_variant = base_data_variant
            elif source_name == "email":
                if not base_data_variant.startswith("lm_"):
                    continue
                concrete_data_variant = f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}{base_data_variant}"
            else:
                raise ValueError(
                    "run_data_variant_eval: unsupported eval source:\n"
                    f"\t{source_name}"
                )
            base_data_variant_to_source_to_encoder_settings[base_data_variant][source_name] = {
                encoder_name: {
                    **encoder_settings,
                    "collection_name": concrete_data_variant,
                }
                for encoder_name, encoder_settings in eval_variant_settings["encoders"].items()
            }
    return base_data_variant_to_source_to_encoder_settings

########################################################
# Helper 13: Select split samples for retrieval emails #
########################################################
def select_split_samples_for_retrieval_emails(
        target_retrieval_emails,
        retrieval_emails,
        split_samples,
        ):
    matched_split_samples = []
    target_email_index = 0
    for retrieval_email, split_sample in zip(retrieval_emails, split_samples):
        if target_email_index >= len(target_retrieval_emails):
            break
        if retrieval_email is target_retrieval_emails[target_email_index]:
            matched_split_samples.append(split_sample)
            target_email_index += 1

    if target_email_index != len(target_retrieval_emails):
        raise ValueError(
            "select_split_samples_for_retrieval_emails: could not align retrieval emails with split samples"
        )

    return matched_split_samples

#######################################################
# Helper 14: Attach split samples to retrieval output #
#######################################################
def attach_split_samples_to_retrieval_output(retrieval_output, rewritten_split_samples):
    if len(retrieval_output["results"]) != len(rewritten_split_samples):
        raise ValueError(
            "attach_split_samples_to_retrieval_output: retrieval results do not match rewritten split sample count"
        )

    results_with_samples = []
    for split_sample, result in zip(rewritten_split_samples, retrieval_output["results"]):
        result_with_sample = {
            "query_type_to_rewritten_queries": result["query_type_to_rewritten_queries"],
            "reranker_query": result["reranker_query"],
            "sample": split_sample,
            "retrieval_failed": result["retrieval_failed"],
        }
        if "retrieval_results" in result:
            result_with_sample["retrieval_results"] = result["retrieval_results"]
        results_with_samples.append(result_with_sample)

    return {
        **{
            key: value
            for key, value in retrieval_output.items()
            if key != "results"
        },
        "results": results_with_samples,
    }

################################################
# Helper 15: Get number of empty-result emails #
################################################
def get_n_empty_result_emails(retrieval_output):
    return sum(
        1
        for result in retrieval_output["results"]
        if not result["retrieval_failed"] and not result["retrieval_results"]
    )

############################################
# Helper 16: Get query rewrite cache paths #
############################################
def get_query_rewrite_cache_paths(
        project_root,
        query_rewrite_cache_dir,
        split_name,
        context_emails_mode,
        n_eval_samples_per_folder_uri,
        configured_cache_filename,
        ):
    from config.decoder import QUERY_REWRITER_CACHE_TAG

    cache_dir = Path(project_root) / query_rewrite_cache_dir
    if configured_cache_filename is not None:
        query_rewrite_cache_path = cache_dir / configured_cache_filename
        no_requests_cache_path = query_rewrite_cache_path.with_name(
            f"{query_rewrite_cache_path.stem}_no_requests"
            f"{query_rewrite_cache_path.suffix}"
        )
    else:
        filter_mode_suffix = get_context_emails_mode_suffix(context_emails_mode)
        n_eval_samples_suffix = get_n_eval_samples_per_folder_uri_suffix(
            n_eval_samples_per_folder_uri
        )
        query_rewrite_cache_path = (
            cache_dir
            / (
                f"{split_name}{filter_mode_suffix}_"
                f"{QUERY_REWRITER_CACHE_TAG}{n_eval_samples_suffix}.json"
            )
        )
        no_requests_cache_path = (
            cache_dir
            / (
                f"{split_name}{filter_mode_suffix}_"
                f"{QUERY_REWRITER_CACHE_TAG}{n_eval_samples_suffix}_no_requests.json"
            )
        )

    return query_rewrite_cache_path, no_requests_cache_path

################################################
# Helper 17: Group query entries by query type #
################################################
def group_query_entries_by_query_type(query_entries, reranker_query):
    query_type_to_rewritten_queries = {}
    for query_entry in query_entries:
        query_type = query_entry["query_type"]
        if query_type not in query_type_to_rewritten_queries:
            query_type_to_rewritten_queries[query_type] = []
        query_type_to_rewritten_queries[query_type].append(query_entry["query"])

    if reranker_query:
        if "reranker" not in query_type_to_rewritten_queries:
            query_type_to_rewritten_queries["reranker"] = []
        if reranker_query not in query_type_to_rewritten_queries["reranker"]:
            query_type_to_rewritten_queries["reranker"].append(reranker_query)

    return query_type_to_rewritten_queries

#######################################################
# Helper 18: Build rewrite summary from cache entries #
#######################################################
def build_rewrite_summary_from_cache_entries(request_entries, no_requests):
    return {
        "rewritten_emails": [
            {
                "email": build_retrieval_email_from_split_sample(request_entry["sample"]),
                "query_type_to_rewritten_queries": group_query_entries_by_query_type(
                    request_entry["queries"],
                    request_entry["reranker_query"],
                ),
                "reranker_query": request_entry["reranker_query"],
                "anonymized_request": request_entry["anonymized_request"],
            }
            for request_entry in request_entries
        ],
        "no_request_emails": [
            build_retrieval_email_from_split_sample(no_request["sample"])
            for no_request in no_requests
        ],
        "n_query_rewriter_exceptions": 0,
        "n_empty_query_rewrite_outputs": 0,
        "n_no_usable_query_rewrite_outputs": 0,
        "n_no_reranker_query_outputs": 0,
        "n_no_anonymized_request_outputs": 0,
        "n_duplicate_queries_removed": 0,
        "n_query_cap_hits": 0,
        "n_capped_queries_removed": 0,
    }

################################################
# Helper 19: Build query rewrite cache entries #
################################################
def build_query_rewrite_cache_entries(rewrite_summary, retrieval_emails, split_samples):
    request_split_samples = select_split_samples_for_retrieval_emails(
        target_retrieval_emails=[
            rewritten_email["email"]
            for rewritten_email in rewrite_summary["rewritten_emails"]
        ],
        retrieval_emails=retrieval_emails,
        split_samples=split_samples,
    )
    no_request_split_samples = select_split_samples_for_retrieval_emails(
        target_retrieval_emails=rewrite_summary["no_request_emails"],
        retrieval_emails=retrieval_emails,
        split_samples=split_samples,
    )

    request_entries = []
    for split_sample, rewritten_email in zip(
            request_split_samples,
            rewrite_summary["rewritten_emails"]):
        request_entries.append({
            "sample": split_sample,
            "queries": [
                {
                    "query": query,
                    "query_type": query_type,
                }
                for query_type, queries in (
                    rewritten_email["query_type_to_rewritten_queries"].items()
                )
                for query in queries
            ],
            "reranker_query": rewritten_email["reranker_query"],
            "anonymized_request": rewritten_email["anonymized_request"],
        })
    no_requests = [
        {"sample": split_sample}
        for split_sample in no_request_split_samples
    ]

    return request_entries, no_requests

################################################
# Helper 20: Load or create query rewrite data #
################################################
def load_or_create_query_rewrite_data(
        project_root,
        query_rewrite_cache_dir,
        split_name,
        context_emails_mode,
        n_eval_samples_per_folder_uri,
        configured_cache_filename=None,
        max_query_rewrite_failure_rate=None,
        log_prefix="run_data_variant_eval",
        ):
    import asyncio

    from helpers.retrieval_pipeline import rewrite_emails_async

    query_rewrite_cache_path, no_requests_cache_path = get_query_rewrite_cache_paths(
        project_root=project_root,
        query_rewrite_cache_dir=query_rewrite_cache_dir,
        split_name=split_name,
        context_emails_mode=context_emails_mode,
        n_eval_samples_per_folder_uri=n_eval_samples_per_folder_uri,
        configured_cache_filename=configured_cache_filename,
    )
    split_samples, folder_uri_to_n_split_samples = load_selected_split_samples(
        project_root=project_root,
        split_name=split_name,
        context_emails_mode=context_emails_mode,
        n_eval_samples_per_folder_uri=n_eval_samples_per_folder_uri,
    )
    can_load_cache = False
    if query_rewrite_cache_path.exists() and no_requests_cache_path.exists():
        with open(query_rewrite_cache_path, "r", encoding="utf-8") as query_rewrite_cache_file:
            request_entries = json.load(query_rewrite_cache_file)
        with open(no_requests_cache_path, "r", encoding="utf-8") as no_requests_cache_file:
            no_requests = json.load(no_requests_cache_file)
        can_load_cache = all(
            isinstance(request_entry.get("anonymized_request"), str)
            and request_entry["anonymized_request"].strip()
            for request_entry in request_entries
        )

    if can_load_cache:
        rewrite_summary = build_rewrite_summary_from_cache_entries(
            request_entries=request_entries,
            no_requests=no_requests,
        )
        did_create_cache = False
    else:
        retrieval_emails = [
            build_retrieval_email_from_split_sample(split_sample)
            for split_sample in split_samples
        ]
        rewrite_summary = asyncio.run(
            rewrite_emails_async(
                emails=retrieval_emails,
                log_prefix=log_prefix,
            )
        )
        n_query_rewrite_failures = (
            rewrite_summary["n_query_rewriter_exceptions"]
            + rewrite_summary["n_empty_query_rewrite_outputs"]
            + rewrite_summary["n_no_usable_query_rewrite_outputs"]
            + rewrite_summary["n_no_reranker_query_outputs"]
            + rewrite_summary["n_no_anonymized_request_outputs"]
        )
        query_rewrite_failure_rate = (
            0.0 if not retrieval_emails else n_query_rewrite_failures / len(retrieval_emails)
        )
        if (
                max_query_rewrite_failure_rate is not None
                and query_rewrite_failure_rate > max_query_rewrite_failure_rate):
            raise RuntimeError(
                f"{log_prefix}: query rewrite failure rate exceeded threshold:\n"
                f"\tfailure rate: {query_rewrite_failure_rate:.2%}\n"
                f"\tmax failure rate: {max_query_rewrite_failure_rate:.2%}"
            )

        request_entries, no_requests = build_query_rewrite_cache_entries(
            rewrite_summary=rewrite_summary,
            retrieval_emails=retrieval_emails,
            split_samples=split_samples,
        )
        query_rewrite_cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(query_rewrite_cache_path, "w", encoding="utf-8") as query_rewrite_cache_file:
            json.dump(request_entries, query_rewrite_cache_file, ensure_ascii=False, indent=2)
        with open(no_requests_cache_path, "w", encoding="utf-8") as no_requests_cache_file:
            json.dump(no_requests, no_requests_cache_file, ensure_ascii=False, indent=2)
        did_create_cache = True

    return {
        "split_samples": split_samples,
        "folder_uri_to_n_split_samples": folder_uri_to_n_split_samples,
        "request_entries": request_entries,
        "no_requests": no_requests,
        "rewrite_summary": rewrite_summary,
        "rewritten_split_samples": [
            request_entry["sample"]
            for request_entry in request_entries
        ],
        "query_rewrite_cache_path": query_rewrite_cache_path,
        "no_requests_cache_path": no_requests_cache_path,
        "did_create_cache": did_create_cache,
    }

#########################################
# Helper 21: Load query-rewrite entries #
#########################################
def load_query_rewrite_entries(
        project_root,
        query_rewrite_cache_dir,
        split_name,
        context_emails_mode,
        n_eval_samples_per_folder_uri,
        configured_cache_filename,
        max_samples,
        ):
    query_rewrite_data = load_or_create_query_rewrite_data(
        project_root=project_root,
        query_rewrite_cache_dir=query_rewrite_cache_dir,
        split_name=split_name,
        context_emails_mode=context_emails_mode,
        n_eval_samples_per_folder_uri=n_eval_samples_per_folder_uri,
        configured_cache_filename=configured_cache_filename,
        log_prefix="load_query_rewrite_entries",
    )
    request_entries = query_rewrite_data["request_entries"]

    if max_samples is not None:
        request_entries = request_entries[:max_samples]

    return request_entries, query_rewrite_data["query_rewrite_cache_path"]

#####################################################
# Helper 22: Build oracle chunks from dumped points #
#####################################################
def build_oracle_chunks_from_dumped_points(dumped_points, source_name, collection_name):
    chunks = []
    id_to_chunk = {}
    for dumped_point in dumped_points:
        chunk_id = f"{source_name}_{dumped_point['id']}"
        chunks.append({
            "id": chunk_id,
            "text": get_text_from_payload(
                dumped_point["payload"]
            ).strip(),
        })
        id_to_chunk[chunk_id] = {
            **dumped_point,
            "id": chunk_id,
            "raw_id": dumped_point["id"],
            "source_name": source_name,
            "collection_name": collection_name,
        }

    return chunks, id_to_chunk

############################################
# Helper 23: Load corpus oracle input data #
############################################
def load_corpus_oracle_inputs(
        project_root,
        query_rewrite_cache_dir,
        split_name,
        context_emails_mode,
        n_eval_samples_per_folder_uri,
        configured_cache_filename,
        max_samples,
        data_variant,
        data_sources,
        dump_script_name,
        dump_timestamp,
        ):
    request_entries, query_rewrite_cache_path = load_query_rewrite_entries(
        project_root=project_root,
        query_rewrite_cache_dir=query_rewrite_cache_dir,
        split_name=split_name,
        context_emails_mode=context_emails_mode,
        n_eval_samples_per_folder_uri=n_eval_samples_per_folder_uri,
        configured_cache_filename=configured_cache_filename,
        max_samples=max_samples,
    )
    collection_names = []
    collection_dump_paths = []
    chunks = []
    id_to_chunk = {}
    for data_source in data_sources:
        collection_name = (
            f"{EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX}{data_variant}"
            if (
                data_source == "email"
                and
                not data_variant.startswith(EMAIL_KNOWLEDGE_BASE_VARIANT_PREFIX)
            )
            else data_variant
        )
        collection_dump_path = resolve_dumped_collection_payloads_path(
            project_root=project_root,
            collection_name=collection_name,
            dump_script_name=dump_script_name,
            dump_timestamp=dump_timestamp,
            output_name="dump",
        )
        with open(collection_dump_path, "r", encoding="utf-8") as collection_dump_file:
            dumped_points = json.load(collection_dump_file)

        source_chunks, source_id_to_chunk = build_oracle_chunks_from_dumped_points(
            dumped_points=dumped_points,
            source_name=data_source,
            collection_name=collection_name,
        )
        collection_names.append(collection_name)
        collection_dump_paths.append(collection_dump_path)
        chunks.extend(source_chunks)
        id_to_chunk.update(source_id_to_chunk)

    oracle_entries = []
    for request_entry in request_entries:
        reranker_query = request_entry["reranker_query"]
        anonymized_request = request_entry["anonymized_request"]
        if not anonymized_request or not chunks:
            continue
        oracle_entries.append({
            "sample": request_entry["sample"],
            "reranker_query": reranker_query,
            "anonymized_request": anonymized_request,
            "chunks": chunks,
            "id_to_chunk": id_to_chunk,
        })

    return oracle_entries, {
        "query_rewrite_cache_path": query_rewrite_cache_path,
        "data_sources": data_sources,
        "collection_names": collection_names,
        "collection_dump_paths": collection_dump_paths,
        "collection_dump_timestamps": [
            collection_dump_path.parent.parent.name
            for collection_dump_path in collection_dump_paths
        ],
        "n_corpus_chunks": len(chunks),
    }

#######################################
# Helper 24: Load retrieval eval data #
#######################################
def load_retrieval_eval_output(
        project_root,
        split_name,
        data_variant,
        source_name,
        output_name,
        timestamp,
        ):
    retrieval_output_path = resolve_data_variant_eval_output_path(
        project_root=project_root,
        split_name=split_name,
        variant=data_variant,
        source_name=source_name,
        output_name=output_name,
        timestamp=timestamp,
    )
    with open(retrieval_output_path, "r", encoding="utf-8") as retrieval_output_file:
        retrieval_output = json.load(retrieval_output_file)

    return retrieval_output, retrieval_output_path

########################################################
# Helper 25: Build oracle chunks from retrieval result #
########################################################
def build_oracle_chunks_from_retrieval_results(
        retrieval_results,
        source_name,
        retrieval_output_name=None,
        top_k=None,
        ):
    if top_k is not None:
        retrieval_results = retrieval_results[:top_k]
    chunks = []
    id_to_chunk = {}
    for retrieval_result_index, retrieval_result in enumerate(retrieval_results):
        chunk_source_name = retrieval_result.get("source", source_name)
        chunk_id_source = (
            f"{chunk_source_name}_{retrieval_output_name}"
            if retrieval_output_name is not None
            else chunk_source_name
        )
        chunk_id = f"{chunk_id_source}_{retrieval_result_index}"
        chunks.append({
            "id": chunk_id,
            "text": get_text_from_payload(
                retrieval_result["payload"]
            ).strip(),
        })
        id_to_chunk[chunk_id] = {
            "id": chunk_id,
            "source_name": chunk_source_name,
            "retrieval_result": retrieval_result,
        }

    return chunks, id_to_chunk

###############################################
# Helper 26: Load retrieval oracle input data #
###############################################
def load_retrieval_oracle_inputs(
        project_root,
        split_name,
        data_variant,
        data_sources,
        retrieval_output_names,
        retrieval_timestamp,
        max_samples,
        top_k_per_retrieval_output,
        ):
    def load_single_retrieval_oracle_inputs(retrieval_output_name):
        if retrieval_output_name == "reranker":
            retrieval_output, retrieval_output_path = load_retrieval_eval_output(
                project_root=project_root,
                split_name=split_name,
                data_variant=data_variant,
                source_name=None,
                output_name=retrieval_output_name,
                timestamp=retrieval_timestamp,
            )

            retrieval_results = retrieval_output["results"]
            if max_samples is not None:
                retrieval_results = retrieval_results[:max_samples]

            oracle_entries = []
            for result_entry in retrieval_results:
                reranker_query = result_entry["reranker_query"]
                if result_entry["retrieval_failed"]:
                    continue
                chunks, id_to_chunk = build_oracle_chunks_from_retrieval_results(
                    retrieval_results=result_entry["retrieval_results"],
                    source_name=retrieval_output_name,
                    top_k=top_k_per_retrieval_output,
                )
                if not reranker_query or not chunks:
                    continue
                oracle_entries.append({
                    "sample": result_entry["sample"],
                    "reranker_query": reranker_query,
                    "chunks": chunks,
                    "id_to_chunk": id_to_chunk,
                })

            return oracle_entries, {
                "data_sources": data_sources,
                "retrieval_output_names": [retrieval_output_name],
                "retrieval_output_paths": [retrieval_output_path],
            }

        source_outputs = []
        retrieval_output_paths = []
        for data_source in data_sources:
            retrieval_output, retrieval_output_path = load_retrieval_eval_output(
                project_root=project_root,
                split_name=split_name,
                data_variant=data_variant,
                source_name=data_source,
                output_name=retrieval_output_name,
                timestamp=retrieval_timestamp,
            )
            source_outputs.append((data_source, retrieval_output))
            retrieval_output_paths.append(retrieval_output_path)

        first_source_results = source_outputs[0][1]["results"]
        if max_samples is not None:
            first_source_results = first_source_results[:max_samples]

        oracle_entries = []
        for result_index, first_result_entry in enumerate(first_source_results):
            reranker_query = first_result_entry["reranker_query"]
            chunks = []
            id_to_chunk = {}
            for source_name, retrieval_output in source_outputs:
                result_entry = retrieval_output["results"][result_index]
                if result_entry["retrieval_failed"]:
                    continue
                source_chunks, source_id_to_chunk = build_oracle_chunks_from_retrieval_results(
                    retrieval_results=result_entry["retrieval_results"],
                    source_name=source_name,
                    top_k=top_k_per_retrieval_output,
                )
                chunks.extend(source_chunks)
                id_to_chunk.update(source_id_to_chunk)

            if not reranker_query or not chunks:
                continue
            oracle_entries.append({
                "sample": first_result_entry["sample"],
                "reranker_query": reranker_query,
                "chunks": chunks,
                "id_to_chunk": id_to_chunk,
            })

        return oracle_entries, {
            "data_sources": data_sources,
            "retrieval_output_names": [retrieval_output_name],
            "retrieval_output_paths": retrieval_output_paths,
        }

    if len(retrieval_output_names) == 1:
        return load_single_retrieval_oracle_inputs(retrieval_output_names[0])

    retrieval_outputs = []
    retrieval_output_paths = []
    for retrieval_output_name in retrieval_output_names:
        if retrieval_output_name == "reranker":
            retrieval_output, retrieval_output_path = load_retrieval_eval_output(
                project_root=project_root,
                split_name=split_name,
                data_variant=data_variant,
                source_name=None,
                output_name=retrieval_output_name,
                timestamp=retrieval_timestamp,
            )
            retrieval_outputs.append((retrieval_output_name, None, retrieval_output))
            retrieval_output_paths.append(retrieval_output_path)
            continue
        for data_source in data_sources:
            retrieval_output, retrieval_output_path = load_retrieval_eval_output(
                project_root=project_root,
                split_name=split_name,
                data_variant=data_variant,
                source_name=data_source,
                output_name=retrieval_output_name,
                timestamp=retrieval_timestamp,
            )
            retrieval_outputs.append((retrieval_output_name, data_source, retrieval_output))
            retrieval_output_paths.append(retrieval_output_path)

    first_retrieval_results = retrieval_outputs[0][2]["results"]
    if max_samples is not None:
        first_retrieval_results = first_retrieval_results[:max_samples]

    oracle_entries = []
    for result_index, first_result_entry in enumerate(first_retrieval_results):
        reranker_query = first_result_entry["reranker_query"]
        chunks = []
        id_to_chunk = {}
        chunk_text_to_chunk_id = {}
        for retrieval_output_name, data_source, retrieval_output in retrieval_outputs:
            result_entry = retrieval_output["results"][result_index]
            if result_entry["retrieval_failed"]:
                continue
            source_name = data_source or retrieval_output_name
            source_chunks, source_id_to_chunk = build_oracle_chunks_from_retrieval_results(
                retrieval_results=result_entry["retrieval_results"],
                source_name=source_name,
                retrieval_output_name=retrieval_output_name,
                top_k=top_k_per_retrieval_output,
            )
            for source_chunk in source_chunks:
                chunk_text = source_chunk["text"]
                origin_chunk = source_id_to_chunk[source_chunk["id"]]
                if chunk_text in chunk_text_to_chunk_id:
                    original_chunk_id = chunk_text_to_chunk_id[chunk_text]
                    id_to_chunk[original_chunk_id]["retrieval_origins"].append(origin_chunk)
                    continue
                chunk_text_to_chunk_id[chunk_text] = source_chunk["id"]
                chunks.append(source_chunk)
                id_to_chunk[source_chunk["id"]] = {
                    **origin_chunk,
                    "retrieval_origins": [origin_chunk],
                }

        if not reranker_query or not chunks:
            continue
        oracle_entries.append({
            "sample": first_result_entry["sample"],
            "reranker_query": reranker_query,
            "chunks": chunks,
            "id_to_chunk": id_to_chunk,
        })

    return oracle_entries, {
        "data_sources": data_sources,
        "retrieval_output_names": retrieval_output_names,
        "retrieval_output_paths": retrieval_output_paths,
    }

#####################################
# Helper 27: Load oracle input data #
#####################################
def load_oracle_inputs(
        input_mode,
        project_root,
        query_rewrite_cache_dir,
        split_name,
        context_emails_mode,
        n_eval_samples_per_folder_uri,
        configured_cache_filename,
        max_samples,
        data_variant,
        dump_script_name,
        dump_timestamp,
        data_sources,
        retrieval_output_names,
        retrieval_timestamp,
        top_k_per_retrieval_output,
        ):
    if input_mode == "corpus":
        return load_corpus_oracle_inputs(
            project_root=project_root,
            query_rewrite_cache_dir=query_rewrite_cache_dir,
            split_name=split_name,
            context_emails_mode=context_emails_mode,
            n_eval_samples_per_folder_uri=n_eval_samples_per_folder_uri,
            configured_cache_filename=configured_cache_filename,
            max_samples=max_samples,
            data_variant=data_variant,
            data_sources=data_sources,
            dump_script_name=dump_script_name,
            dump_timestamp=dump_timestamp,
        )
    if input_mode == "retrieval":
        return load_retrieval_oracle_inputs(
            project_root=project_root,
            split_name=split_name,
            data_variant=data_variant,
            data_sources=data_sources,
            retrieval_output_names=retrieval_output_names,
            retrieval_timestamp=retrieval_timestamp,
            max_samples=max_samples,
            top_k_per_retrieval_output=top_k_per_retrieval_output,
        )
    raise KeyError(input_mode)

##################################################################
# Helper 28: Attach selected chunks to discriminator result data #
##################################################################
def attach_selected_chunks_to_discriminator_result(discriminator_result, id_to_chunk):
    subqueries = discriminator_result.get("subqueries") or []
    for subquery in subqueries:
        supporting_chunk_ids = subquery.get("supporting_chunk_ids") or []
        insufficient_chunk_ids = subquery.get("insufficient_chunk_ids") or []
        subquery["supporting_chunks"] = [
            id_to_chunk.get(str(chunk_id), {"id": str(chunk_id), "payload": None})
            for chunk_id in supporting_chunk_ids
        ]
        subquery["insufficient_chunks"] = [
            id_to_chunk.get(str(chunk_id), {"id": str(chunk_id), "payload": None})
            for chunk_id in insufficient_chunk_ids
        ]
