import json
from pathlib import Path

from config.eval import RESULTS_DIR_NAME

def serialize_payload(payload):
    return json.dumps(payload, sort_keys=True, ensure_ascii=False)

def load_json(json_path):
    with open(json_path, "r", encoding="utf-8") as json_file:
        return json.load(json_file)

def format_path(path, project_root):
    try:
        return Path(path).relative_to(project_root)
    except ValueError:
        return path

def resolve_path(path, project_root):
    resolved_path = Path(path)
    if resolved_path.is_absolute():
        return resolved_path
    return Path(project_root) / resolved_path

def resolve_existing_path(path, project_root, path_name, script_name):
    resolved_path = resolve_path(path, project_root)
    if not resolved_path.exists():
        raise ValueError(
            f"{script_name}: {path_name} file does not exist:\n"
            f"\t{resolved_path}"
        )
    return resolved_path

def get_oracle_results_root(project_root, split_name):
    return (
        Path(project_root)
        / "eval"
        / RESULTS_DIR_NAME
        / "run_oracle_discriminator"
        / split_name
    )

def resolve_oracle_path_from_timestamp(
        project_root,
        split_name,
        variant,
        timestamp,
        script_name,
        ):
    oracle_path = (
        get_oracle_results_root(project_root, split_name)
        / timestamp
        / variant
        / "oracle_discriminator.json"
    )
    if not oracle_path.exists():
        raise ValueError(
            f"{script_name}: oracle file does not exist:\n"
            f"\t{oracle_path}"
        )
    return oracle_path

def get_reranker_retrieval_path(oracle_data, project_root, script_name):
    metadata = oracle_data.get("oracle_input_metadata") or {}
    retrieval_output_paths = metadata.get("retrieval_output_paths") or []
    for retrieval_output_path in retrieval_output_paths:
        if Path(str(retrieval_output_path)).name != "reranker.json":
            continue
        reranker_path = resolve_path(retrieval_output_path, project_root)
        if reranker_path.exists():
            return reranker_path
        raise ValueError(
            f"{script_name}: reranker file does not exist:\n"
            f"\t{reranker_path}"
        )
    raise ValueError(
        f"{script_name}: oracle file does not reference a reranker.json "
        "retrieval output"
    )

def get_answerability_label(oracle_result, answerability_order):
    if oracle_result.get("generation_failed"):
        return None
    discriminator_result = oracle_result.get("discriminator_result") or {}
    ans_label = str(discriminator_result.get("answerability"))
    if ans_label in answerability_order:
        return ans_label
    return None

def get_rank_map(retrieval_result):
    return (
        retrieval_result.get("ranked_list_name_to_rank")
        or retrieval_result.get("ranked_list_ranks")
        or {}
    )

def get_encoder_from_ranked_list_name(ranked_list_name, encoder_order, encoder_aliases=None):
    encoder_aliases = encoder_aliases or {}
    encoder = ranked_list_name.split("::", 1)[0]
    encoder = encoder_aliases.get(encoder, encoder)
    if encoder in encoder_order:
        return encoder
    return None

def get_encoder_from_origin_id(origin_id, encoder_order, data_sources, encoder_aliases=None):
    encoder_aliases = encoder_aliases or {}
    for source_name in data_sources:
        source_prefix = f"{source_name}_"
        if not origin_id.startswith(source_prefix):
            continue
        origin_encoder = origin_id[len(source_prefix):].rsplit("_", 1)[0]
        origin_encoder = encoder_aliases.get(origin_encoder, origin_encoder)
        if origin_encoder in encoder_order:
            return origin_encoder
    return None

def get_supporting_chunk_encoders(
        supporting_chunk,
        encoder_order,
        data_sources,
        encoder_aliases=None,
        ):
    encoders = {
        encoder
        for encoder in [
            get_encoder_from_origin_id(
                origin_id=str(origin.get("id", "")),
                encoder_order=encoder_order,
                data_sources=data_sources,
                encoder_aliases=encoder_aliases,
            )
            for origin in supporting_chunk.get("retrieval_origins") or []
        ]
        if encoder is not None
    }
    if encoders:
        return encoders

    retrieval_result = supporting_chunk.get("retrieval_result") or {}
    encoders = {
        encoder
        for encoder in [
            get_encoder_from_ranked_list_name(
                ranked_list_name=ranked_list_name,
                encoder_order=encoder_order,
                encoder_aliases=encoder_aliases,
            )
            for ranked_list_name in get_rank_map(retrieval_result)
        ]
        if encoder is not None
    }
    if encoders:
        return encoders

    encoder = get_encoder_from_origin_id(
        origin_id=str(supporting_chunk.get("id", "")),
        encoder_order=encoder_order,
        data_sources=data_sources,
        encoder_aliases=encoder_aliases,
    )
    return {encoder} if encoder is not None else set()

def get_supporting_chunk_encoder_ranks(
        supporting_chunk,
        encoder_order,
        data_sources,
        encoder_aliases=None,
        ):
    encoder_to_ranks = {
        encoder: []
        for encoder in encoder_order
    }

    for origin in supporting_chunk.get("retrieval_origins") or []:
        encoder = get_encoder_from_origin_id(
            origin_id=str(origin.get("id", "")),
            encoder_order=encoder_order,
            data_sources=data_sources,
            encoder_aliases=encoder_aliases,
        )
        rank = (origin.get("retrieval_result") or {}).get("rank")
        if encoder is not None and rank is not None:
            encoder_to_ranks[encoder].append(int(rank))

    if any(encoder_to_ranks.values()):
        return encoder_to_ranks

    retrieval_result = supporting_chunk.get("retrieval_result") or {}
    for ranked_list_name, rank in get_rank_map(retrieval_result).items():
        encoder = get_encoder_from_ranked_list_name(
            ranked_list_name=ranked_list_name,
            encoder_order=encoder_order,
            encoder_aliases=encoder_aliases,
        )
        if encoder is not None:
            encoder_to_ranks[encoder].append(int(rank))

    if any(encoder_to_ranks.values()):
        return encoder_to_ranks

    encoder = get_encoder_from_origin_id(
        origin_id=str(supporting_chunk.get("id", "")),
        encoder_order=encoder_order,
        data_sources=data_sources,
        encoder_aliases=encoder_aliases,
    )
    rank = retrieval_result.get("rank")
    if encoder is not None and rank is not None:
        encoder_to_ranks[encoder].append(int(rank))
    return encoder_to_ranks

def get_supporting_chunk_data(
        oracle_result,
        encoder_order,
        data_sources,
        encoder_aliases=None,
        include_payload=False,
        ):
    discriminator_result = oracle_result.get("discriminator_result") or {}
    supporting_chunk_data = []
    seen_supporting_chunk_ids = set()
    for subquery in discriminator_result.get("subqueries") or []:
        for supporting_chunk in subquery.get("supporting_chunks") or []:
            chunk_id = supporting_chunk.get("id")
            if chunk_id in seen_supporting_chunk_ids:
                continue
            seen_supporting_chunk_ids.add(chunk_id)

            current_supporting_chunk_data = {
                "encoders": get_supporting_chunk_encoders(
                    supporting_chunk=supporting_chunk,
                    encoder_order=encoder_order,
                    data_sources=data_sources,
                    encoder_aliases=encoder_aliases,
                ),
                "encoder_to_ranks": get_supporting_chunk_encoder_ranks(
                    supporting_chunk=supporting_chunk,
                    encoder_order=encoder_order,
                    data_sources=data_sources,
                    encoder_aliases=encoder_aliases,
                ),
            }
            if include_payload:
                payload = (
                    supporting_chunk.get("retrieval_result") or {}
                ).get("payload")
                current_supporting_chunk_data["payload"] = (
                    serialize_payload(payload)
                    if payload is not None
                    else None
                )
            supporting_chunk_data.append(current_supporting_chunk_data)
    return supporting_chunk_data
