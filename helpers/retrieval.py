##########################################################
# Helper 1: Fuse multiple query results (for one sample) #
##########################################################
def fuse_multiple_query_results_for_one_sample(sample_query_entries_with_top_k_chunks, top_k):
    import json

    # given we do query rewritting, having multiple queries per sample (email),
    # we get the top k results across all queries (which are 'rewrites' of
    # the original sample or email, hopefully optimized for retrieval)

    # sample_query_entries_with_top_k_chunks:
    # several query entries for one sample
    # and for each query entry, its top-k retrieved chunks
    # (with each chunk having its own payload)

    # multiple queries could return the same chunk,
    # so for a unique top k we index by payload
    payload_to_highest_scored_query_candidate = {}
    for sample_query_entry in sample_query_entries_with_top_k_chunks:
        rewritten_query = sample_query_entry["query"]
        query_type = sample_query_entry["query_type"]
        query_top_k_chunks = sample_query_entry["top_k_chunks"]
        for retrieved_chunk in query_top_k_chunks:
            payload_key = json.dumps(retrieved_chunk["payload"], sort_keys=True, ensure_ascii=False)
            score = retrieved_chunk["score"]
            best_query_candidate = payload_to_highest_scored_query_candidate.get(payload_key)
            if best_query_candidate is None or score > best_query_candidate["score"]:
                best_query_candidate = retrieved_chunk.copy()
                best_query_candidate["matched_query"] = {
                    "query": rewritten_query,
                    "query_type": query_type,
                }
                payload_to_highest_scored_query_candidate[payload_key] = best_query_candidate
    sorted_candidates = sorted(
        payload_to_highest_scored_query_candidate.values(),
        key=lambda retrieved_chunk: retrieved_chunk["score"],
        reverse=True
    )
    return sorted_candidates[:top_k]

#####################################################################
# Helper 2: Fuse multiple encoder results (for one sample) with RRF #
#####################################################################
def fuse_multiple_encoder_results_with_rrf(encoder_to_top_k_chunks, top_k, rank_constant=60):
    import json

    # even after fusing results from multiple queries, multiple encoders
    # will probably disagree on the top k chunks; a way to choose the top k also across
    # encoders, even with different scores, is RRF (Reciprocal Rank Fusion)

    payload_to_rrf_candidate = {}
    for encoder_name, encoder_top_k_chunks in encoder_to_top_k_chunks.items():
        for retrieved_chunk in encoder_top_k_chunks:
            payload = retrieved_chunk["payload"]
            payload_key = json.dumps(payload, sort_keys=True, ensure_ascii=False)
            if payload_key not in payload_to_rrf_candidate:
                payload_to_rrf_candidate[payload_key] = {
                    "score": 0.0,
                    "matched_queries": {},
                    "encoder_scores": {},
                    "encoder_ranks": {},
                    "payload": payload,
                }
            rrf_chunk_for_payload = payload_to_rrf_candidate[payload_key]
            rrf_chunk_for_payload["score"] += 1.0 / (rank_constant + retrieved_chunk["rank"])
            rrf_chunk_for_payload["matched_queries"][encoder_name] = retrieved_chunk["matched_query"]
            rrf_chunk_for_payload["encoder_scores"][encoder_name] = retrieved_chunk["score"]
            rrf_chunk_for_payload["encoder_ranks"][encoder_name] = retrieved_chunk["rank"]

    sorted_candidates = sorted(
        payload_to_rrf_candidate.values(),
        key=lambda retrieved_chunk: retrieved_chunk["score"],
        reverse=True,
    )
    if top_k is None:
        return sorted_candidates
    return sorted_candidates[:top_k]

################################################################
# Helper 3: Keep category minimums from an already ranked list #
################################################################
def keep_category_minimums_from_ranked_chunks(ranked_chunks, top_k, category_to_min_final_count):
    # ranked chunks are assummed to be **already** deduplicated
    # (e.g., from fuse_multiple_encoder_results_with_rrf)
    plain_top_k_original_chunks = ranked_chunks[:top_k]

    if not category_to_min_final_count:
        return [
            {
                **ranked_chunk,
                "selection_reason": "top_k",
            }
            for ranked_chunk in ranked_chunks[:top_k]
        ]

    requested_min_count = sum(category_to_min_final_count.values())
    if requested_min_count > top_k:
        raise ValueError(
            "keep_category_minimums_from_ranked_chunks: requested minimum category count "
            f"({requested_min_count}) exceeds top_k ({top_k})"
        )

    selected_original_chunks = []
    selected_chunks = []
    for category, min_count in category_to_min_final_count.items():
        if min_count <= 0:
            continue
        category_chunks = [
            ranked_chunk
            for ranked_chunk in ranked_chunks
            if ranked_chunk["payload"].get("category") == category
        ][:min_count]
        selected_original_chunks.extend(category_chunks)
        selected_chunks.extend(
            {
                **ranked_chunk,
                "selection_reason": (
                    "category_minimum_and_top_k"
                    if ranked_chunk in plain_top_k_original_chunks
                    else "category_minimum_only"
                ),
            }
            for ranked_chunk in category_chunks
        )

    for ranked_chunk in ranked_chunks:
        if len(selected_chunks) == top_k:
            break
        if ranked_chunk in selected_original_chunks:
            continue
        selected_chunks.append({
            **ranked_chunk,
            "selection_reason": "top_k",
        })

    return selected_chunks
