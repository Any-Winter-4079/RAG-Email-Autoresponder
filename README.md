# RAG-based Email Autoresponder

This repository contains a RAG-based email autoresponder for the MUIA master's program. The system is organized around Modal applications that crawl and encode knowledge-base collections, retrieve and rerank context for incoming email threads, run local decoder/encoder models, manage Qdrant collections, and support evaluation.

## Overview

<img width="5097" height="4729" alt="muia_prod_pipeline" src="https://github.com/user-attachments/assets/e01e88dc-2b1e-4f5a-afdd-2b82038f7f48" />

## Fine-tuning

M3 is fine-tuned with its default InfoNCE-style loss and in-batch negatives (IBN):

```math
\mathcal{L}_{*,\mathrm{IBN}}
=
-\frac{1}{|\mathcal{B}_Q|}
\sum_{i=1}^{|\mathcal{B}_Q|}
\log
\frac{
\exp\left(s_*\left(q_i,p_i^{+}\right)/\tau\right)
}{
Z_i
}
```

To reduce the number of false negatives in softmax's denominator (due to topic collision and data augmentation), three increasigly more aggresive masking strategies are used.

### Exact positive-passage mask

Exact token-matching instances of the positive passage for query $q_i$ are masked from the in-batch positives and negatives:

```math
\begin{aligned}
Z_i ={}&
\exp\left(s_*\left(q_i,p_i^{+}\right)/\tau\right)
\\
&+
\sum_{p\in\mathcal{P}_i^{-}}
\exp\left(s_*\left(q_i,p\right)/\tau\right)
\\
&+
\sum_{\substack{j=1\\j\neq i}}^{|\mathcal{B}_Q|}
m^{\mathrm{pos}}_{i,p_j^{+}}
\exp\left(s_*\left(q_i,p_j^{+}\right)/\tau\right)
\\
&+
\sum_{\substack{j=1\\j\neq i}}^{|\mathcal{B}_Q|}
\sum_{p\in\mathcal{P}_j^{-}}
m^{\mathrm{pos}}_{i,p}
\exp\left(s_*\left(q_i,p\right)/\tau\right)
\end{aligned}
```

where the collision mask $m^{\mathrm{pos}}_{i,p}$ zeroes out any additional IBN term whose passage tokens exactly match those of the positive passage for $q_i$:

```math
m^{\mathrm{pos}}_{i,p}=
\begin{cases}
0 & \text{if } \mathrm{tok}(p)=\mathrm{tok}(p_i^{+}),\\
1 & \text{otherwise.}
\end{cases}
```

### Same-group positive-passage mask

Any passage matching a positive passage associated with the same query-expansion group as $q_i$ is masked:

```math
\begin{aligned}
Z_i ={}&
\exp\left(s_*\left(q_i,p_i^{+}\right)/\tau\right)
\\
&+
\sum_{p\in\mathcal{P}_i^{-}}
\exp\left(s_*\left(q_i,p\right)/\tau\right)
\\
&+
\sum_{\substack{j=1\\j\neq i}}^{|\mathcal{B}_Q|}
m^{\mathrm{group}}_{i,p_j^{+}}
\exp\left(s_*\left(q_i,p_j^{+}\right)/\tau\right)
\\
&+
\sum_{\substack{j=1\\j\neq i}}^{|\mathcal{B}_Q|}
\sum_{p\in\mathcal{P}_j^{-}}
m^{\mathrm{group}}_{i,p}
\exp\left(s_*\left(q_i,p\right)/\tau\right)
\end{aligned}
```

where the mask $m^{\mathrm{group}}_{i,p}$ zeroes out a passage whose tokens match a positive passage for any query in the same expansion group as $q_i$:

```math
m^{\mathrm{group}}_{i,p}=
\begin{cases}
0 & \text{if } \exists k\in\{1,\dots,|\mathcal{B}_Q|\}: g(q_k)=g(q_i) \text{ and } \mathrm{tok}(p)=\mathrm{tok}(p_k^{+}),\\
1 & \text{otherwise.}
\end{cases}
```

### Similar-group positive-passage mask

Additionally, positive passages associated with queries considered similar to $q_i$ by a cross-encoder are also masked:

```math
\begin{aligned}
Z_i ={}&
\exp\left(s_*\left(q_i,p_i^{+}\right)/\tau\right)
\\
&+
\sum_{p\in\mathcal{P}_i^{-}}
\exp\left(s_*\left(q_i,p\right)/\tau\right)
\\
&+
\sum_{\substack{j=1\\j\neq i}}^{|\mathcal{B}_Q|}
m^{\mathrm{sim\_group}}_{i,p_j^{+}}
\exp\left(s_*\left(q_i,p_j^{+}\right)/\tau\right)
\\
&+
\sum_{\substack{j=1\\j\neq i}}^{|\mathcal{B}_Q|}
\sum_{p\in\mathcal{P}_j^{-}}
m^{\mathrm{sim\_group}}_{i,p}
\exp\left(s_*\left(q_i,p\right)/\tau\right)
\end{aligned}
```

where the mask $m^{\mathrm{sim\_group}}_{i,p}$ zeroes out passages matching a positive from the same expansion group or from a query whose cross-encoder similarity score with $q_i$ is at least $0$:

```math
m^{\mathrm{sim\_group}}_{i,p}=
\begin{cases}
0 & \begin{aligned}
    &\text{if } \exists k\in\{1,\dots,|\mathcal{B}_Q|\}:\\
    &\left(g(q_k)=g(q_i) \text{ or } \mathrm{score}_{\mathrm{cross\text{-}encoder}}(q_i,q_k)\geq 0\right)\\
    &\text{and } \mathrm{tok}(p)=\mathrm{tok}(p_k^{+}),
\end{aligned}\\
1 & \text{otherwise.}
\end{cases}
```

## Modal Applications

The tables below summarize the remote Modal applications and functions used by the system.

### `crawler-agent`

| Function            | Schedule     | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ------------------- | ------------ | -------- | ----------: | ------------: | ------ |
| `run_crawler_agent` | `0 9 10 9 *` | CPU      |       86400 |       Default | Yes    |

### `email-agent`

| Function          | Schedule                      | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ----------------- | ----------------------------- | -------- | ----------: | ------------: | ------ |
| `run_email_agent` | `0 9 * * *` (`Europe/Madrid`) | CPU      |        5400 |       Default | No     |

### `decoder-legacy`

| Function                     | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ---------------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `run_local_lm_or_vlm_legacy` | On demand | L40S GPU |         900 |           180 | No     |

### `decoder-latest`

| Function                     | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ---------------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `run_local_lm_or_vlm_latest` | On demand | H100 GPU |        1800 |           180 | No     |

### `encoder-cpu`

| Function                                                    | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ----------------------------------------------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `run_encoder_cpu_batch_document_embedder`                   | On demand | CPU      |        3600 |            60 | Yes    |
| `run_encoder_cpu_batch_query_embedder_and_qdrant_retriever` | On demand | CPU      |        3600 |            60 | Yes    |

### `encoder-gpu`

| Function                                                    | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ----------------------------------------------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `run_encoder_gpu_batch_document_embedder`                   | On demand | L40S GPU |        1800 |            60 | Yes    |
| `run_encoder_gpu_batch_query_embedder_and_qdrant_retriever` | On demand | L40S GPU |        1800 |            60 | Yes    |
| `run_encoder_gpu_reranker`                                  | On demand | L40S GPU |        1800 |            60 | Yes    |

### `qdrant-server`

| Function              | Schedule     | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --------------------- | ------------ | -------- | ----------: | ------------: | ------ |
| `serve_qdrant_server` | Web endpoint | CPU      |        3600 |           900 | Yes    |

The Qdrant endpoint runs with one non-preemptible container (`min_containers=1`, `max_containers=1`) and accepts up to 100 concurrent inputs.

### `storage-handler`

| Function              | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `write_chunk_records` | On demand | CPU      |        3600 |            60 | Yes    |
| `read_jsonl_records`  | On demand | CPU      |        3600 |            60 | Yes    |

### `volume-handler`

These functions are not deployed by default; volume removal is only available from local execution.

| Function                 | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ------------------------ | --------- | -------- | ----------: | ------------: | ------ |
| `delete_volume_folders`  | On demand | CPU      |        3600 |            60 | Yes    |
| `count_lm_output_tokens` | On demand | CPU      |        3600 |            60 | Yes    |

### `collection-handler`

| Function                          | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --------------------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `drop_legacy_collections`         | On demand | CPU      |        3600 |            60 | No     |
| `create_collections`              | On demand | CPU      |        3600 |            60 | No     |
| `enable_collection_optimizations` | On demand | CPU      |        3600 |            60 | No     |
| `write_batch_points`              | On demand | CPU      |        3600 |            60 | No     |
| `dump_collection_payloads`        | On demand | CPU      |        3600 |            60 | No     |

### `decoder-latest-tokenizer`

| Function                        | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ------------------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `count_decoder_latest_tokens`   | On demand | CPU      |        1800 |            60 | No     |
| `truncate_decoder_latest_texts` | On demand | CPU      |        1800 |            60 | No     |

### `curator`

| Function                                    | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| ------------------------------------------- | --------- | -------- | ----------: | ------------: | ------ |
| `run_email_knowledge_base_curator_pipeline` | On demand | CPU      |       28800 |            60 | Yes    |

### `llm-judge`

| Function        | Schedule  | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --------------- | --------- | -------- | ----------: | ------------: | ------ |
| `run_llm_judge` | On demand | CPU      |         900 |            60 | No     |

## Fine-Tuning

BGE-M3 fine-tuning instructions are documented in [`finetune/README.md`](finetune/README.md).
