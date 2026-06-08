# RAG-based Email Autoresponder

This repository contains a RAG-based email autoresponder for the MUIA master's program. The system is organized around Modal applications that crawl and encode knowledge-base collections, retrieve and rerank context for incoming email threads, run local decoder/encoder models, manage Qdrant collections, and support evaluation.

## Modal Applications

The tables below summarize the remote Modal applications and functions used by the system.

### `crawler-agent`

| Function | Schedule | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --- | --- | --- | ---: | ---: | --- |
| `run_crawler_agent` | `0 9 10 9 *` | CPU | 86400 | 60 | Yes |

### `email-agent`

| Function | Schedule | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --- | --- | --- | ---: | ---: | --- |
| `run_email_agent` | `0 9 * * *` | CPU | 600 | 60 | No |

### `decoder-legacy`

| Function | Schedule | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --- | --- | --- | ---: | ---: | --- |
| `run_local_lm_or_vlm_legacy` | On demand | L40S GPU | 900 | 60 | No |

### `decoder-latest`

| Function | Schedule | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --- | --- | --- | ---: | ---: | --- |
| `run_local_lm_or_vlm_latest` | On demand | H100 GPU | 1800 | 60 | No |

### `encoder-cpu`

| Function | Schedule | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --- | --- | --- | ---: | ---: | --- |
| `run_encoder_cpu_batch_document_embedder` | On demand | CPU | 3600 | 60 | Yes |
| `run_encoder_cpu_batch_query_embedder_and_qdrant_retriever` | On demand | CPU | 3600 | 60 | Yes |

### `encoder-gpu`

| Function | Schedule | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --- | --- | --- | ---: | ---: | --- |
| `run_encoder_gpu_batch_document_embedder` | On demand | L40S GPU | 1800 | 60 | Yes |
| `run_encoder_gpu_batch_query_embedder_and_qdrant_retriever` | On demand | L40S GPU | 1800 | 60 | Yes |
| `run_encoder_gpu_reranker` | On demand | L40S GPU | 1800 | 60 | Yes |

### `qdrant-server`

| Function | Schedule | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --- | --- | --- | ---: | ---: | --- |
| `serve_qdrant_server` | On demand | CPU | 3600 | 900 | Yes |

### `storage-handler`

| Function | Schedule | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --- | --- | --- | ---: | ---: | --- |
| `write_chunk_records` | On demand | CPU | 3600 | 60 | Yes |
| `read_jsonl_records` | On demand | CPU | 3600 | 60 | Yes |

### `volume-handler`

These functions are not deployed by default; volume removal is only available from local execution.

| Function | Schedule | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --- | --- | --- | ---: | ---: | --- |
| `delete_volume_folders` | On demand | CPU | 3600 | 60 | Yes |
| `count_lm_output_tokens` | On demand | CPU | 3600 | 60 | Yes |

### `collection-handler`

| Function | Schedule | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --- | --- | --- | ---: | ---: | --- |
| `drop_legacy_collections` | On demand | CPU | 3600 | 60 | No |
| `create_collections` | On demand | CPU | 3600 | 60 | No |
| `enable_collection_optimizations` | On demand | CPU | 3600 | 60 | No |
| `write_batch_points` | On demand | CPU | 3600 | 60 | No |
| `dump_collection_payloads` | On demand | CPU | 3600 | 60 | No |

### `decoder-latest-tokenizer`

| Function | Schedule | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --- | --- | --- | ---: | ---: | --- |
| `count_decoder_latest_tokens` | On demand | CPU | 1800 | 60 | No |

### `curator`

| Function | Schedule | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --- | --- | --- | ---: | ---: | --- |
| `run_email_knowledge_base_curator_pipeline` | On demand | CPU | 28800 | 60 | Yes |

### `llm-judge`

| Function | Schedule | Hardware | Timeout (s) | Scaledown (s) | Volume |
| --- | --- | --- | ---: | ---: | --- |
| `run_llm_judge` | On demand | CPU | 900 | 60 | No |

## Fine-Tuning

BGE-M3 fine-tuning instructions are documented in [`finetune/README.md`](finetune/README.md).
