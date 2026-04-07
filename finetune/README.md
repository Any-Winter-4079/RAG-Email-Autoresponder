# M3 Fine-Tuning on RunPod

## Overview

This folder documents fine-tuning `BAAI/bge-m3` with `FlagEmbedding`.

- Upstream reference shell script: `FlagEmbedding/examples/finetune/embedder/encoder_only/m3.sh`
- Pinned `FlagEmbedding` commit: `dbc600560b2dadcc1514989092f7b849673bb67d`
- No knowledge distillation
- No query or passage instruction prefixes
- `cls` pooling for the dense embedding path
- `unified_finetuning=True`

## Local Data Preparation

### 1. Create and activate a local macOS virtual environment

Run:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install dotenv transformers torch torchvision matplotlib modal cycler
```

These local packages are needed for the data-preparation scripts.

When you open a new terminal, reactivate the environment with:

```bash
source .venv/bin/activate
```

### 2. Set fine-tuning state

In `config/m3.py`, set: `IS_FINETUNED = False`

This keeps things in the pre-fine-tune phase:

- `config/eval.py` uses the `train` split
- RRF uses `bge_m3`
- Crawler agent uses `bge_m3`

### 3. Deploy the Modal services

Run:

```bash
./deploy.sh
```

### 4. Prepare the cleaned email dataset

Run:

```bash
python data/prepare_dataset.py
```

This writes:

- `data/email_messages/messages_with_threads.csv`
- `data/knowledge_base_messages/messages_for_knowledge_base.jsonl`

### 5. Split the dataset

Run:

```bash
python data/split_dataset.py
```

This writes:

- `data/split_datasets/train.json`
- `data/split_datasets/dev.json`
- `data/split_datasets/test.json`

### 6. Curate train-split grouped email threads for the knowledge base

Run:

```bash
python data/curate_email_knowledge_base.py
```

This writes:

- `data/knowledge_base_messages/email_thread_candidates.json`
- `data/knowledge_base_messages/email_lm_abstract_chunks.jsonl`
- `data/knowledge_base_messages/email_lm_summary_chunks.jsonl`
- `data/knowledge_base_messages/email_lm_cleaned_text_chunks.jsonl`
- `data/knowledge_base_messages/email_lm_q_and_a_chunks.jsonl`

### 7. Encode the curated email knowledge base

Run:

```bash
python data/encode_email_knowledge_base.py
```

This encodes the curated email knowledge-base artifacts into their own Qdrant collections:

- `email_lm_summary_chunks`
- `email_lm_cleaned_text_chunks`
- `email_lm_q_and_a_chunks`

### 8. Run the crawler agent so the collections are populated and encoded

Run:

```bash
modal run services/crawler_agent.py::run_crawler_agent
```

This step is needed before dumping collection payloads locally for the oracle discriminator workflow.

### 9. Dump the collection payloads used by the oracle discriminator

Run:

```bash
modal run eval/run_dump_collection_payloads.py --collection-names '["lm_summary_chunks"]'
```

This writes dumped collection payloads under:

- `eval/results/run_dump_collection_payloads/<timestamp>/lm_summary_chunks/dump.json`

### 10. Generate the query rewrite cache

Run:

```bash
modal run eval/run_data_variant_eval.py
```

This step is required because `eval/run_oracle_discriminator.py` reads the reranker queries from the query rewrite cache written by `run_data_variant_eval.py`, and because `data/prepare_m3_finetune_dataset.py` later reads the retrieval outputs written here (currently the `rrf` outputs) to build the fine-tuning queries and mine encoder-side negatives.

This writes query rewriting cache files (to obtain diverse and anonymized queries) under:

- `eval/cache/query_rewrites/`

And writes retrieval outputs (to mine hard-negatives from the encoders) under:

- `eval/results/run_data_variant_eval/<split>/<timestamp>/<data_variant>/`

### 11. Run the oracle discriminator

Run:

```bash
modal run eval/run_oracle_discriminator.py
```

This writes results (to mine positives from the oracle's `supporting_chunks` and negatives from the oracle's `insufficient_chunks`) under:

- `eval/results/run_oracle_discriminator/<split>/<timestamp>/lm_summary_chunks/oracle_discriminator.json`

### 12. Build the final M3 training files from the oracle and retrieval results

Run (with `M3_FINETUNE_ORACLE_DISCRIMINATOR_SPLIT = "train"` in `config/data.py`):

```bash
python data/prepare_m3_finetune_dataset.py
```

This writes:

- `data/finetune/<TIMESTAMP>/train.jsonl`

Training examples are stored in this format:

```json
{"query":"...", "pos":["positive passage 1", "positive passage 2"], "neg":["negative passage 1", "negative passage 2"]}
{"query":"...", "pos":["another positive passage 1", "another positive passage 2"], "neg":["another negative passage 1", "another negative passage 2"]}
```

### 13. Update `finetune/sft.sh` with the maximum query and passage sequence lengths

Run:

```bash
python finetune/get_max_sequence_lengths.py
```

This updates:

- `finetune/sft.sh`

using the latest:

- `data/finetune/<TIMESTAMP>/train.jsonl`

## Docker Image

On a local terminal, run:

```bash
mkdir -p docker_build && \
cat > docker_build/Dockerfile <<'EOF'
FROM anywinter4079/pytorch:2.10.0-cu128

WORKDIR /workspace

RUN apt-get update && apt-get install -y git nano && rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/FlagOpen/FlagEmbedding.git /workspace/FlagEmbedding \
 && cd /workspace/FlagEmbedding \
 && git checkout dbc600560b2dadcc1514989092f7b849673bb67d \
 && pip install -e . \
 && pip install "transformers==4.44.2" "peft==0.17.1" \
 && mkdir -p /workspace/finetune
EOF
```

Or, if you want `flash-attn`, run:

```bash
mkdir -p docker_build && \
cat > docker_build/Dockerfile <<'EOF'
FROM anywinter4079/pytorch:2.10.0-cu128

WORKDIR /workspace

RUN apt-get update && apt-get install -y git nano && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip \
 && pip install flash-attn==2.7.4.post1 --no-build-isolation

RUN git clone https://github.com/FlagOpen/FlagEmbedding.git /workspace/FlagEmbedding \
 && cd /workspace/FlagEmbedding \
 && git checkout dbc600560b2dadcc1514989092f7b849673bb67d \
 && pip install -e . \
 && pip install "transformers==4.44.2" "peft==0.17.1" \
 && mkdir -p /workspace/finetune
EOF
```

Then run:

```bash
docker login
```

If the builder already exists, run:

```bash
docker buildx use amd64-builder
docker buildx build --platform linux/amd64 -t anywinter4079/m3-finetune:pytorch-2.10.0-cu128-flagembedding-dbc6005 ./docker_build --push
```

If it does not exist yet, run:

```bash
docker buildx create --use --name amd64-builder
docker buildx build --platform linux/amd64 -t anywinter4079/m3-finetune:pytorch-2.10.0-cu128-flagembedding-dbc6005 ./docker_build --push
```

## SFT on RunPod

### Create `HF_USER` and `HF_TOKEN` as secrets to push the fine-tuned model to Hugging Face

On RunPod:

1. Click `Secrets` in the left sidebar.
2. Click `+ Create Secret`.
3. Type `HF_USER` as Secret Name.
4. Paste or type your Hugging Face username as Secret Value.
5. Click `Create Secret`.
6. Click `+ Create Secret`.
7. Type `HF_TOKEN` as Secret Name.
8. Paste or type your Hugging Face token as Secret Value.
9. Click `Create Secret`.

### Create a RunPod template

The template config is:

1. Container Image: `anywinter4079/m3-finetune:pytorch-2.10.0-cu128-flagembedding-dbc6005`
2. Container Disk: `250 GB`
3. Volume Disk: `500 GB`
4. Volume Mount Path: `/workspace`
5. TCP Ports: `22` for SSH

### Deploy the template, setting `HF_USER` and `HF_TOKEN` as environment variables

Note `HF_USER` and `HF_TOKEN` are added as environment variables on the deployment page, not to the template itself.

1. On `Deploy a Pod`, choose the GPU and click `Edit Template`.
2. Select the GPU count.
3. Click `Edit`.
4. Click `Environment Variables`.
5. Add `HF_TOKEN` as a locked secret value with key `HF_TOKEN`.
6. Add `HF_USER` as a locked secret value with key `HF_USER`.
7. Click `Set Overrides`.
8. Click `Deploy On-Demand`.

### Connect to the RunPod template via SSH

On your terminal, run the provided RunPod SSH command, in the format:

```bash
ssh <CONNECTION_STRING>@ssh.runpod.io -i ~/.ssh/<PRIVATE_KEY_FILE>
```

### Open another terminal window locally and send the training files to RunPod

From the root project directory locally, run:

```bash
scp -P <PORT> -i ~/.ssh/<PRIVATE_KEY_FILE> -r \
    data/finetune/<TIMESTAMP>/train.jsonl \
    finetune/sft.sh \
    root@<IP>:/workspace/finetune/
```

### Go back to the SSH session and run the fine-tuning script

Run:

```bash
cd finetune
chmod +x sft.sh
./sft.sh
```

## After Fine-Tuning

### 1. Set fine-tuning state

In `config/m3.py`, set: `IS_FINETUNED = True`

This switches:

- `config/eval.py` uses the `dev` split
- RRF swaps `bge_m3` for `bge_m3_muia`
- Eval runs `bge_m3` and `bge_m3_muia`
- `config/crawler_agent.py` adds `bge_m3_muia` to the encode variants

### 2. Redeploy the Modal services

Run:

```bash
./deploy.sh
```

This makes the encoder service aware of the fine-tuned encoder entry in `config/encoder.py`.

### 3. Re-encode the existing crawl with the fine-tuned encoder

Run:

```bash
modal run services/crawler_agent.py::run_crawler_agent
```

Keep:

- `REUSE_CRAWL = True`
- the same `REUSE_TIMESTAMP`

This reuses the same chunk files and rebuilds vectors.

### 4. Re-run the retrieval evaluation

Run:

```bash
modal run eval/run_data_variant_eval.py
```

This evaluates on the `dev` split and writes fresh retrieval outputs for the fine-tuned setup.
