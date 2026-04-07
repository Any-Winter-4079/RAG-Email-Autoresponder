#!/bin/bash

export WANDB_MODE=disabled

push_to_hub=true
negatives_cross_device=false
gradient_checkpointing=false

master_addr="127.0.0.1"
master_port=$(shuf -i 20000-29999 -n 1)
num_gpus=1

model_name_or_path="BAAI/bge-m3"
train_data="./train.jsonl"
output_dir="./checkpoints"
cache_path="./cache"
hub_model_id="$HF_USER/bge-m3-MUIA"

if [[ "$train_data" == *no_in_batch_neg* ]]; then
    echo "train_data name is disabling in-batch negatives: $train_data"
    exit 1
fi

if [ -z "$HF_HUB_CACHE" ]; then
    export HF_HUB_CACHE="$HOME/.cache/huggingface/hub"
fi

if [ "$push_to_hub" = true ]; then
    push_to_hub_args="--push_to_hub --hub_model_id $hub_model_id"
else
    push_to_hub_args=""
fi

if [ "$negatives_cross_device" = true ]; then
    negatives_cross_device_arg="--negatives_cross_device"
else
    negatives_cross_device_arg=""
fi

if [ "$gradient_checkpointing" = true ]; then
    gradient_checkpointing_arg="--gradient_checkpointing"
else
    gradient_checkpointing_arg=""
fi

model_args="\
    --model_name_or_path $model_name_or_path \
    --cache_dir $HF_HUB_CACHE \
"

data_args="\
    --train_data $train_data \
    --cache_path $cache_path \
    --train_group_size 8 \
    --query_max_len 73 \
    --passage_max_len 666 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
"

training_args="\
    --output_dir $output_dir \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --bf16 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 2 \
    --sub_batch_size -1 \
    --dataloader_drop_last True \
    --warmup_ratio 0.1 \
    $gradient_checkpointing_arg \
    --logging_steps 2 \
    --save_steps 100 \
    --save_total_limit 1 \
    $negatives_cross_device_arg \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --unified_finetuning True \
    --use_self_distill False \
    --fix_encoder False \
    $push_to_hub_args \
"

# `pip install -e .` from `/workspace/FlagEmbedding` registers the
# `FlagEmbedding` package in the active Python environment
# so `python -m FlagEmbedding...` works even though this script runs from the
# sibling `/workspace/finetune` directory. `-e` flag makes the install
# editable, albeit *not* used for now
cmd="torchrun --nproc_per_node $num_gpus \
    --master_addr $master_addr \
    --master_port $master_port \
    -m FlagEmbedding.finetune.embedder.encoder_only.m3 \
    $model_args \
    $data_args \
    $training_args \
"

echo $cmd
eval $cmd
