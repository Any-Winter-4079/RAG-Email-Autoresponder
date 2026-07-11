#!/bin/bash

export WANDB_MODE=disabled
export PYTORCH_ALLOC_CONF=expandable_segments:True

push_to_hub=true
negatives_cross_device=false

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

if [ "$sample_all_effective_in_batch_negatives" = true ]; then
    sample_all_effective_in_batch_negatives_arg="--sample_all_effective_in_batch_negatives True"
else
    sample_all_effective_in_batch_negatives_arg=""
fi

model_args="\
    --model_name_or_path $model_name_or_path \
    --cache_dir $HF_HUB_CACHE \
"

data_args="\
    --train_data $train_data \
    --cache_path $cache_path \
    --train_group_size 8 \
    --query_max_len 120 \
    --passage_max_len 1953 \
    --pad_to_multiple_of 8 \
    --knowledge_distillation False \
"

training_args="\
    --output_dir $output_dir \
    --overwrite_output_dir \
    --learning_rate 1e-5 \
    --bf16 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 8 \
    --sub_batch_size 4 \
    --gradient_checkpointing True \
    --dataloader_drop_last True \
    --lr_scheduler_type linear \
    --warmup_ratio 0.1 \
    --logging_steps 25 \
    --save_strategy epoch \
    --save_total_limit 1 \
    $negatives_cross_device_arg \
    --temperature 0.02 \
    --sentence_pooling_method cls \
    --normalize_embeddings True \
    --filter_duplicate_positive_passages True \
    --filter_group_duplicate_positive_passages False \
    --filter_similar_group_duplicate_positive_passages False \
    --max_effective_in_batch_denominator_size 16 \
    --sample_all_effective_in_batch_negatives False \
    --sparse_loss_weight 0.1 \
    --colbert_loss_weight 1.0 \
    --sparse_self_distill_loss_weight 0.1 \
    --colbert_self_distill_loss_weight 1.0 \
    --unified_finetuning True \
    --use_self_distill True \
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
