#!/bin/bash
export HF_ALLOW_CODE_EVAL="1"
export CUDA_VISIBLE_DEVICES=0,1
export confirm_run_unsafe_code=True
# few-shot
declare -A TASK_SHOTS
# Reasoning
TASK_SHOTS["bbh"]=3
TASK_SHOTS["gsm8k"]=8
# Knowledge      
TASK_SHOTS["mmlu"]=4
TASK_SHOTS["truthfulqa_mc2"]=0
# Code Generation
TASK_SHOTS["mbpp"]=1
TASK_SHOTS["humaneval"]=0


MODEL_PATH="/path/to/your/adg_llama3_checkpoints"


for task in "${!TASK_SHOTS[@]}"; do
    shots=${TASK_SHOTS[$task]}
    echo "eval $task ($shots-shot)..."
    
    torchrun --nproc_per_node=2 --master_port=29500 \
        -m lm_eval \
        --model hf \
        --model_args pretrained=${MODEL_PATH},dtype=bfloat16 \
        --tasks $task \
        --num_fewshot $shots \
        --batch_size 8 \
        --output_path /path/to/your/${task} \
        --confirm_run_unsafe_code
    
    echo "$task finish"
    echo "---"
done