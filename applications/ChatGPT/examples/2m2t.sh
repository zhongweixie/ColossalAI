set_n_least_used_CUDA_VISIBLE_DEVICES() {
    local n=${1:-"9999"}
    echo "GPU Memory Usage:"
    local FIRST_N_GPU_IDS=$(nvidia-smi --query-gpu=memory.used --format=csv \
        | tail -n +2 \
        | nl -v 0 \
        | tee /dev/tty \
        | sort -g -k 2 \
        | awk '{print $1}' \
        | head -n $n)
    export CUDA_VISIBLE_DEVICES=$(echo $FIRST_N_GPU_IDS | sed 's/ /,/g')
    echo "Now CUDA_VISIBLE_DEVICES is set to:"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
}

set_n_least_used_CUDA_VISIBLE_DEVICES 2

export CUDA_VISIBLE_DEVICES="0,1,2,3"
export RAY_NAMESPACE="lccsr"

python 2m2t.py "./awesome-chatgpt-prompts/prompts.csv" \
    --maker_strategy naive --trainer_strategy naive --lora_rank 2 \
    --num_episodes 10 --max_timesteps 10 --update_timesteps 10 \
    --max_epochs 10  --debug 