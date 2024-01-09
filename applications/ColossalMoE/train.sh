
PROJECT_NAME="Mixtral-Instruct-v0.1"

NUM_GPU=8
MODEL="/home/jiatong/models/Mixtral-8x7B-Instruct-v0.1"
SEQ_LENGTH=4096
LR=1e-5

TIMESTAMP=$(date +%Y-%m-%d-%H-%M-%S)
FULL_PROJECT_NAME="${PROJECT_NAME}-${TIMESTAMP}"
PARENT_SAVE_DIR="/home/litong/workspace/MyDev/MOE/checkpoints/"
PARENT_TENSORBOARD_DIR="/home/litong/workspace/MyDev/MOE/logs/"
PARENT_CONFIG_FILE="/home/litong/workspace/MyDev/MOE/configs/"

SAVE_DIR="${PARENT_SAVE_DIR}${FULL_PROJECT_NAME}"
TENSORBOARD_DIR="${PARENT_TENSORBOARD_DIR}${FULL_PROJECT_NAME}"
CONFIG_FILE="${PARENT_CONFIG_FILE}${FULL_PROJECT_NAME}.json"

declare -a datasets=(
    "/home/jiatong/data/mixtral-8x7b_32000/biology/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/chemistry/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/code/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/common/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/cot/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/cs/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/diet/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/finance/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/grammar/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/information_extraction/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/law/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/literature/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/lyrics/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/math/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/medical/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/persona/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/physics/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/psychology/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/rag/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/safety_and_responsibility/tokenized_and_packed"
    "/home/jiatong/data/mixtral-8x7b_32000/summarization/tokenized_and_packed"

)

data_bins=()
for dataset in "${datasets[@]}"; do
    for file in "$dataset"/*; do
        data_bins+=("$file")
    done
done

# hybrid
# torchrun --standalone --nproc_per_node $NUM_GPU \
colossalai run --nproc_per_node $NUM_GPU --hostfile "hostfile" \
    train_moe.py \
    --num_epoch 1 \
    --model_name $MODEL \
    --plugin "hybrid" \
    --batch_size 4 \
    --lr $LR \
    --zero_stage 1 \
    --pp_size 2 \
    --dp_size 1 \
    --ep_size 8 \
    --dataset ${data_bins[@]} \
    --tensorboard_dir $TENSORBOARD_DIR \
    --config_file $CONFIG_FILE \
    --output_path $SAVE_DIR \
    --microbatch_size 2 \
    --save_interval 5000 \
    --warmup_steps 100 \
