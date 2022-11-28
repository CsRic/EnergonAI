CUDA_VISIBLE_DEVICES_set_n_least_memory_usage() {
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

export GPU_NUM=2
export DATASET="/home/lccsr/data2/files_2022/bloom/sandbox/BloomFunctionSandbox/bloom40B" 
# /data2/users/lczht/bloom-560m 
#/data2/users/lccsr/bloom3b/data 
#/data2/users/lccsr/bloom1b7/data
# "/home/lccsr/data2/files_2022/bloom/sandbox/BloomFunctionSandbox/bloom40B"

export USE_CONFIG=0 # set up a random model from config.json
export USE_INT8=1

CUDA_VISIBLE_DEVICES_set_n_least_memory_usage ${GPU_NUM} 
if [[ ${USE_CONFIG} == 1 ]]; then
USE_CONFIG_FLAG="--use_config"
else
USE_CONFIG_FLAG=""
fi

if [[ ${USE_INT8} == 1 ]]; then
DTYPE="int8"
else
DTYPE="fp16"
fi

python server.py --tp ${GPU_NUM} --name ${DATASET} ${USE_CONFIG_FLAG} --dtype ${DTYPE} --max_batch_size 4
