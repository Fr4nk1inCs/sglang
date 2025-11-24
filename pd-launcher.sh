#!/usr/bin/env bash

set -e

export NCCL_P2P_LEVEL=PXB
export NCCL_CROSS_NIC=0
export NCCL_NET_GDR_LEVEL=SYS
export NCCL_SOCKET_IFNAME=ens5f0
export GLOO_SOCKET_IFNAME="ens5f0"
export NCCL_IB_HCA=mlx5_0,mlx5_1

# Load Balancer
MODEL="/nfs/shared_LLM_model/Qwen/Qwen3-8B"
PORT=12000
MEM_FRAC="0.7"

# Server
PP_SIZE=4
PREFILL_IP="127.0.0.1"
PREFILL_PORT=30000
MAX_PREFILL_TOKENS=8192
DECODE_IP="127.0.0.1"
DECODE_PORT=30001
if [ "$PREFILL_IP" == "$DECODE_IP" ]; then
    DECODE_BASE_GPU_ID="$PP_SIZE"
else
    DECODE_BASE_GPU_ID=0
fi

# SGLang Config
COMMON_ARGS=(
    python -m sglang.launch_server
    --model-path "$MODEL"
    --disaggregation-transfer-backend nixl
    --disaggregation-ib-device "$NCCL_IB_HCA"
    --mem-fraction-static "$MEM_FRAC"
    # --chunked-prefill-size -1
    --disable-radix-cache
    --disable-cuda-graph
    --trust-remote-code
    --random-seed 1234
    --pp-size "$PP_SIZE"
    --page-size 128
)

prefill() {
    local args=(
        "${COMMON_ARGS[@]}"
        --disaggregation-mode prefill
        --host "$PREFILL_IP"
        --port "$PREFILL_PORT"
        --max-prefill-tokens "$MAX_PREFILL_TOKENS"
    )

    echo "${args[@]}"
    "${args[@]}"
}

decode() {
    local args=(
        "${COMMON_ARGS[@]}"
        --disaggregation-mode decode
        --host "$DECODE_IP"
        --port "$DECODE_PORT"
        --base-gpu-id "$DECODE_BASE_GPU_ID"
    )

    echo "${args[@]}"
    "${args[@]}"
}

wait_for_address() {
    local ip="$1"
    local port="${2:-22}"

    while ! nc -z "$ip" "$port"; do
        echo "Waiting for ${ip}:${port} to be ready..."
        sleep 1
    done
}

load_balancer() {
    wait_for_address "$PREFILL_IP" "$PREFILL_PORT"
    wait_for_address "$DECODE_IP" "$DECODE_PORT"

    python -m sglang.srt.disaggregation.mini_lb \
        --prefill "http://$PREFILL_IP:$PREFILL_PORT" \
        --decode "http://$DECODE_IP:$DECODE_PORT" \
        --port "$PORT"
}

# Benchmark
NUM_PROMPTS=1024
IO_LENS=(
    "500,500"
    "500,1000"
)
RANGE_RATIO=0.8
MAX_CONCURRENCY=500

benchmark() {
    wait_for_address "localhost" "$PORT"

    for IO_LEN in "${IO_LENS[@]}"; do
        INPUT_LEN=$(echo "$IO_LEN" | cut -d',' -f1)
        OUTPUT_LEN=$(echo "$IO_LEN" | cut -d',' -f2)
        echo "Running benchmark with INPUT_LEN=$INPUT_LEN, OUTPUT_LEN=$OUTPUT_LEN"

        local args=(
            python3 -m sglang.bench_serving --backend sglang
            --port "$PORT"
            --model "$MODEL"
            --request-rate 10000
            --dataset-name random
            --dataset-path /nfs/shared_LLM_dataset/ShareGPT_Vicuna_unfiltered/ShareGPT_V3_unfiltered_cleaned_split.json
            --random-output-len "$OUTPUT_LEN"
            --random-input-len "$INPUT_LEN"
            --random-range-ratio "$RANGE_RATIO"
            --num-prompt "$NUM_PROMPTS"
            --output-file "$OUTPUT_FILE"
            --max-concurrency "$MAX_CONCURRENCY"
            --warmup-requests 16
        )

        wait_for_address "$PREFILL_IP" "$PREFILL_PORT"

        echo "${args[@]}"
        echo "Input/Output length: $INPUT_LEN/$OUTPUT_LEN"
        "${args[@]}"
    done
}

if [ "$1" == "prefill" ]; then
    prefill
elif [ "$1" == "decode" ]; then
    decode
elif [ "$1" == "load_balancer" ]; then
    load_balancer
elif [ "$1" == "benchmark" ]; then
    benchmark
else
    echo "Usage: $0 {prefill|decode|load_balancer|benchmark}"
    exit 1
fi
