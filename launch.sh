#!/usr/bin/env bash
set -eo pipefail
source "$(dirname -- "${BASH_SOURCE[0]}")/scripts/env.sh"
set -u
export DSV32_SHAPE_DEMO=1
export DSV32_DUAL_BATCH="${DSV32_DUAL_BATCH:-1}"
export VLLM_ASCEND_ENABLE_MLAPO=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
mode="${DSV32_EXECUTION_MODE:-graph}"
case "$mode" in
    graph) execution=(--compilation-config '{"mode":0,"cudagraph_mode":"FULL_DECODE_ONLY","cudagraph_capture_sizes":[1,2,3,4,5,6,7,8]}') ;;
    eager) execution=(--enforce-eager) ;;
    *) echo 'DSV32_EXECUTION_MODE must be graph or eager' >&2; exit 2 ;;
esac
if [[ "${DSV32_DUAL_BATCH_VALIDATE:-0}" == 1 && "$mode" != eager ]]; then
    echo 'DSV32_DUAL_BATCH_VALIDATE=1 requires launch.eager.sh (no graph capture).' >&2
    exit 2
fi
printf 'demo=%s\noutput=%s\nmode=%s dual_batch=%s\n' "$DSV32_DEMO_DIR" "$DSV32_OUTPUT_DIR" "$mode" "$DSV32_DUAL_BATCH"
# Use the active Python interpreter, not a global vllm executable's shebang.
exec "${PYTHON:-python3}" -m vllm.entrypoints.openai.api_server \
    --model "$DSV32_REPO_DIR/model" \
    --served-model-name deepseek-v3.2-shape-demo \
    --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}" \
    --tensor-parallel-size 1 --seed 0 \
    --dtype bfloat16 --load-format dummy \
    "${execution[@]}" \
    --max-model-len 32896 --max-num-seqs 8 --max-num-batched-tokens 1024 \
    --long-prefill-token-threshold 128 \
    --gpu-memory-utilization 0.90 --num-gpu-blocks-override 2057 \
    --no-enable-prefix-caching --generation-config vllm \
    --profiler-config.profiler torch \
    --profiler-config.torch_profiler_dir "$DSV32_OUTPUT_DIR/profiles" \
    --profiler-config.torch_profiler_record_shapes true
