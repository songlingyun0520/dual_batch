#!/usr/bin/env bash
# Source this file. Vendor environment scripts are not safe under nounset.
DSV32_REPO_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${CANN_ENV_FILE:-}" ]]; then
    source "$CANN_ENV_FILE"
elif [[ -f /usr/local/Ascend/cann/set_env.sh ]]; then
    source /usr/local/Ascend/cann/set_env.sh
elif [[ -f /usr/local/Ascend/ascend-toolkit/set_env.sh ]]; then
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
else
    echo 'CANN set_env.sh not found; set CANN_ENV_FILE.' >&2
    return 1
fi
if [[ -n "${ATB_ENV_FILE:-}" ]]; then
    source "$ATB_ENV_FILE"
elif [[ -f /usr/local/Ascend/nnal/atb/set_env.sh ]]; then
    source /usr/local/Ascend/nnal/atb/set_env.sh
else
    echo 'NNAL/ATB set_env.sh not found; set ATB_ENV_FILE.' >&2
    return 1
fi
export DSV32_DEMO_DIR="$DSV32_REPO_DIR"
export DSV32_OUTPUT_DIR="${DSV32_OUTPUT_DIR:-$DSV32_REPO_DIR/runtime}"
mkdir -p "$DSV32_OUTPUT_DIR"
export PYTHONPATH="$DSV32_REPO_DIR${PYTHONPATH:+:$PYTHONPATH}"
