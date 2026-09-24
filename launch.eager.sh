#!/usr/bin/env bash
set -eo pipefail
export DSV32_EXECUTION_MODE=eager
exec bash "$(dirname -- "${BASH_SOURCE[0]}")/launch.sh" "$@"
