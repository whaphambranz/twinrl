#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-flightplans/experiment/main.toml}"
CKPT="${2:-runs/twinrl_main.pt}"

twinrl evaluate --config "${CONFIG}" --checkpoint "${CKPT}"
twinrl forecast --config "${CONFIG}" --checkpoint "${CKPT}" --limit 12
