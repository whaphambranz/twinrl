#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-flightplans/experiment/main.toml}"
CKPT="${2:-runs/twinrl_main.pt}"

twinrl train --config "${CONFIG}" --checkpoint "${CKPT}"
