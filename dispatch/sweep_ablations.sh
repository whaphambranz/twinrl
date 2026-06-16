#!/usr/bin/env bash
set -euo pipefail

for cfg in flightplans/experiment/ablation_*.toml; do
    name="$(basename "${cfg}" .toml)"
    twinrl train --config "${cfg}" --checkpoint "runs/${name}.pt"
    twinrl evaluate --config "${cfg}" --checkpoint "runs/${name}.pt"
done
