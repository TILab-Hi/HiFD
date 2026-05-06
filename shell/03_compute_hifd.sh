#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
source "$SCRIPT_DIR/env.sh"

python "$ROOT/scripts/compute_hifd.py" \
  --predictions-dir "$PRED_DIR" \
  --results-dir "$RESULTS_DIR" \
  --methods-config "$ROOT/configs/methods.yaml" \
  --profiles-config "$ROOT/configs/profiles.yaml" \
  --constants-config "$ROOT/configs/constants.yaml"
