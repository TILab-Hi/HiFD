#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
source "$SCRIPT_DIR/env.sh"

python "$ROOT/scripts/extract_face_detections.py" \
  --lmdb-path "$DATA_DIR/balanced.lmdb" \
  --retinaface-weights "$WEIGHTS_DIR/retinaface/Resnet50_Final.pth" \
  --output "$PRED_DIR/face_embedding/detections.json" \
  --device cuda
