#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$SCRIPT_DIR/.." && pwd)
source "$SCRIPT_DIR/env.sh"

# Iterate over original + every method LMDB; produce one prediction LMDB per
# (task, source). Edit the SOURCES array to subset.
SOURCES=(balanced MI-FGSM PGD TI-DIM TIP-IM Chameleon Adv-Makeup AMT-GAN \
         CIAGAN DeID-rPPG G2Face WeakenDiff DiffAM)

for SRC_NAME in "${SOURCES[@]}"; do
  if [[ "$SRC_NAME" == "balanced" ]]; then
    LMDB="$DATA_DIR/balanced.lmdb"
  else
    LMDB="$DATA_DIR/deid/$SRC_NAME.lmdb"
  fi

  echo "=== $SRC_NAME ==="

  python "$ROOT/scripts/extract_age_gender.py" \
    --lmdb-paths "$LMDB" \
    --detections "$PRED_DIR/face_embedding/detections.json" \
    --weights "$WEIGHTS_DIR/mivolo/model_utk_age_gender_4.23_97.69.pth" \
    --output "$PRED_DIR/age_gender/$SRC_NAME.lmdb" \
    --device cuda --resume

  python "$ROOT/scripts/extract_ethnicity.py" \
    --lmdb-paths "$LMDB" \
    --detections "$PRED_DIR/face_embedding/detections.json" \
    --weights "$WEIGHTS_DIR/fairface/res34_fair_align_multi_7_20190809.pt" \
    --output "$PRED_DIR/ethnicity/$SRC_NAME.lmdb" \
    --device cuda --resume

  python "$ROOT/scripts/extract_macro_exp.py" \
    --lmdb-paths "$LMDB" \
    --weights "$WEIGHTS_DIR/poster/affectnet-7cls.pth" \
    --output "$PRED_DIR/macro_exp/$SRC_NAME.lmdb" \
    --device cuda --resume

  python "$ROOT/scripts/extract_landmark.py" \
    --lmdb-paths "$LMDB" \
    --face-landmarker "$WEIGHTS_DIR/mediapipe/face_landmarker.task" \
    --output "$PRED_DIR/landmark/$SRC_NAME.lmdb" \
    --resume

  python "$ROOT/scripts/extract_gaze.py" \
    --lmdb-paths "$LMDB" \
    --detections "$PRED_DIR/face_embedding/detections.json" \
    --weights "$WEIGHTS_DIR/l2cs/L2CSNet_gaze360.pkl" \
    --output "$PRED_DIR/gaze/$SRC_NAME.lmdb" \
    --device cuda --resume

  python "$ROOT/scripts/extract_embeddings.py" \
    --lmdb-paths "$LMDB" \
    --detections "$PRED_DIR/face_embedding/detections.json" \
    --weights-root "$WEIGHTS_DIR" \
    --output "$PRED_DIR/face_embedding/$SRC_NAME.lmdb" \
    --device cuda --resume

  if [[ "$SRC_NAME" != "balanced" ]]; then
    python "$ROOT/scripts/extract_lpips.py" \
      --balanced-lmdb "$DATA_DIR/balanced.lmdb" \
      --lmdb-paths "$LMDB" \
      --output "$PRED_DIR/lpips/$SRC_NAME.lmdb" \
      --device cuda

    python "$ROOT/scripts/extract_niqe.py" \
      --lmdb-paths "$LMDB" \
      --output "$PRED_DIR/niqe/$SRC_NAME.lmdb" \
      --device cuda --resume
  fi
done
