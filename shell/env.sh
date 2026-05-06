# $DST/shell/env.sh — source this before running any pipeline script.
# Edit the paths below for your environment.

# Where the original LMDB and per-method de-id LMDBs live
export DATA_DIR=${DATA_DIR:-/path/to/datasets}

# Where pretrained model weights are stored (layout in configs/estimators.yaml)
export WEIGHTS_DIR=${WEIGHTS_DIR:-/path/to/pretrained_weights}

# Where to write per-task LMDB predictions and final results
export PRED_DIR=${PRED_DIR:-./predictions}
export RESULTS_DIR=${RESULTS_DIR:-./results}

# External heavy repos (clone these separately)
export MIVOLO_REPO=${MIVOLO_REPO:-/path/to/MiVOLO}
export SAMER_REPO=${SAMER_REPO:-/path/to/SAMER}
export RPPG_REPO=${RPPG_REPO:-/path/to/rPPG-Toolbox}

export OMP_NUM_THREADS=1
