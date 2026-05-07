# HiFD: A Hierarchical Metric for Face De-Identification Evaluation

This repository is the official implementation of paper titled *How Private is Private? A Comparative Study for Face De-Identification* (under double-blind review).

## Requirements

```bash
pip install -r requirements.txt
```

Pretrained weight files are listed under `configs/estimators.yaml`. Download
each into `$WEIGHTS_DIR/<subpath>` matching the layout in that file. Public
sources for each weight are linked in the [Pre-trained Models](#pre-trained-models)
section below.

## Dataset preparation

Pack a folder of images into the expected LMDB format:

```bash
python scripts/build_dataset_lmdb.py \
    --input-dir /path/to/face/images \
    --output    $DATA_DIR/balanced.lmdb \
    --map-size-gb 600
```

The same script can be used for each de-identification method's outputs.

## Extraction (Stage A)

Source your environment and run the three numbered shell scripts in order:

```bash
source shell/env.sh
bash shell/01_extract_face_detections.sh
bash shell/02_extract_all_predictions.sh
```

Each Python script underneath supports `--resume` to checkpoint progress and
skip already-processed keys.

## Evaluation (Stage B)

```bash
bash shell/03_compute_hifd.sh
```

Outputs:

- `$RESULTS_DIR/scores.json`  — per-method full breakdown
- `$RESULTS_DIR/scores.csv`   — flat table

## Pre-trained Models

| Component | Source | Layout under `$WEIGHTS_DIR` |
|---|---|---|
| RetinaFace (ResNet-50) | https://github.com/biubug6/Pytorch_Retinaface | `retinaface/Resnet50_Final.pth` |
| ArcFace R100 (MS1MV3, fp16) | https://github.com/deepinsight/insightface | `ms1mv3_arcface_r100_fp16/backbone.pth` |
| CosFace R50 (Glint360k, fp16) | https://github.com/deepinsight/insightface | `glint360k_cosface_r50_fp16_0.1/backbone.pth` |
| AdaFace IR50 (MS1MV2) | https://github.com/mk-minchul/AdaFace | `adaface_pre_trained/adaface_ir50_ms1mv2.ckpt` |
| MiVOLO (UTK age+gender) | https://github.com/WildChlamydia/MiVOLO | `mivolo/model_utk_age_gender_4.23_97.69.pth` |
| FairFace (7-race) | https://github.com/joojs/fairface | `fairface/res34_fair_align_multi_7_20190809.pt` |
| L2CS (Gaze360) | https://github.com/Ahmednull/L2CS-Net | `l2cs/L2CSNet_gaze360.pkl` |
| MediaPipe Face Landmarker | https://developers.google.com/mediapipe | `mediapipe/face_landmarker.task` |
| POSTER (AffectNet 7-class) | https://github.com/zczcwh/POSTER | `poster/affectnet-7cls.pth` |
| SAMER (CASME II 5-class, 26 LOSO folds) | https://github.com/Justin900429/mimicking-annotation-micro-expression-recognition | `download.sh` |
| PhysNet | https://github.com/ubicomplab/rPPG-Toolbox | `rppg/PhysNet.pth` |
| EfficientPhys | https://github.com/ubicomplab/rPPG-Toolbox | `rppg/EfficientPhys.pth` |
| FactorizePhys | https://github.com/ubicomplab/rPPG-Toolbox | `rppg/FactorizePhys.pth` |

## Results

Twelve face de-identification methods evaluated under the **Balanced** profile
(equal weights on Privacy / Quality / U1 / U2 / U3):

| Method     | $\bar{P}$ | $Q$   | $U_1$ | $U_2$ | $U_3$ | HiFD (Balanced) |
|------------|-----------|-------|-------|-------|-------|-----------------|
| MI-FGSM    | 0.134     | 0.360 | 0.933 | 0.868 | 0.474 | 0.343           |
| PGD        | 0.113     | 0.363 | 0.941 | 0.879 | 0.566 | 0.322           |
| TI-DIM     | 0.500     | 0.342 | 0.794 | 0.821 | 0.424 | 0.512           |
| TIP-IM     | 0.496     | 0.446 | 0.834 | 0.878 | 0.624 | **0.610**       |
| Chameleon  | 0.239     | 0.474 | 0.873 | 0.889 | 0.799 | 0.509           |
| Adv-Makeup | 0.012     | 0.524 | 0.962 | 0.941 | 0.931 | 0.056           |
| AMT-GAN    | 0.156     | 0.481 | 0.877 | 0.745 | 0.148 | 0.282           |
| CIAGAN     | 0.450     | 0.276 | 0.540 | 0.660 | 0.247 | 0.377           |
| DeID-rPPG  | 0.013     | 0.407 | 0.935 | 0.889 | 0.907 | 0.061           |
| G2Face     | 0.210     | 0.443 | 0.833 | 0.848 | 0.326 | 0.401           |
| WeakenDiff | 0.222     | 0.371 | 0.850 | 0.792 | 0.185 | 0.332           |
| DiffAM     | 0.159     | 0.471 | 0.820 | 0.811 | 0.557 | 0.395           |

Best HiFD (Balanced) shown in **bold**. Full per-sub-score breakdown (age,
gender, ethnicity, macro-exp, landmark, gaze, micro-exp, BVP, HR) and the
PrivacyFirst / Clinical profiles are available in `$RESULTS_DIR/scores.csv`
after running the pipeline. Numbers reproduce Table 1 of the paper.

## License

MIT — see `LICENSE`.
