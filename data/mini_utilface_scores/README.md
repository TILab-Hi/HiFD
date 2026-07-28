# mini-UtilFace validation bundle

Lets you reproduce and interrogate HiFD without acquiring WebFace42M, Glint360K,
MS1MV2 or MegaFace. Requires only `numpy`.

```
python verify.py
```

## What is here

| file | contents |
|---|---|
| `subscores.npz` | per-image sub-scores, 12 methods x 2,000 images x 10 fields, float32 |
| `keys.csv` | the 2,000 image keys with identity label, gender, ethnicity |
| `video_axes.json` | per-method micro-expression and rPPG scores (see caveat 2) |
| `published_scores.csv` | the full-benchmark per-method table, for comparison |
| `verify.py` | recomputes everything below from `subscores.npz` |

The 2,000 images are a stratified sample across all 14 gender-by-ethnicity
strata (131 to 160 images each) over 1,034 identities, so the subset inherits
the benchmark's demographic balance rather than being a convenience sample.

## Why sub-scores and not images

The release position of this work is that we distribute identity keys and
scripts, never redistributed source images, because the four source corpora are
research-licensed and one of them derives from MS-Celeb-1M. Shipping 2,000
source faces to make validation convenient would contradict that position, so
this bundle ships the scalar agreement values HiFD is actually built from.

It also contains **no face embeddings**. Embeddings are themselves biometric
data, and the privacy sub-score is the only thing they are needed for, so we
ship the sub-score rather than the template.

## What `verify.py` checks

1. **Every component, recomputed from per-image values.** `P`, `Q`, `U1`, `U2`,
   `U3` per method, then the weighted harmonic mean. Reproduces the published
   `HiFD_Balanced` to within 0.004 on all twelve methods.
2. **Ranking agreement** with the published table: Spearman 0.993, TIP-IM top-1.
   The only pair that reorders is TI-DIM and Chameleon, which the paper reports
   as a statistical tie, so the subset reproduces even the tie.
3. **All three profiles** (PrivacyFirst, Balanced, Clinical).
4. **The continuous weight sweep**, `w_P` from 0.2 to 0.6, showing TIP-IM leads
   throughout and CIAGAN rises with privacy weight.
5. **An identity-level bootstrap** over the 1,034 identities, resampling
   identity blocks rather than images: TIP-IM is top-1 in 100% of resamples.

## What it does not check, and why

1. **The estimators themselves.** These are cached outputs. Verifying that
   MiVOLO, FairFace, POSTER, L2CS-Net and the three recognizers produce these
   values requires the images, which is precisely what we do not redistribute.
   Reproducing the estimators end to end needs the full pipeline in the code
   repository plus the source corpora obtained from their providers.
2. **The video axes are per-method constants here, not per-image.** L2
   micro-expression is measured on DFME and L3 rPPG on PURE, neither of which is
   image-level, so `video_axes.json` carries the published scalars so that the
   composite can still be assembled. A consequence: the bootstrap in `verify.py`
   resamples only the image axes, so its intervals are narrower than the paper's
   full bootstrap, which also resamples DFME and PURE subjects.
3. **WeakenDiff covers 1,526 of the 2,000 images (76%).** That method fails to
   produce output on part of the cohort, the same 75% coverage it has on the
   full benchmark. Its column is computed over the images it does cover, and
   missing entries are stored as `NaN` rather than silently dropped.

## Field reference

`subscores.npz` holds one `(12, 2000)` float32 array per field, indexed by the
`methods` and `keys` arrays in the same file. `NaN` means the estimator produced
no output for that image.

| field | definition |
|---|---|
| `privacy` | mean over ArcFace/CosFace/AdaFace of `0.5 * (1 - cos)` |
| `age` | `1 - abs(age_o - age_d) / 100` |
| `gender`, `ethnicity`, `macro_exp` | `1 - TV(posterior_o, posterior_d)` |
| `landmark` | `1 - NME / 0.10`, interocular-normalised, floored at 0 |
| `gaze` | `1 - angle_deg / 90` between gaze vectors |
| `lpips`, `niqe` | raw values on the de-identified image |
| `quality` | `0.5 * (1 - min(lpips,1)) + 0.5 * clip((5 - niqe)/5, 0, 1)` |

`U1` is the mean of the five L1 fields; `U2` the mean of `gaze` and the DFME
micro-expression score; `U3` the mean of the PURE BVP and HR scores.
