"""Aggregate per-method HiFD scores from the predictions/ directory.

Reads the per-key LMDBs (landmark, face_embedding, age_gender, ethnicity,
macro_exp, micro_exp, gaze, niqe, lpips) and the rPPG .npy files, computes
all sub-agreements according to the formulas in the manuscript, and writes:

    <results_dir>/scores.json   — per-method full breakdown
    <results_dir>/scores.csv    — flat table
    <results_dir>/scores.tex    — LaTeX booktabs table

Usage
-----
python scripts/compute_hifd.py \
    --predictions-dir  data/predictions \
    --results-dir      results/hifd \
    --methods-config   configs/methods.yaml \
    --profiles-config  configs/profiles.yaml \
    --constants-config configs/constants.yaml
"""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import yaml

# Allow running as `python scripts/compute_hifd.py` from the repo root.
_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from src.agreements import s_age, s_categorical, s_landmark, s_gaze
from src.privacy import P_single
from src.quality import Q_score
from src.rppg_signal import bvp_agreement, hr_agreement, hr_from_bvp
from src.composite import hifd, load_profiles
from src.lmdb_utils import lmdb_to_dict, load_paths_config


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def log(msg: str):
    from datetime import datetime
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def gender_to_prob(gender: str, gender_score: float) -> np.ndarray:
    """Convention: index 0 = male, index 1 = female."""
    if gender == "male":
        return np.array([gender_score, 1.0 - gender_score])
    return np.array([1.0 - gender_score, gender_score])


def softmax_np(logits) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def _gaze_vec(yaw: float, pitch: float) -> np.ndarray:
    """L2CS convention: x=-cos(p)*sin(y), y=-sin(p), z=-cos(p)*cos(y)."""
    cy, sy = math.cos(yaw), math.sin(yaw)
    cp, sp = math.cos(pitch), math.sin(pitch)
    v = np.array([-cp * sy, -sp, -cp * cy])
    n = np.linalg.norm(v)
    return v / (n + 1e-12)


# ---------------------------------------------------------------------------
# Per-method aggregation
# ---------------------------------------------------------------------------

def compute_method(method: str, pred_dir: Path, bal: dict,
                   dfme: dict | None, pure: dict | None,
                   rppg_methods: list[str]) -> dict:
    """Compute per-method HiFD component scores.

    Parameters
    ----------
    method      : method name (used to locate LMDB files)
    pred_dir    : path to the predictions directory
    bal         : preloaded dict of original (balanced) prediction LMDBs
    dfme        : preloaded dict for DFME micro-expression originals (or None)
    pure        : preloaded dict for PURE rPPG originals (or None)
    rppg_methods: list of rPPG estimator names

    Returns
    -------
    dict with all sub-scores and composite HiFD scores.
    """
    r: dict = {}

    # -- Privacy --
    # For each image, compute P_single per backbone then average across
    # backbones, giving one privacy value per image.  P_bar across images
    # is the final mean.
    dei_emb = lmdb_to_dict(str(pred_dir / "face_embedding" / f"{method}.lmdb"))
    priv = []
    for k, v_d in dei_emb.items():
        v_o = bal["face_embedding"].get(k)
        if v_o is None:
            continue
        bb_scores = [
            P_single(
                np.asarray(v_o[bb], dtype=np.float64),
                np.asarray(v_d[bb], dtype=np.float64),
            )
            for bb in ("arcface", "cosface", "adaface")
        ]
        priv.append(float(np.mean(bb_scores)))
    r["privacy"] = float(np.mean(priv)) if priv else None
    r["privacy_n"] = len(priv)

    # -- Quality: LPIPS (de-id vs orig) + NIQE (de-id only) --
    lp = lmdb_to_dict(str(pred_dir / "lpips" / f"{method}.lmdb"))
    nq_d = lmdb_to_dict(str(pred_dir / "niqe" / f"{method}.lmdb"))
    qs = []
    for k, v_l in lp.items():
        v_n = nq_d.get(k)
        if v_n is None:
            continue
        qs.append(Q_score(v_l["lpips"], v_n["niqe"]))
    r["quality"] = float(np.mean(qs)) if qs else None
    r["quality_n"] = len(qs)
    r["lpips_mean"] = float(np.mean([v["lpips"] for v in lp.values()])) if lp else None
    r["niqe_mean"] = float(np.mean([v["niqe"] for v in nq_d.values()])) if nq_d else None

    # -- U1: age, gender, ethnicity, macro-expression, landmark --
    ag_d = lmdb_to_dict(str(pred_dir / "age_gender" / f"{method}.lmdb"))
    eth_d = lmdb_to_dict(str(pred_dir / "ethnicity" / f"{method}.lmdb"))
    mx_d = lmdb_to_dict(str(pred_dir / "macro_exp" / f"{method}.lmdb"))
    lm_d = lmdb_to_dict(str(pred_dir / "landmark" / f"{method}.lmdb"))

    age_s, gen_s = [], []
    for k, v_d in ag_d.items():
        v_o = bal["age_gender"].get(k)
        if v_o is None:
            continue
        age_s.append(s_age(float(v_o["age"]), float(v_d["age"])))
        p_o = gender_to_prob(v_o["gender"], v_o["gender_score"])
        p_d = gender_to_prob(v_d["gender"], v_d["gender_score"])
        gen_s.append(s_categorical(p_o, p_d))

    eth_s = []
    for k, v_d in eth_d.items():
        v_o = bal["ethnicity"].get(k)
        if v_o is None:
            continue
        eth_s.append(s_categorical(
            np.asarray(v_o["race_prob"], dtype=np.float64),
            np.asarray(v_d["race_prob"], dtype=np.float64),
        ))

    mx_s = []
    for k, v_d in mx_d.items():
        v_o = bal["macro_exp"].get(k)
        if v_o is None:
            continue
        mx_s.append(s_categorical(
            np.asarray(v_o["probs"], dtype=np.float64),
            np.asarray(v_d["probs"], dtype=np.float64),
        ))

    lm_s = []
    for k, v_d in lm_d.items():
        v_o = bal["landmark"].get(k)
        if v_o is None:
            continue
        lm_s.append(s_landmark(
            np.asarray(v_o["points"], dtype=np.float64),
            np.asarray(v_d["points"], dtype=np.float64),
            np.asarray(v_o["left_eye_center"], dtype=np.float64),
            np.asarray(v_o["right_eye_center"], dtype=np.float64),
        ))

    r["age"] = float(np.mean(age_s)) if age_s else None
    r["gender"] = float(np.mean(gen_s)) if gen_s else None
    r["ethnicity"] = float(np.mean(eth_s)) if eth_s else None
    r["macro_exp"] = float(np.mean(mx_s)) if mx_s else None
    r["landmark"] = float(np.mean(lm_s)) if lm_s else None

    u1_parts = [r[k] for k in ("age", "gender", "ethnicity", "macro_exp", "landmark")
                if r[k] is not None]
    r["U1"] = float(np.mean(u1_parts)) if u1_parts else None

    # -- U2: gaze (balanced) + micro-expression (DFME) --
    gz_d = lmdb_to_dict(str(pred_dir / "gaze" / f"{method}.lmdb"))
    gaze_s = []
    for k, v_d in gz_d.items():
        v_o = bal["gaze"].get(k)
        if v_o is None:
            continue
        g_o = _gaze_vec(float(v_o["yaw"]), float(v_o["pitch"]))
        g_d = _gaze_vec(float(v_d["yaw"]), float(v_d["pitch"]))
        gaze_s.append(s_gaze(g_o, g_d))
    r["gaze"] = float(np.mean(gaze_s)) if gaze_s else None

    me_s = []
    if dfme is not None:
        me_d = lmdb_to_dict(str(pred_dir / "micro_exp" / f"{method}.lmdb"))
        for k, v_d in me_d.items():
            v_o = dfme["micro_exp"].get(k)
            if v_o is None:
                continue
            p_o = softmax_np(v_o["logits"])
            p_d = softmax_np(v_d["logits"])
            me_s.append(s_categorical(p_o, p_d))
    r["micro_exp"] = float(np.mean(me_s)) if me_s else None

    u2_parts = [x for x in (r["gaze"], r["micro_exp"]) if x is not None]
    r["U2"] = float(np.mean(u2_parts)) if u2_parts else None

    # -- U3: rPPG ensemble (BVP + HR) on PURE --
    bvp_all, hr_all = [], []
    if pure is not None:
        for rm in rppg_methods:
            d_path = pred_dir / "rppg" / f"{method}_{rm}.npy"
            if not d_path.exists():
                continue
            d = np.load(str(d_path), allow_pickle=True).item()
            o = pure["rppg"][rm]
            for sid, chunks in o.items():
                for cid, sig_o in chunks.items():
                    sig_d = d.get(sid, {}).get(cid)
                    if sig_d is None:
                        continue
                    bvp_all.append(bvp_agreement(sig_o, sig_d))
                    hr_all.append(hr_agreement(sig_o, sig_d))
    r["bvp"] = float(np.mean(bvp_all)) if bvp_all else None
    r["hr"] = float(np.mean(hr_all)) if hr_all else None
    if r["bvp"] is not None and r["hr"] is not None:
        r["U3"] = 0.5 * (r["bvp"] + r["hr"])
    else:
        r["U3"] = None

    # -- Composites: one HiFD score per application profile --
    # hifd() from src.composite auto-handles missing components (None → excluded
    # and weights renormalized).  We pass each level as a named positional arg.
    # Note: the 'P' key in profiles maps to P_bar; we pass r["privacy"] for it.
    r["_components"] = {
        "P": r["privacy"], "Q": r["quality"],
        "U1": r["U1"], "U2": r["U2"], "U3": r["U3"],
    }
    return r


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-method HiFD scores from extracted predictions.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--predictions-dir", required=True,
        help="Directory containing per-modality LMDB subdirs and rPPG .npy files.",
    )
    parser.add_argument(
        "--results-dir", required=True,
        help="Output directory for scores.json, scores.csv, scores.tex.",
    )
    parser.add_argument(
        "--methods-config",
        default=str(_REPO / "configs" / "methods.yaml"),
        help="YAML file listing de-identification methods to evaluate.",
    )
    parser.add_argument(
        "--profiles-config",
        default=str(_REPO / "configs" / "profiles.yaml"),
        help="YAML file with application profile weight dicts.",
    )
    parser.add_argument(
        "--estimators-config",
        default=str(_REPO / "configs" / "estimators.yaml"),
        help="YAML file listing estimator models; rPPG method names are read "
             "from the 'rppg' key.",
    )
    parser.add_argument(
        "--constants-config",
        default=str(_REPO / "configs" / "constants.yaml"),
        help="YAML file with metric hyper-parameter constants (currently unused "
             "by this script; reserved for future use).",
    )
    args = parser.parse_args()

    pred_dir = Path(args.predictions_dir)
    results_dir = Path(args.results_dir)

    # -- Load configs --
    with open(args.methods_config) as f:
        methods_cfg = yaml.safe_load(f)
    methods = [m["name"] for m in methods_cfg["methods"]]

    with open(args.estimators_config) as f:
        estimators_cfg = yaml.safe_load(f)
    rppg_methods = list(estimators_cfg["rppg"].keys())

    profiles = load_profiles(args.profiles_config)

    # -- Load original predictions --
    log("Loading balanced (original) predictions…")
    bal = {
        "face_embedding": lmdb_to_dict(str(pred_dir / "face_embedding" / "balanced.lmdb")),
        "age_gender":     lmdb_to_dict(str(pred_dir / "age_gender"     / "balanced.lmdb")),
        "ethnicity":      lmdb_to_dict(str(pred_dir / "ethnicity"      / "balanced.lmdb")),
        "macro_exp":      lmdb_to_dict(str(pred_dir / "macro_exp"      / "balanced.lmdb")),
        "landmark":       lmdb_to_dict(str(pred_dir / "landmark"       / "balanced.lmdb")),
        "gaze":           lmdb_to_dict(str(pred_dir / "gaze"           / "balanced.lmdb")),
    }
    log(f"  face_embedding: {len(bal['face_embedding']):,} keys")

    log("Loading DFME original (for micro-expression)…")
    dfme_path = pred_dir / "micro_exp" / "dfme.lmdb"
    if dfme_path.exists():
        dfme = {"micro_exp": lmdb_to_dict(str(dfme_path))}
        log(f"  micro_exp: {len(dfme['micro_exp']):,} clips")
    else:
        dfme = None
        log("  micro_exp DFME LMDB not found — U2 will rely on gaze only.")

    log("Loading PURE original rPPG predictions…")
    pure_rppg = {}
    for rm in rppg_methods:
        p = pred_dir / "rppg" / f"pure_{rm}.npy"
        if p.exists():
            pure_rppg[rm] = np.load(str(p), allow_pickle=True).item()
    if pure_rppg:
        pure = {"rppg": pure_rppg}
        log(f"  rppg: {len(pure_rppg)} estimator(s), "
            f"{len(next(iter(pure_rppg.values())))} subjects")
    else:
        pure = None
        log("  rPPG .npy files not found — U3 will be None.")

    # -- Compute per-method scores --
    all_results = {}
    for method in methods:
        log(f"Computing HiFD for {method}…")
        r = compute_method(method, pred_dir, bal, dfme, pure, rppg_methods)
        components = r.pop("_components")

        # Compute one HiFD per profile
        for pname, pweights in profiles.items():
            r[f"HiFD_{pname}"] = hifd(
                P_bar=components["P"],
                Q=components["Q"],
                U1=components["U1"],
                U2=components["U2"],
                U3=components["U3"],
                weights=pweights,
            )

        all_results[method] = r
        p_str = f"{r['privacy']:.4f}" if r["privacy"] is not None else "N/A"
        q_str = f"{r['quality']:.4f}" if r["quality"] is not None else "N/A"
        u1_str = f"{r['U1']:.4f}" if r["U1"] is not None else "N/A"
        u2_str = f"{r['U2']:.4f}" if r["U2"] is not None else "N/A"
        u3_str = f"{r['U3']:.4f}" if r["U3"] is not None else "N/A"
        bal_key = "HiFD_Balanced"
        b_str = f"{r[bal_key]:.4f}" if r.get(bal_key) is not None else "N/A"
        log(f"  P={p_str}  Q={q_str}  U1={u1_str}  U2={u2_str}  U3={u3_str}  "
            f"HiFD(Bal)={b_str}")

    os.makedirs(str(results_dir), exist_ok=True)

    # -- Write scores.json --
    out_json = results_dir / "scores.json"
    with open(str(out_json), "w") as f:
        json.dump(all_results, f, indent=2)
    log(f"Saved {out_json}")

    # -- Write scores.csv --
    profile_cols = [f"HiFD_{pname}" for pname in profiles]
    cols = (
        ["method", "privacy", "lpips_mean", "niqe_mean", "quality",
         "age", "gender", "ethnicity", "macro_exp", "landmark", "U1",
         "gaze", "micro_exp", "U2",
         "bvp", "hr", "U3"]
        + profile_cols
    )
    out_csv = results_dir / "scores.csv"
    with open(str(out_csv), "w") as f:
        f.write(",".join(cols) + "\n")
        for m in methods:
            rv = all_results[m]
            row = [m] + [
                (f"{rv[c]:.4f}" if rv.get(c) is not None else "")
                for c in cols[1:]
            ]
            f.write(",".join(row) + "\n")
    log(f"Saved {out_csv}")

    # -- Write scores.tex (LaTeX booktabs table) --
    def fmt(v):
        return f"{v:.3f}" if v is not None else "--"

    tex_lines = []
    tex_lines.append(r"% Auto-generated by compute_hifd.py")
    tex_lines.append(r"\begin{tabular}{lcccccccccccccc}")
    tex_lines.append(r"\toprule")
    tex_lines.append(
        r"Method & $\bar{P}$ & $Q$ & Age & Gen. & Eth. & MExp & Lmk & $U_1$ "
        r"& Gaze & ME & BVP & HR & $U_3$ & HiFD$_{\text{Bal}}$ \\"
    )
    tex_lines.append(r"\midrule")
    for m in methods:
        rv = all_results[m]
        tex_lines.append(
            f"{m} & " + " & ".join([
                fmt(rv.get("privacy")), fmt(rv.get("quality")),
                fmt(rv.get("age")), fmt(rv.get("gender")),
                fmt(rv.get("ethnicity")), fmt(rv.get("macro_exp")),
                fmt(rv.get("landmark")), fmt(rv.get("U1")),
                fmt(rv.get("gaze")), fmt(rv.get("micro_exp")),
                fmt(rv.get("bvp")), fmt(rv.get("hr")), fmt(rv.get("U3")),
                fmt(rv.get("HiFD_Balanced")),
            ]) + r" \\"
        )
    tex_lines.append(r"\bottomrule")
    tex_lines.append(r"\end{tabular}")
    out_tex = results_dir / "scores.tex"
    with open(str(out_tex), "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    log(f"Saved {out_tex}")

    log("Done.")


if __name__ == "__main__":
    main()
