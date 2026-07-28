"""Recompute HiFD from the mini-UtilFace sub-scores. Requires only numpy.

    python verify.py

Everything below is recomputed from subscores.npz. Nothing is read back from
published_scores.csv except for the final comparison column.
"""

import csv
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# HiFD aggregation, transcribed from the paper (Sec. 3.2, App. B.2)
PROFILES = {
    "PrivacyFirst": {"P": 0.5,  "Q": 0.125, "U1": 0.125, "U2": 0.125, "U3": 0.125},
    "Balanced":     {"P": 0.2,  "Q": 0.2,   "U1": 0.2,   "U2": 0.2,   "U3": 0.2},
    "Clinical":     {"P": 0.15, "Q": 0.15,  "U1": 0.15,  "U2": 0.15,  "U3": 0.4},
}
L1 = ["age", "gender", "ethnicity", "macro_exp", "landmark"]


def harmonic(scores, weights, eps=1e-6):
    """Weighted harmonic mean. This is what gives HiFD its vetoing property:
    one near-zero component drags the composite down regardless of the rest."""
    num = sum(weights.values())
    den = sum(weights[k] / (scores[k] + eps) for k in weights)
    return num / den


def resolve(profile, scores):
    present = {k: profile[k] for k in profile if scores.get(k) is not None}
    tot = sum(present.values())
    return {k: v / tot for k, v in present.items()}


def components(z, video, mi, sel=None):
    """Per-method component scores. sel optionally restricts to a subset of
    image indices, which is what the bootstrap below resamples."""
    g = lambda f: z[f][mi] if sel is None else z[f][mi][sel]
    with np.errstate(invalid="ignore"):
        u1 = float(np.nanmean([np.nanmean(g(f)) for f in L1]))
        c = {"P": float(np.nanmean(g("privacy"))),
             "Q": float(np.nanmean(g("quality"))),
             "U1": u1,
             "U2": float(np.nanmean([np.nanmean(g("gaze")), video["U2_micro_exp"]])),
             "U3": video["U3"]}
    return c


def main():
    z = np.load(os.path.join(HERE, "subscores.npz"), allow_pickle=False)
    methods = [str(m) for m in z["methods"]]
    keys = [str(k) for k in z["keys"]]
    video = json.load(open(os.path.join(HERE, "video_axes.json")))
    pub = {r["method"]: r for r in
           csv.DictReader(open(os.path.join(HERE, "published_scores.csv")))}
    print(f"{len(keys)} images x {len(methods)} methods\n")

    print(f"{'method':12s} {'P':>7s} {'Q':>7s} {'U1':>7s} {'U2':>7s} {'U3':>7s}"
          f" {'HiFD':>8s} {'published':>10s} {'diff':>7s}")
    rows = {}
    for mi, m in enumerate(methods):
        c = components(z, video[m], mi)
        w = resolve(PROFILES["Balanced"], c)
        h = harmonic({k: c[k] for k in w}, w)
        rows[m] = (c, h)
        p = float(pub[m]["HiFD_Balanced"])
        print(f"{m:12s} {c['P']:7.4f} {c['Q']:7.4f} {c['U1']:7.4f} {c['U2']:7.4f}"
              f" {c['U3']:7.4f} {h:8.4f} {p:10.4f} {h-p:+7.4f}")
    print("\nThe residual column is subset sampling: these 2,000 images are a"
          "\nstratified 2% sample of the 99,365 scored images, so the means differ"
          "\nslightly from the full-benchmark table. The RANKING is what to check.")

    # ---- ranking agreement with the published table
    def rank(d):
        o = sorted(d, key=lambda k: -d[k])
        return {k: i for i, k in enumerate(o)}
    r_mini = rank({m: rows[m][1] for m in methods})
    r_pub = rank({m: float(pub[m]["HiFD_Balanced"]) for m in methods})
    a = np.array([r_mini[m] for m in methods], float)
    b = np.array([r_pub[m] for m in methods], float)
    rho = float(np.corrcoef(a, b)[0, 1])
    moved = [m for m in methods if r_mini[m] != r_pub[m]]
    print(f"\nSpearman(mini ranking, published ranking) = {rho:.3f}")
    print(f"methods whose rank changes on the 2% subset: {moved or 'none'}")
    top3 = sorted(methods, key=lambda m: -rows[m][1])[:3]
    print(f"top-3 on the mini set: {top3}")

    # ---- all three profiles
    print(f"\n{'method':12s} " + " ".join(f"{p:>14s}" for p in PROFILES))
    for m in methods:
        c = rows[m][0]
        vals = []
        for pname, prof in PROFILES.items():
            w = resolve(prof, c)
            vals.append(harmonic({k: c[k] for k in w}, w))
        print(f"{m:12s} " + " ".join(f"{v:14.4f}" for v in vals))

    # ---- weight sweep: the reviewer-requested check that the leader is stable
    print(f"\nw_P sweep, remaining weight split equally:")
    sweep = [0.2, 0.3, 0.4, 0.5, 0.6]
    print(f"{'method':12s} " + " ".join(f"{w:>7.1f}" for w in sweep))
    for m in methods:
        c = rows[m][0]
        out = []
        for wp in sweep:
            rest = (1.0 - wp) / 4.0
            prof = {"P": wp, "Q": rest, "U1": rest, "U2": rest, "U3": rest}
            w = resolve(prof, c)
            out.append(harmonic({k: c[k] for k in w}, w))
        print(f"{m:12s} " + " ".join(f"{v:7.4f}" for v in out))

    # ---- identity-level bootstrap, resampling identity blocks not images
    ident = [r["identity_label"] for r in
             csv.DictReader(open(os.path.join(HERE, "keys.csv")))]
    uniq = sorted(set(ident))
    idx_by_id = {u: np.where(np.array(ident) == u)[0] for u in uniq}
    rng = np.random.default_rng(0)
    B = 500
    boot = {m: [] for m in methods}
    for _ in range(B):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([idx_by_id[uniq[i]] for i in pick])
        for mi, m in enumerate(methods):
            c = components(z, video[m], mi, sel)
            w = resolve(PROFILES["Balanced"], c)
            boot[m].append(harmonic({k: c[k] for k in w}, w))
    print(f"\nidentity-level bootstrap over {len(uniq)} identities, B={B}:")
    print(f"{'method':12s} {'HiFD':>8s} {'95% CI':>18s} {'top-1 share':>12s}")
    stacked = np.array([boot[m] for m in methods])
    top1 = stacked.argmax(axis=0)
    for mi, m in enumerate(methods):
        v = np.array(boot[m])
        print(f"{m:12s} {rows[m][1]:8.4f} "
              f"[{np.percentile(v,2.5):.4f},{np.percentile(v,97.5):.4f}]"
              f" {100.0*(top1==mi).mean():11.1f}%")


if __name__ == "__main__":
    main()
