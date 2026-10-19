"""Sanity checks on the prepared data, before trusting any SuStaIn result.

    python diagnostics/diagnose.py --config configs/quick.yaml

1. Alignment: does every GM map filename contain the subject ID on the same metadata row?
2. Known-truth check: regional GM should correlate positively with nWBV (OASIS's own
   whole-brain volume measure) and negatively with age. If it does not, maps and
   metadata are mismatched or the extraction is wrong.
3. Signal: per-region effect size (CDR>0 vs CDR=0) and correlation with MMSE in patients.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oasis_sustain import config, data  # noqa: E402
import run  # noqa: E402


def cohens_d(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    s = np.sqrt(((len(a) - 1) * a.var(ddof=1) + (len(b) - 1) * b.var(ddof=1)) / (len(a) + len(b) - 2))
    return (a.mean() - b.mean()) / s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    cfg = config.load_config(ap.parse_args().config)
    dirs = config.run_dirs(cfg)
    f = run._files(cfg, dirs)

    print("=== 1. Alignment of GM maps and metadata ===")
    meta = data.fetch_oasis(dirs["data"], cfg["cohort"]["n_subjects"], cfg["cohort"]["dartel_version"])
    names = meta["gm_path"].map(lambda p: os.path.basename(str(p)))
    ok = np.array([str(i) in n for i, n in zip(meta["id"], names)])
    print(f"{ok.sum()}/{len(ok)} rows: filename contains the row's subject ID")
    print("First 3 pairs:")
    for i, n in list(zip(meta["id"], names))[:3]:
        print(f"  {i:<16} {n}")
    if not ok.all():
        print("MISMATCH on rows:", np.flatnonzero(~ok)[:10], "...")

    print("\n=== 2. Known-truth correlations (expect GM~nWBV positive, GM~age negative) ===")
    regional = pd.read_csv(f["regional"])
    cohort = pd.read_csv(f["subjects"])
    df = cohort.merge(regional, on="id", how="inner")
    regions = [c for c in regional.columns if c != "id"]
    rows = []
    for r in regions:
        pat, ctl = df[df.is_patient == 1], df[df.is_patient == 0]
        rows.append({
            "region": r,
            "rho_nWBV": stats.spearmanr(df[r], df["nwbv"]).statistic if "nwbv" in df else np.nan,
            "rho_age": stats.spearmanr(df[r], df["age"]).statistic,
            "d_CDR>0_vs_0": cohens_d(pat[r], ctl[r]),
            "rho_MMSE_pat": stats.spearmanr(pat[r], pat["mmse"], nan_policy="omit").statistic,
        })
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    if "nwbv" in df:
        nw = stats.spearmanr(df["nwbv"], df["mmse"], nan_policy="omit")
        nwp = df[df.is_patient == 1]
        nwp_r = stats.spearmanr(nwp["nwbv"], nwp["mmse"], nan_policy="omit")
        print(f"\nReference (OASIS's own nWBV, not my pipeline): rho(nWBV, MMSE) "
              f"all = {nw.statistic:.2f}, patients only = {nwp_r.statistic:.2f}")

    print("\n=== 3. Model inputs: mean z by group (atrophy is positive) ===")
    d = run.load_npz(f["inputs"])
    z = pd.DataFrame(d["z"], columns=[str(x) for x in d["labels"]])
    z["group"] = np.where(d["is_patient"] == 1, "CDR>0", "CDR=0")
    print(z.groupby("group").mean().T.round(2).to_string())
    print("\n(Section 2 uses RAW values. Modulated GM scales with head size, so raw correlations "
          "with nWBV, which is already divided by head size, are confounded. Sections 4-5 adjust.)")

    print("\n=== 4. Are the maps modulated? (total GM should track eTIV strongly) ===")
    import nibabel as nib
    m2 = df[["id", "age", "etiv", "nwbv", "sex_m"]].merge(meta[["id", "gm_path"]], on="id")
    tot = []
    for p in m2["gm_path"]:
        img = nib.load(p)
        vox_ml = float(np.prod(img.header.get_zooms()[:3])) / 1000.0
        tot.append(float(np.asanyarray(img.dataobj, dtype=np.float32).sum()) * vox_ml)
    m2["total_gm_ml"] = tot
    print(f"total GM: mean {np.mean(tot):.0f} mL;  rho(total GM, eTIV) = "
          f"{stats.spearmanr(m2.total_gm_ml, m2.etiv).statistic:.2f}")
    frac = m2.total_gm_ml / m2.etiv
    print(f"rho(total GM / eTIV, age) = {stats.spearmanr(frac, m2.age).statistic:.2f}   (expect clearly negative)")
    print(f"rho(total GM / eTIV, nWBV) = {stats.spearmanr(frac, m2.nwbv).statistic:.2f}  (expect clearly positive)")

    print("\n=== 5. Adjusted signal: pipeline z (atrophy +, adjusted for age/sex/eTIV on controls) ===")
    from oasis_sustain import zscore as zs
    covs = cfg["zscore"]["covariates"]
    ctrl = df["is_patient"].to_numpy() == 0
    zr, _ = zs.residual_zscores(df[regions], df, ctrl, covs)
    zn, _ = zs.residual_zscores(df[["nwbv"]], df, ctrl, covs)  # same adjustment applied to nWBV
    zr["nWBV (reference)"] = zn["nwbv"].to_numpy()
    pat = df["is_patient"].to_numpy() == 1
    rows = []
    for c in zr.columns:
        rows.append({
            "feature": c,
            "mean_z_CDR>0": zr.loc[pat, c].mean(),
            "rho_MMSE_all": stats.spearmanr(zr[c], df["mmse"], nan_policy="omit").statistic,
            "rho_MMSE_pat": stats.spearmanr(zr.loc[pat, c], df.loc[pat, "mmse"], nan_policy="omit").statistic,
            "rho_CDR_all": stats.spearmanr(zr[c], df["cdr"]).statistic,
        })
    print(pd.DataFrame(rows).round(2).to_string(index=False))
    print("\nWhat good looks like: section 4 rho(total GM, eTIV) high (> 0.6) and the two ratio "
          "checks with the expected signs. Section 5: mean_z_CDR>0 clearly positive for medial "
          "temporal regions; rho_MMSE negative, rho_CDR positive; regions in the same ballpark "
          "as the nWBV reference row.")


if __name__ == "__main__":
    main()
