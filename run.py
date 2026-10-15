"""OASIS-1 SuStaIn pipeline.

    python run.py --config configs/quick.yaml                 # all steps
    python run.py --config configs/full.yaml --steps fit,cv   # selected steps

Steps: prepare -> fit -> cv -> analyze. Each step reads what the previous one wrote,
so you can rerun `analyze` without refitting.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from oasis_sustain import analysis, config, data, features, model, plots, zscore  # noqa: E402

STEPS = ["prepare", "fit", "cv", "analyze"]


def load_npz(path) -> dict:
    """Read an .npz fully and close it (an open NpzFile locks the file on Windows)."""
    with np.load(path, allow_pickle=True) as d:
        return {k: d[k] for k in d.files}


def _files(cfg, dirs):
    return {
        "regional": dirs["processed"] / f"{cfg['name']}_regional_gm.csv",
        "inputs": dirs["out"] / "sustain_inputs.npz",
        "subjects": dirs["out"] / "cohort.csv",
        "zfit": dirs["out"] / "zscore_fit.json",
        "cv": dirs["out"] / "cv.npz",
    }


def step_prepare(cfg, dirs, f):
    zc = cfg["zscore"]
    print("Fetching OASIS-1 VBM via nilearn (first run downloads ~1 GB)...")
    meta = data.fetch_oasis(dirs["data"], cfg["cohort"]["n_subjects"], cfg["cohort"]["dartel_version"])
    cohort = data.select_cohort(meta, cfg["cohort"], zc["covariates"])
    print(f"Cohort: {len(cohort)} subjects ({cohort.is_patient.sum()} CDR>0, "
          f"{(cohort.is_patient == 0).sum()} CDR=0), age >= {cfg['cohort']['min_age']}")

    if f["regional"].exists():
        regional = pd.read_csv(f["regional"])
        if list(regional.columns[1:]) != list(cfg["regions"]) or len(regional) != len(cohort) \
                or not (regional["id"].to_numpy() == cohort["id"].to_numpy()).all():
            print("Cached regional values do not match config; re-extracting.")
            regional = None
    else:
        regional = None
    if regional is None:
        atl = features.load_atlases(cfg["atlases"], dirs["data"], cohort["gm_path"].iloc[0])
        masks = features.build_region_masks(atl, cfg["regions"])
        print("Voxels per region:", {k: int(v.sum()) for k, v in masks.items()})
        regional = features.extract_regional_means(cohort["gm_path"].tolist(), masks)
        regional.insert(0, "id", cohort["id"].to_numpy())
        regional.to_csv(f["regional"], index=False)

    X = regional.drop(columns="id")
    if not np.isfinite(X.to_numpy()).all():
        bad = cohort.loc[~np.isfinite(X.to_numpy()).all(1), "id"].tolist()
        raise ValueError(f"Non-finite regional GM values for subjects {bad}")

    # QC: subjects with implausible z in any region are almost always processing failures.
    # Drop them, refit the control model without them, and report who was dropped.
    excluded = []
    while True:
        z, fit = zscore.residual_zscores(X, cohort, cohort["is_patient"].to_numpy() == 0, zc["covariates"])
        ctrl = cohort["is_patient"].to_numpy() == 0
        out = zscore.flag_outliers(z, ctrl, zc["outlier_abs_z"])
        if not out.any():
            break
        rz = zscore.robust_z(z, ctrl)
        for i in np.flatnonzero(out):
            worst = rz.iloc[i].abs().idxmax()
            excluded.append({"id": cohort.loc[i, "id"], "group": cohort.loc[i, "group"],
                             "region": worst, "robust_z": round(float(rz.iloc[i][worst]), 1)})
        keep = ~out
        cohort, X = cohort.loc[keep].reset_index(drop=True), X.loc[keep].reset_index(drop=True)
    n_start = len(cohort) + len(excluded)
    if len(excluded) > zc["max_exclude_frac"] * n_start:
        raise RuntimeError(
            f"QC would exclude {len(excluded)}/{n_start} subjects. That is not a few bad scans; "
            f"it means disease effects exceed the threshold or the control spread is tiny. "
            f"Inspect the data, or raise zscore.outlier_abs_z. First flagged: {excluded[:5]}")
    if excluded:
        print(f"QC: excluded {len(excluded)} subject(s) with |z| > {zc['outlier_abs_z']}:")
        print(pd.DataFrame(excluded).to_string(index=False))
    pd.DataFrame(excluded, columns=["id", "group", "region", "robust_z"]).to_csv(dirs["out"] / "qc_excluded.csv", index=False)
    print("z range per region:\n" + z.agg(["min", "max"]).T.round(2).to_string())

    z_raw = z
    Z_vals = zscore.choose_z_vals(z.to_numpy(), cohort["is_patient"].to_numpy(), zc["levels"],
                                  zc["min_patient_frac"])
    Z_max = zscore.choose_z_max(z.to_numpy(), Z_vals, zc["z_max_percentile"])
    # Clip into the range the model represents (abnormality 0..Z_max). Values far below 0
    # ("healthier than controls") or above Z_max add likelihood underflow, not information.
    z = zscore.clip_to_model(z, zc["clip_min"], Z_max)
    print(f"Cohort after QC: {len(cohort)} ({cohort.is_patient.sum()} CDR>0, {(cohort.is_patient == 0).sum()} CDR=0)")
    events = zscore.event_table(Z_vals, list(X.columns))
    print(f"{len(X.columns)} biomarkers -> {len(events)} events")
    print(pd.DataFrame(Z_vals, index=X.columns, columns=[f"L{i+1}" for i in range(Z_vals.shape[1])]))

    np.savez(f["inputs"], z=z.to_numpy(), Z_vals=Z_vals, Z_max=Z_max,
             labels=np.array(X.columns), is_patient=cohort["is_patient"].to_numpy())
    cohort.drop(columns="gm_path").to_csv(f["subjects"], index=False)
    f["zfit"].write_text(json.dumps(fit, indent=2))
    plots.plot_zscores(z_raw, cohort["group"], dirs["figures"])


def _sustain_from_inputs(cfg, dirs, f):
    if not f["inputs"].exists():
        sys.exit("Run the 'prepare' step first.")
    d = load_npz(f["inputs"])
    s = model.make_sustain(d["z"], d["Z_vals"], d["Z_max"], [str(x) for x in d["labels"]],
                           cfg["sustain"], dirs["sustain"], cfg["name"], cfg["seed"])
    return s, d


def step_fit(cfg, dirs, f):
    s, _ = _sustain_from_inputs(cfg, dirs, f)
    print(f"Fitting SuStaIn with 1..{cfg['sustain']['n_subtypes_max']} subtypes "
          f"({cfg['sustain']['n_mcmc']:,} MCMC iterations)...")
    model.fit(s)


def step_cv(cfg, dirs, f):
    s, d = _sustain_from_inputs(cfg, dirs, f)
    folds = model.cv_folds(d["is_patient"], cfg["cv"]["n_folds"], cfg["seed"])
    print(f"{cfg['cv']['n_folds']}-fold cross-validation (refits every fold)...")
    cvic, loglike = model.cross_validate(s, folds)
    np.savez(f["cv"], cvic=cvic, loglike=loglike)
    print("CVIC by number of subtypes:", np.round(cvic, 1))
    plots.plot_cvic(cvic, dirs["figures"])


def step_analyze(cfg, dirs, f):
    d = load_npz(f["inputs"])
    labels = [str(x) for x in d["labels"]]
    n = cfg["analysis"]["n_subtypes"]
    if n == "auto":
        if not f["cv"].exists():
            sys.exit("n_subtypes is 'auto' but no CV results; run 'cv' or set n_subtypes.")
        n = int(np.argmin(load_npz(f["cv"])["cvic"]) + 1)
        print(f"Selected {n} subtype(s) by lowest CVIC")
    res = model.load_result(dirs["sustain"], cfg["name"], int(n))
    cohort = pd.read_csv(f["subjects"])
    z = pd.DataFrame(d["z"], columns=labels)
    events = zscore.event_table(d["Z_vals"], labels)

    df = analysis.subject_table(cohort, z, res)
    summary = analysis.summarize(df, labels, res["samples_f"], cfg["analysis"]["certainty_threshold"])
    summary["n_subtypes"] = int(n)
    df.to_csv(dirs["out"] / f"subjects_{n}subtypes.csv", index=False)
    (dirs["out"] / f"summary_{n}subtypes.json").write_text(json.dumps(summary, indent=2))

    plots.plot_pvd(res, events, dirs["figures"], labels)
    plots.plot_stage_vs_clinical(df, dirs["figures"])
    plots.plot_subtype_certainty(df, dirs["figures"])
    print(json.dumps({k: summary[k] for k in ["stage_vs_mmse_patients", "composite_vs_mmse_patients",
                                             "controls_stage0_frac", "subtype_counts_staged"]}, indent=2))
    print(f"Results in {dirs['out']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", default=",".join(STEPS))
    a = ap.parse_args()
    cfg = config.load_config(a.config)
    dirs = config.run_dirs(cfg)
    f = _files(cfg, dirs)
    for step in a.steps.split(","):
        step = step.strip()
        if step not in STEPS:
            sys.exit(f"Unknown step '{step}'. Choose from {STEPS}")
        print(f"\n=== {step} ===")
        globals()[f"step_{step}"](cfg, dirs, f)


if __name__ == "__main__":
    main()
