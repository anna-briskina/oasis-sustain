"""Validation of SuStaIn output against clinical variables the model never saw."""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def positional_variance(samples_sequence_s: np.ndarray, n_events: int) -> np.ndarray:
    """P[event, position] = fraction of MCMC samples placing `event` at `position`.

    samples_sequence_s: (n_events, n_samples) array of event indices per position.
    """
    seq = np.asarray(samples_sequence_s).astype(int)
    P = np.zeros((n_events, n_events))
    for pos in range(n_events):
        P[:, pos] = np.bincount(seq[pos], minlength=n_events)[:n_events] / seq.shape[1]
    return P


def subject_table(meta: pd.DataFrame, z: pd.DataFrame, res: dict) -> pd.DataFrame:
    df = meta.drop(columns=[c for c in ["gm_path"] if c in meta]).copy()
    df["stage"] = np.asarray(res["ml_stage"]).reshape(-1).astype(int)
    df["prob_stage"] = np.asarray(res["prob_ml_stage"]).reshape(-1)
    sub = np.asarray(res["ml_subtype"]).reshape(-1).astype(int)
    df["subtype"] = np.where(df["stage"] > 0, sub + 1, 0)  # 0 = no detectable abnormality
    df["prob_subtype"] = np.asarray(res["prob_ml_subtype"]).reshape(-1)
    df["z_composite"] = z.mean(axis=1).to_numpy()  # naive baseline: average atrophy
    return df


def _spearman(a, b):
    m = pd.notna(a) & pd.notna(b)
    if m.sum() < 5:
        return {"rho": None, "p": None, "n": int(m.sum())}
    r, p = stats.spearmanr(np.asarray(a)[m], np.asarray(b)[m])
    return {"rho": float(r), "p": float(p), "n": int(m.sum())}


def summarize(df: pd.DataFrame, features: list[str], samples_f: np.ndarray,
              certainty_threshold: float) -> dict:
    pat = df["is_patient"] == 1
    ctrl = ~pat
    out = {"n_subjects": int(len(df)), "n_controls": int(ctrl.sum()), "n_patients": int(pat.sum())}

    # 1. Does stage track clinical severity? (CDR/MMSE were never model inputs)
    out["stage_vs_mmse_all"] = _spearman(df["stage"], df["mmse"])
    out["stage_vs_mmse_patients"] = _spearman(df.loc[pat, "stage"], df.loc[pat, "mmse"])
    out["stage_vs_cdr_all"] = _spearman(df["stage"], df["cdr"])
    groups = [g["stage"].to_numpy() for _, g in df.groupby("cdr") if len(g) >= 3]
    if len(groups) >= 2:
        h, p = stats.kruskal(*groups)
        out["stage_by_cdr_kruskal"] = {"H": float(h), "p": float(p)}
    out["stage_by_cdr_median"] = {str(k): float(v) for k, v in df.groupby("cdr")["stage"].median().items()}

    # 2. Baseline: is a staged model better than "average atrophy"?
    out["composite_vs_mmse_patients"] = _spearman(df.loc[pat, "z_composite"], df.loc[pat, "mmse"])

    # 3. Controls should sit at stage 0 / low stages
    out["controls_stage0_frac"] = float((df.loc[ctrl, "stage"] == 0).mean())
    out["patients_stage0_frac"] = float((df.loc[pat, "stage"] == 0).mean())

    # 4. Confound check: after regression, stage should not follow age in controls
    out["stage_vs_age_controls"] = _spearman(df.loc[ctrl, "stage"], df.loc[ctrl, "age"])

    # 5. Subtypes: sizes, certainty, clinical profile
    staged = df[df["subtype"] > 0]
    out["subtype_fraction_mcmc"] = [float(x) for x in np.asarray(samples_f).mean(axis=1)]
    out["subtype_counts_staged"] = {int(k): int(v) for k, v in staged["subtype"].value_counts().sort_index().items()}
    out["subtype_certainty"] = {
        "median_prob": float(staged["prob_subtype"].median()) if len(staged) else None,
        f"frac_above_{certainty_threshold}": float((staged["prob_subtype"] >= certainty_threshold).mean()) if len(staged) else None,
    }
    # Who is in each subtype? A subtype made mostly of controls is likely noise, not disease.
    out["subtype_by_group"] = {
        int(s): {str(g): int(n) for g, n in grp["group"].value_counts().items()}
        for s, grp in staged.groupby("subtype")}
    pat_staged = staged[staged["is_patient"] == 1]
    out["subtype_profiles_patients_only"] = (
        pat_staged.groupby("subtype").agg(n=("stage", "size"), age=("age", "mean"), mmse=("mmse", "mean"),
                                          cdr=("cdr", "mean"), stage=("stage", "mean"),
                                          prob_subtype=("prob_subtype", "median"))
        .round(3).reset_index().to_dict(orient="records"))
    if pat_staged["subtype"].nunique() >= 2:
        gs = [g["mmse"].dropna().to_numpy() for _, g in pat_staged.groupby("subtype")]
        if all(len(g) >= 3 for g in gs):
            h, p = stats.kruskal(*gs)
            out["subtype_mmse_kruskal_patients"] = {"H": float(h), "p": float(p)}
    prof = staged.groupby("subtype").agg(n=("stage", "size"), age=("age", "mean"),
                                         mmse=("mmse", "mean"), stage=("stage", "mean"),
                                         male_frac=("sex_m", "mean"))
    out["subtype_profiles"] = prof.round(3).reset_index().to_dict(orient="records")
    if staged["subtype"].nunique() >= 2:
        for var in ["age", "mmse"]:
            gs = [g[var].dropna().to_numpy() for _, g in staged.groupby("subtype")]
            if all(len(g) >= 3 for g in gs):
                h, p = stats.kruskal(*gs)
                out[f"subtype_{var}_kruskal"] = {"H": float(h), "p": float(p)}
    return out
