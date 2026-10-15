"""OASIS-1 VBM download (via nilearn) and cohort selection."""
from __future__ import annotations

import numpy as np
import pandas as pd

NUMERIC_COLS = ["age", "educ", "ses", "mmse", "cdr", "etiv", "nwbv", "asf", "delay"]


def tidy_ext_vars(ext_vars) -> pd.DataFrame:
    """Normalise nilearn's ext_vars (DataFrame in new versions, recarray of bytes in old)."""
    df = ext_vars.copy() if isinstance(ext_vars, pd.DataFrame) else pd.DataFrame(ext_vars)
    df.columns = [str(c).strip().lower() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].map(lambda v: v.decode().strip() if isinstance(v, bytes) else v)
    for c in NUMERIC_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "mf" in df.columns:
        df["sex_m"] = df["mf"].astype(str).str.upper().str.startswith("M").astype(float)
    return df.reset_index(drop=True)


def fetch_oasis(data_dir, n_subjects=None) -> pd.DataFrame:
    """Download (or load cached) OASIS-1 DARTEL VBM maps. Returns one row per subject."""
    from nilearn import datasets

    ds = datasets.fetch_oasis_vbm(n_subjects=n_subjects, data_dir=str(data_dir))
    meta = tidy_ext_vars(ds.ext_vars)
    gm = list(ds.gray_matter_maps)
    if len(gm) != len(meta):
        raise RuntimeError(f"{len(gm)} GM maps but {len(meta)} metadata rows")
    meta["gm_path"] = gm
    return meta


def select_cohort(meta: pd.DataFrame, cohort_cfg: dict, covariates: list[str]) -> pd.DataFrame:
    """Older adults with CDR and all covariates present. Adds group labels."""
    keep = meta["age"] >= cohort_cfg["min_age"]
    if cohort_cfg.get("require_cdr", True):
        keep &= meta["cdr"].notna()
    for c in covariates:
        keep &= meta[c].notna()
    df = meta.loc[keep].copy().reset_index(drop=True)
    df["is_patient"] = (df["cdr"] > 0).astype(int)
    df["group"] = np.where(df["is_patient"] == 1, "CDR>0", "CDR=0")
    return df
