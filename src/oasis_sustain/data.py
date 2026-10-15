"""OASIS-1 VBM download (via nilearn) and cohort selection."""
from __future__ import annotations

import os
import re

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


def fetch_oasis(data_dir, n_subjects=None, dartel: bool = True) -> pd.DataFrame:
    """Download (or load cached) OASIS-1 VBM maps (DARTEL or standard SPM8). One row per subject."""
    from nilearn import datasets

    ds = datasets.fetch_oasis_vbm(n_subjects=n_subjects, dartel_version=dartel, data_dir=str(data_dir))
    return align_maps(tidy_ext_vars(ds.ext_vars), list(ds.gray_matter_maps))


_ID_RE = re.compile(r"OAS1_\d{4}_MR\d")
_GM_RE = re.compile(r"c1OAS1_\d{4}_MR\d")  # e.g. mwc1OAS1_... (SPM8) or mwrc1OAS1_... (DARTEL)


def align_maps(meta: pd.DataFrame, paths: list[str]) -> pd.DataFrame:
    """Attach each GM map to its metadata row by the subject ID in its filename.

    Never pairs by position: the two lists can differ in length (e.g. 415 maps vs 414 rows
    in the SPM8 release), and a single gap would shift every later subject.
    """
    # SPM tissue classes: c1 = grey matter, c2 = white matter, c3 = CSF. nilearn's SPM8 list
    # includes at least one c2 file, so keep only c1 maps and report anything else.
    not_gm = [p for p in paths if not _GM_RE.search(os.path.basename(str(p)))]
    if not_gm:
        print(f"Dropped {len(not_gm)} non-grey-matter file(s) from nilearn's list: "
              f"{[os.path.basename(str(p)) for p in not_gm[:5]]}")
    paths = [p for p in paths if p not in not_gm]
    rows = []
    for p in paths:
        m = _ID_RE.search(os.path.basename(str(p)))
        if m is None:
            raise ValueError(f"No OASIS ID in filename: {p}")
        rows.append({"id": m.group(0), "gm_path": str(p)})
    files = pd.DataFrame(rows)
    # The same path listed twice is harmless: keep one copy.
    norm = files["gm_path"].map(lambda p: os.path.normcase(os.path.abspath(p)))
    files = files.loc[~norm.duplicated()].reset_index(drop=True)
    dup_ids = files["id"][files["id"].duplicated()].unique().tolist()
    if dup_ids:
        detail = {i: files.loc[files["id"] == i, "gm_path"].tolist() for i in dup_ids[:3]}
        raise ValueError("Different map files for the same subject; cannot tell which is right:\n"
                         + "\n".join(f"  {i}:\n    " + "\n    ".join(ps) for i, ps in detail.items()))
    meta = meta.drop(columns=[c for c in ["gm_path"] if c in meta])
    out = meta.merge(files, on="id", how="inner", validate="one_to_one")
    no_map = sorted(set(meta["id"]) - set(files["id"]))
    no_meta = sorted(set(files["id"]) - set(meta["id"]))
    if no_map or no_meta:
        print(f"Alignment: {len(out)} matched; dropped {len(no_meta)} map(s) without metadata "
              f"{no_meta[:5]} and {len(no_map)} metadata row(s) without a map {no_map[:5]}")
    return out.reset_index(drop=True)


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
