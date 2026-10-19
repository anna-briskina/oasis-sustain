"""Covariate-adjusted z-scores relative to controls, and SuStaIn z-level selection."""
from __future__ import annotations

import numpy as np
import pandas as pd


def residual_zscores(X: pd.DataFrame, meta: pd.DataFrame, control: np.ndarray,
                     covariates: list[str], atrophy_sign: float = -1.0):
    """Regress covariates out using controls only, then z-score against control residuals.

    atrophy_sign=-1 flips the sign so that LOWER grey matter gives POSITIVE z
    (SuStaIn's z-score model assumes abnormality increases with z).
    Returns (z DataFrame, fit dict for reproducibility/applying to new data).
    """
    control = np.asarray(control, bool)
    C = np.column_stack([np.ones(len(meta))] + [meta[c].to_numpy(float) for c in covariates])
    Y = X.to_numpy(float)
    beta, *_ = np.linalg.lstsq(C[control], Y[control], rcond=None)
    resid = Y - C @ beta
    dof = control.sum() - C.shape[1]
    if dof < 5:
        raise ValueError(f"Too few controls ({control.sum()}) for {C.shape[1]} parameters")
    sd = np.sqrt((resid[control] ** 2).sum(0) / dof)
    z = atrophy_sign * resid / sd
    fit = {"covariates": covariates, "beta": beta.tolist(), "sd": sd.tolist(),
           "features": list(X.columns), "n_controls": int(control.sum())}
    return pd.DataFrame(z, columns=X.columns, index=X.index), fit


def robust_z(z: pd.DataFrame, control) -> pd.DataFrame:
    """Re-express z with control median/MAD, which one corrupted control cannot inflate."""
    c = z[np.asarray(control, bool)]
    med = c.median()
    mad = 1.4826 * (c - med).abs().median()
    return (z - med) / mad.replace(0, np.nan)


def flag_outliers(z: pd.DataFrame, control, abs_threshold: float) -> np.ndarray:
    """Subjects with any |robust z| above threshold (likely QC failures, not disease)."""
    return (robust_z(z, control).abs() > abs_threshold).any(axis=1).to_numpy()


def clip_to_model(z: pd.DataFrame, clip_min: float, Z_max: np.ndarray) -> pd.DataFrame:
    """Clip each biomarker to [clip_min, Z_max] so no single value dominates the likelihood."""
    return z.clip(lower=clip_min, upper=pd.Series(np.asarray(Z_max, float), index=z.columns), axis=1)


def choose_z_vals(z: np.ndarray, patient: np.ndarray, levels: list[float],
                  min_frac: float) -> np.ndarray:
    """Per biomarker, keep consecutive z levels that enough patients actually reach.

    The first level is always kept so every biomarker contributes at least one event.
    Returns Z_vals of shape (n_biomarkers, len(levels)), zero-padded.
    """
    zp = np.asarray(z)[np.asarray(patient, bool)]
    Z = np.zeros((zp.shape[1], len(levels)))
    for b in range(zp.shape[1]):
        Z[b, 0] = levels[0]
        for j, lv in enumerate(levels[1:], start=1):
            if (zp[:, b] >= lv).mean() >= min_frac:
                Z[b, j] = lv
            else:
                break
    return Z


def choose_z_max(z: np.ndarray, Z_vals: np.ndarray, percentile: float) -> np.ndarray:
    top = np.percentile(np.asarray(z), percentile, axis=0)
    return np.maximum(Z_vals.max(1) + 1, np.ceil(top))


def event_table(Z_vals: np.ndarray, labels: list[str]) -> pd.DataFrame:
    """Events in pySuStaIn's ZscoreSustain order: z level outer loop, biomarker inner."""
    n_bio, n_z = Z_vals.shape
    ix = np.tile(np.arange(n_bio), n_z)          # biomarker index, level-major
    zs = Z_vals.T.flatten()
    keep = np.nonzero(zs)[0]
    return pd.DataFrame({"biomarker_index": ix[keep], "z": zs[keep],
                         "biomarker": [labels[i] for i in ix[keep]],
                         "event": [f"{labels[i]} z={zs[k]:g}" for i, k in zip(ix[keep], keep)]})
