"""Config loading with defaults, so a YAML file only needs to override what it changes."""
from __future__ import annotations

import copy
from pathlib import Path

import yaml

DEFAULTS: dict = {
    "name": "oasis1_sustain",
    "seed": 42,
    "paths": {
        "data_dir": "data/raw",          # nilearn download cache
        "processed_dir": "data/processed",
        "output_dir": "outputs",
    },
    "cohort": {
        "min_age": 60,                    # OASIS-1 subjects < 60 have no CDR
        "require_cdr": True,
        "n_subjects": None,               # None = all subjects nilearn provides
        "dartel_version": True,           # False = nilearn's standard SPM8-normalised maps
    },
    "atlases": ["cort-maxprob-thr25-2mm", "sub-maxprob-thr25-2mm"],
    # feature name -> Harvard-Oxford label names (union of voxels, both hemispheres)
    "regions": {},
    "zscore": {
        "covariates": ["age", "sex_m", "etiv"],
        "levels": [1, 2, 3],
        "min_patient_frac": 0.10,         # keep a z level only if >= this fraction of patients reach it
        "z_max_percentile": 99,
        "homogeneity_robust_z": 4,        # primary QC: map-to-mean correlation this far below median
        "outlier_abs_z": 15,              # safety net: |robust z| beyond this in any region
        "max_exclude_frac": 0.05,         # abort if QC would drop more than this fraction
        "clip_min": -3,                  # lower clip; upper clip is each biomarker's Z_max
    },
    "sustain": {
        "n_subtypes_max": 3,
        "n_startpoints": 25,
        "n_mcmc": 100000,
        "parallel": False,
    },
    "cv": {"n_folds": 5},
    "analysis": {
        "n_subtypes": "auto",             # int, or "auto" = smallest CVIC
        "certainty_threshold": 0.75,
    },
}


def _merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict) and k != "regions":
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path) -> dict:
    with open(path) as f:
        user = yaml.safe_load(f) or {}
    cfg = _merge(DEFAULTS, user)
    if not cfg["regions"]:
        raise ValueError(f"{path}: 'regions' is empty; define at least one feature.")
    return cfg


def run_dirs(cfg: dict) -> dict[str, Path]:
    p = cfg["paths"]
    out = Path(p["output_dir"]) / cfg["name"]
    dirs = {
        "data": Path(p["data_dir"]),
        "processed": Path(p["processed_dir"]),
        "out": out,
        "sustain": out / "sustain",
        "figures": out / "figures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
