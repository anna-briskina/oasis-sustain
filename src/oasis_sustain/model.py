"""Thin, version-tolerant wrappers around pySuStaIn's ZscoreSustain."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold


def make_sustain(z, Z_vals, Z_max, labels, scfg: dict, out_dir, dataset_name: str, seed: int):
    from pySuStaIn import ZscoreSustain

    args = (np.asarray(z, float), np.asarray(Z_vals, float), np.asarray(Z_max, float),
            list(labels), int(scfg["n_startpoints"]), int(scfg["n_subtypes_max"]),
            int(scfg["n_mcmc"]), str(out_dir), dataset_name, bool(scfg["parallel"]))
    try:
        return ZscoreSustain(*args, seed=seed)
    except TypeError:  # older pySuStaIn without a seed argument
        np.random.seed(seed)
        return ZscoreSustain(*args)


def fit(sustain):
    try:
        return sustain.run_sustain_algorithm(plot=False)
    except TypeError:
        return sustain.run_sustain_algorithm()


def cv_folds(is_patient, n_folds: int, seed: int):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    dummy = np.zeros(len(is_patient))
    # 1-D object array of int index arrays. Built element-wise: np.array(list, dtype=object)
    # would silently make a 2-D object array (non-integer indices) when folds are equal-sized.
    tests = [test.astype(int) for _, test in skf.split(dummy, is_patient)]
    folds = np.empty(len(tests), dtype=object)
    for i, t in enumerate(tests):
        folds[i] = t
    return folds


def cross_validate(sustain, test_idxs):
    try:
        cvic, loglike = sustain.cross_validate_sustain_model(test_idxs, plot=False)
    except TypeError:
        cvic, loglike = sustain.cross_validate_sustain_model(test_idxs)
    return np.asarray(cvic, float), np.asarray(loglike, float)


def assign_from_samples(sustain, z, res: dict, n_samples: int = 1000) -> dict:
    """Subtype and stage per person, recomputed from the SAME posterior samples as the figures.

    pySuStaIn's pickled ml_subtype is not guaranteed to share an ordering with its pickled
    samples_sequence (we observed mismatches), so we never use it. Recomputing from
    samples_sequence/samples_f makes people's labels and the plotted sequences one source.
    """
    out = sustain.subtype_and_stage_individuals_newData(
        np.asarray(z, float), np.asarray(res["samples_sequence"]), np.asarray(res["samples_f"]), n_samples)
    ml_subtype, prob_ml_subtype, ml_stage, prob_ml_stage = out[:4]
    return {**res, "ml_subtype": ml_subtype, "prob_ml_subtype": prob_ml_subtype,
            "ml_stage": ml_stage, "prob_ml_stage": prob_ml_stage}


def pickle_path(sustain_dir, dataset_name: str, n_subtypes: int) -> Path:
    # pySuStaIn names results by 0-based subtype index: subtype0 = 1-subtype model
    return Path(sustain_dir) / "pickle_files" / f"{dataset_name}_subtype{n_subtypes - 1}.pickle"


def load_result(sustain_dir, dataset_name: str, n_subtypes: int) -> dict:
    p = pickle_path(sustain_dir, dataset_name, n_subtypes)
    if not p.exists():
        found = sorted(x.name for x in (Path(sustain_dir) / "pickle_files").glob("*.pickle"))
        raise FileNotFoundError(f"{p} missing. Found: {found}")
    with open(p, "rb") as f:
        res = pickle.load(f)
    need = ["samples_sequence", "samples_f", "ml_subtype", "prob_ml_subtype",
            "ml_stage", "prob_ml_stage"]
    missing = [k for k in need if k not in res]
    if missing:
        raise KeyError(f"{p} lacks {missing}; keys are {sorted(res)}")
    return res
