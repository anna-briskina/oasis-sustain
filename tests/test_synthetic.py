"""Tests that run without nilearn, nibabel or pySuStaIn (synthetic data only).

    python tests/test_synthetic.py      # or: pytest tests/
"""
from __future__ import annotations

import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from oasis_sustain import analysis, config, data, model, zscore  # noqa: E402

RNG = np.random.default_rng(0)


def _fake_cohort(n_ctrl=90, n_pat=90):
    n = n_ctrl + n_pat
    meta = pd.DataFrame({
        "id": [f"OAS1_{i:04d}" for i in range(n)],
        "age": RNG.uniform(60, 90, n),
        "sex_m": RNG.integers(0, 2, n).astype(float),
        "etiv": RNG.normal(1500, 150, n),
        "cdr": np.r_[np.zeros(n_ctrl), RNG.choice([0.5, 1, 2], n_pat)],
    })
    meta["is_patient"] = (meta.cdr > 0).astype(int)
    meta["group"] = np.where(meta.is_patient == 1, "CDR>0", "CDR=0")
    meta["mmse"] = 29 - 4 * meta.cdr + RNG.normal(0, 1, n)
    return meta


def test_tidy_ext_vars_handles_bytes_recarray():
    rec = np.array([(b"OAS1_0001", b"F", 74.0, b"N/A"), (b"OAS1_0002", b"M", 55.0, b"0.5")],
                   dtype=[("ID", "S9"), ("MF", "S1"), ("Age", "f8"), ("CDR", "S3")])
    df = data.tidy_ext_vars(rec)
    assert list(df["id"]) == ["OAS1_0001", "OAS1_0002"]
    assert np.isnan(df.loc[0, "cdr"]) and df.loc[1, "cdr"] == 0.5
    assert list(df["sex_m"]) == [0.0, 1.0]


def test_align_maps_by_id_not_position():
    meta = pd.DataFrame({"id": ["OAS1_0001_MR1", "OAS1_0002_MR1", "OAS1_0004_MR1"], "age": [70, 80, 90]})
    paths = ["/d/mwc1OAS1_0004_MR1_mpr.nii.gz", "/d/mwc1OAS1_0001_MR1_mpr.nii.gz",
             "/d/mwc1OAS1_0003_MR1_mpr.nii.gz", "/d/mwc1OAS1_0002_MR1_mpr.nii.gz"]
    out = data.align_maps(meta, paths)
    assert len(out) == 3
    for _, r in out.iterrows():
        assert r["id"] in r["gm_path"]
    assert out.set_index("id").loc["OAS1_0004_MR1", "age"] == 90


def test_align_maps_drops_white_matter_files():
    meta = pd.DataFrame({"id": ["OAS1_0001_MR1", "OAS1_0002_MR1"], "age": [70, 80]})
    paths = ["d/OAS1_0001_MR1/mwc1OAS1_0001_MR1_mpr.nii.gz",
             "d/OAS1_0001_MR1/mwc2OAS1_0001_MR1_mpr.nii.gz",
             "d/OAS1_0002_MR1/mwrc1OAS1_0002_MR1_mpr.nii.gz"]
    out = data.align_maps(meta, paths)
    assert len(out) == 2 and not out["gm_path"].str.contains("c2OAS1").any()


def test_select_cohort():
    meta = pd.DataFrame({"age": [30, 65, 70, 80], "cdr": [np.nan, 0, 1, np.nan],
                         "sex_m": [1, 0, 1, 0], "etiv": [1, 1, 1, 1]})
    c = data.select_cohort(meta, {"min_age": 60, "require_cdr": True}, ["age", "sex_m", "etiv"])
    assert len(c) == 2 and list(c.is_patient) == [0, 1]


def test_residual_zscores_removes_covariates_and_flags_atrophy():
    meta = _fake_cohort()
    n = len(meta)
    # GM falls with age and rises with head size in everyone; patients lose extra GM
    gm = 0.6 - 0.004 * meta.age + 0.0001 * meta.etiv + RNG.normal(0, 0.01, n)
    gm -= 0.03 * meta.is_patient
    X = pd.DataFrame({"hippo": gm, "noise": RNG.normal(0.5, 0.02, n)})
    z, fit = zscore.residual_zscores(X, meta, meta.is_patient.to_numpy() == 0, ["age", "sex_m", "etiv"])
    ctrl = meta.is_patient == 0
    assert abs(z.loc[ctrl, "hippo"].mean()) < 1e-8          # centred on controls
    assert 0.9 < z.loc[ctrl, "hippo"].std() < 1.1
    assert z.loc[~ctrl, "hippo"].mean() > 2                  # atrophy -> positive z
    assert abs(np.corrcoef(z.loc[ctrl, "hippo"], meta.loc[ctrl, "age"])[0, 1]) < 1e-6
    assert fit["n_controls"] == ctrl.sum()


def test_choose_z_vals_and_zmax():
    z = np.column_stack([np.full(10, 2.5), np.full(10, 0.2)])
    Z = zscore.choose_z_vals(z, np.ones(10), [1, 2, 3], 0.1)
    assert Z.tolist() == [[1, 2, 0], [1, 0, 0]]
    zmax = zscore.choose_z_max(z, Z, 99)
    assert zmax[0] >= 3 and zmax[1] >= 2


def test_outlier_flag_and_clipping():
    z = pd.DataFrame({"a": [0.5, 40.0, -1.0, -9.0], "b": [1.0, 2.0, 7.0, 0.0]})
    ctrl = np.array([True, True, True, True])
    zz = pd.DataFrame({"a": np.r_[RNG.normal(0, 1, 50), 40.0], "b": RNG.normal(0, 1, 51)})
    flags = zscore.flag_outliers(zz, np.ones(51, bool), 8)
    assert flags.tolist() == [False] * 50 + [True]
    c = zscore.clip_to_model(z, -3, np.array([4.0, 5.0]))
    assert c["a"].tolist() == [0.5, 4.0, -1.0, -3.0]
    assert c["b"].tolist() == [1.0, 2.0, 5.0, 0.0]
    # Worst case after clipping cannot underflow a product of standard-normal pdfs
    from scipy.stats import norm
    worst = max(3, 5)
    assert np.prod(norm.pdf(np.full(30, worst))) > 0


def test_event_table_matches_pysustain_ordering():
    Z_vals = np.array([[1, 2, 3], [1, 2, 0], [1, 0, 0]], float)
    # Re-implementation of ZscoreSustain.__init__ event construction
    N_bio, N_z = Z_vals.shape
    IX_vals = np.array([[x for x in range(N_bio)]] * N_z).T
    stage_z = Z_vals.T.flatten()
    sel = np.nonzero(stage_z)
    ref_bio = IX_vals.T.flatten()[sel]
    ref_z = stage_z[sel]
    ev = zscore.event_table(Z_vals, ["a", "b", "c"])
    assert ev.biomarker_index.tolist() == ref_bio.tolist()
    assert ev.z.tolist() == ref_z.tolist()


def test_positional_variance_rows_and_columns_sum_to_one():
    n_events, n_samples = 6, 500
    seqs = np.array([RNG.permutation(n_events) for _ in range(n_samples)]).T
    P = analysis.positional_variance(seqs, n_events)
    assert np.allclose(P.sum(0), 1) and np.allclose(P.sum(1), 1)


def test_cv_folds_stratified():
    y = np.r_[np.zeros(60), np.ones(30)].astype(int)
    folds = model.cv_folds(y, 3, 0)
    allidx = np.sort(np.concatenate(list(folds)))
    assert allidx.tolist() == list(range(90))
    assert all(abs(y[f].mean() - 1 / 3) < 0.05 for f in folds)


def test_prepare_step_excludes_outliers_and_clips():
    import run

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cfg = config._merge(config.DEFAULTS, {
            "name": "prep", "regions": {"r1": ["x"], "r2": ["y"]},
            "paths": {"data_dir": f"{tmp}/raw", "processed_dir": f"{tmp}/proc", "output_dir": f"{tmp}/out"},
        })
        dirs = config.run_dirs(cfg)
        f = run._files(cfg, dirs)
        meta = _fake_cohort(50, 50).drop(columns=["is_patient", "group"])
        meta["gm_path"] = "unused"
        n = len(meta)
        gm = pd.DataFrame({"r1": RNG.normal(0.5, 0.02, n), "r2": RNG.normal(0.5, 0.02, n)})
        gm.loc[meta.cdr > 0, "r1"] -= 0.04
        gm.loc[7, "r2"] = 5.0  # a corrupted map: absurd z
        gm.insert(0, "id", meta["id"])
        gm.to_csv(f["regional"], index=False)  # cache hit -> no nilearn needed
        hc = np.full(n, 0.95) + RNG.normal(0, 0.005, n)
        hc[12] = 0.70  # a failed segmentation: low correlation with the mean map
        pd.DataFrame({"id": meta["id"], "map_corr": hc}).to_csv(f["homog"], index=False)

        orig = run.data.fetch_oasis
        run.data.fetch_oasis = lambda *a, **k: meta.copy()
        try:
            run.step_prepare(cfg, dirs, f)
        finally:
            run.data.fetch_oasis = orig
        qc = pd.read_csv(dirs["out"] / "qc_excluded.csv")
        assert sorted(qc["id"]) == sorted([meta.loc[7, "id"], meta.loc[12, "id"]])
        assert set(qc["reason"]) == {"low map homogeneity", "extreme regional value"}
        d = run.load_npz(f["inputs"])
        assert len(d["z"]) == n - 2
        assert (d["z"] <= d["Z_max"][None, :] + 1e-9).all() and (d["z"] >= -3 - 1e-9).all()


def test_prepare_refuses_to_exclude_real_disease():
    import run

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cfg = config._merge(config.DEFAULTS, {
            "name": "guard", "regions": {"r1": ["x"]},
            "paths": {"data_dir": f"{tmp}/raw", "processed_dir": f"{tmp}/proc", "output_dir": f"{tmp}/out"},
        })
        dirs = config.run_dirs(cfg)
        f = run._files(cfg, dirs)
        meta = _fake_cohort(50, 50).drop(columns=["is_patient", "group"])
        meta["gm_path"] = "unused"
        # near-noiseless controls: every patient looks like a huge outlier
        gm = pd.DataFrame({"id": meta["id"], "r1": 0.5 - 0.002 * meta.age - 0.03 * (meta.cdr > 0)
                           + RNG.normal(0, 1e-6, len(meta))})
        gm.to_csv(f["regional"], index=False)
        pd.DataFrame({"id": meta["id"], "map_corr": 0.95 + RNG.normal(0, 0.005, len(meta))}).to_csv(f["homog"], index=False)
        orig = run.data.fetch_oasis
        run.data.fetch_oasis = lambda *a, **k: meta.copy()
        try:
            run.step_prepare(cfg, dirs, f)
            raised = False
        except RuntimeError as e:
            raised = "QC would exclude" in str(e)
        finally:
            run.data.fetch_oasis = orig
        assert raised


def test_analyze_step_end_to_end_with_fake_sustain_output():
    import run

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        cfg = config._merge(config.DEFAULTS, {
            "name": "fake", "regions": {"a": ["x"], "b": ["y"], "c": ["z"]},
            "paths": {"data_dir": f"{tmp}/raw", "processed_dir": f"{tmp}/proc", "output_dir": f"{tmp}/out"},
            "analysis": {"n_subtypes": "auto", "certainty_threshold": 0.75},
        })
        dirs = config.run_dirs(cfg)
        f = run._files(cfg, dirs)
        meta = _fake_cohort(40, 40)
        n = len(meta)
        stage = np.where(meta.is_patient == 1, (meta.cdr * 3).round().astype(int), RNG.integers(0, 2, n))
        z = np.column_stack([stage + RNG.normal(0, 0.5, n) for _ in range(3)])
        Z_vals = np.array([[1, 2], [1, 2], [1, 0]], float)
        n_events = 5
        np.savez(f["inputs"], z=z, Z_vals=Z_vals, Z_max=np.array([4, 4, 3.0]),
                 labels=np.array(["a", "b", "c"]), is_patient=meta.is_patient.to_numpy())
        meta.to_csv(f["subjects"], index=False)
        np.savez(f["cv"], cvic=np.array([500.0, 480.0]), loglike=np.zeros((3, 2)))
        n_s, n_samp = 2, 200
        res = {
            "samples_sequence": np.stack([np.array([RNG.permutation(n_events) for _ in range(n_samp)]).T
                                          for _ in range(n_s)]),
            "samples_f": RNG.dirichlet([2, 1], n_samp).T,
            "ml_subtype": RNG.integers(0, n_s, (n, 1)),
            "prob_ml_subtype": RNG.uniform(0.4, 1, (n, 1)),
            "ml_stage": stage.reshape(-1, 1),
            "prob_ml_stage": RNG.uniform(0.3, 1, (n, 1)),
        }
        p = model.pickle_path(dirs["sustain"], "fake", 2)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump(res, fh)

        run.step_analyze(cfg, dirs, f)
        summ = pd.read_json(dirs["out"] / "summary_2subtypes.json", typ="series")
        assert summ["n_subtypes"] == 2
        assert summ["stage_vs_mmse_patients"]["rho"] < -0.5
        figs = {x.name for x in dirs["figures"].iterdir()}
        assert {"pvd_2subtypes.png", "stage_vs_clinical.png", "subtype_certainty.png"} <= figs


if __name__ == "__main__":
    fns = [v for k, v in dict(globals()).items() if k.startswith("test_")]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"{len(fns)} tests passed")
