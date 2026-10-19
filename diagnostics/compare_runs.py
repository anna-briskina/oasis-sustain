"""Robustness check: do two fitted runs agree on who is in which subtype, and at what stage?

    python diagnostics/compare_runs.py --a configs/full.yaml --b configs/full_no_temporal_pole.yaml --n 2

Uses only saved results (run `analyze` for both configs with the same --n-subtypes first).
Subtype numbers are arbitrary in each run, so subtypes are matched by maximum overlap.
"""
from __future__ import annotations

import argparse
import sys
from itertools import permutations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oasis_sustain import analysis, config, model, zscore  # noqa: E402
import run  # noqa: E402


def leading_regions(cfg, dirs, n, k=3):
    """For each subtype: the k regions whose first (lowest-z) event comes earliest on average."""
    d = run.load_npz(run._files(cfg, dirs)["inputs"])
    labels = [str(x) for x in d["labels"]]
    events = zscore.event_table(d["Z_vals"], labels)
    res = model.load_result(dirs["sustain"], cfg["name"], n)
    seqs = np.asarray(res["samples_sequence"])
    out = []
    for s in range(seqs.shape[0]):
        P = analysis.positional_variance(seqs[s], len(events))
        exp_pos = {row.biomarker: (P[e] * np.arange(len(events))).sum()
                   for e, row in events.iterrows() if row.z == events.z.min()}
        out.append([r for r, _ in sorted(exp_pos.items(), key=lambda x: x[1])[:k]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="config of run A")
    ap.add_argument("--b", required=True, help="config of run B")
    ap.add_argument("--n", type=int, required=True, help="number of subtypes to compare")
    args = ap.parse_args()

    runs = []
    for c in (args.a, args.b):
        cfg = config.load_config(c)
        dirs = config.run_dirs(cfg)
        p = dirs["out"] / f"subjects_{args.n}subtypes.csv"
        if not p.exists():
            sys.exit(f"{p} missing: run `python run.py --config {c} --steps analyze --n-subtypes {args.n}`")
        runs.append((cfg, dirs, pd.read_csv(p)))
    (ca, da, A), (cb, db, B) = runs

    m = A[["id", "group", "stage", "subtype", "mmse"]].merge(
        B[["id", "stage", "subtype"]], on="id", suffixes=("_a", "_b"))
    print(f"Subjects in both runs: {len(m)}")

    # Match subtype labels by maximum overlap among people staged > 0 in both runs
    both = m[(m.subtype_a > 0) & (m.subtype_b > 0)]
    labels = list(range(1, args.n + 1))
    ct = pd.crosstab(both.subtype_a, both.subtype_b).reindex(index=labels, columns=labels, fill_value=0)
    best, best_score = None, -1
    for perm in permutations(labels):  # perm[i] = B label matched to A label i+1
        score = sum(ct.loc[a, b] for a, b in zip(labels, perm))
        if score > best_score:
            best, best_score = {b: a for a, b in zip(labels, perm)}, score
    m["subtype_b_matched"] = m.subtype_b.map(lambda s: best.get(s, s) if s > 0 else 0)
    both = m[(m.subtype_a > 0) & (m.subtype_b > 0)]

    print(f"\n=== Stage agreement ===")
    r = stats.spearmanr(m.stage_a, m.stage_b)
    print(f"Spearman rho(stage A, stage B) = {r.statistic:.2f}  (n = {len(m)})")
    agree0 = ((m.stage_a == 0) == (m.stage_b == 0)).mean()
    print(f"Agree on 'stage 0 vs not': {agree0:.0%}")

    print(f"\n=== Subtype agreement (people staged > 0 in both runs, n = {len(both)}) ===")
    print("Rows: run A subtype. Columns: run B subtype, relabelled to best match A.")
    print(pd.crosstab(both.subtype_a, both.subtype_b_matched, margins=True))
    print(f"Same subtype after matching: {(both.subtype_a == both.subtype_b_matched).mean():.0%}")
    print(f"Adjusted Rand index: {adjusted_rand_score(both.subtype_a, both.subtype_b):.2f}  "
          f"(1 = identical, 0 = chance)")
    pat = both[both.group == "CDR>0"]
    if len(pat):
        print(f"Patients only (n = {len(pat)}): same subtype {(pat.subtype_a == pat.subtype_b_matched).mean():.0%}, "
              f"ARI {adjusted_rand_score(pat.subtype_a, pat.subtype_b):.2f}")

    print("\n=== Earliest regions per subtype (B relabelled to match A) ===")
    la, lb = leading_regions(ca, da, args.n), leading_regions(cb, db, args.n)
    inv = {v: k for k, v in best.items()}
    for s in range(1, args.n + 1):
        sb = inv.get(s, s)
        print(f"Subtype {s}:  A ({ca['name']}): {', '.join(la[s - 1])}")
        print(f"{'':12}B ({cb['name']}): {', '.join(lb[sb - 1])}")


if __name__ == "__main__":
    main()
