"""Do the subtype labels on people match the subtype sequences in the figures?

    python diagnostics/check_subtype_labels.py --config configs/full.yaml --n 3

People's subtype labels (subjects_*.csv, from pySuStaIn's ml_subtype) and the sequences
(pvd figure, from samples_sequence) come from different arrays. This checks they line up
using only the data: people in a subtype should show the most atrophy, at early stages, in
the regions that subtype's sequence puts first.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from oasis_sustain import analysis, config, model, zscore  # noqa: E402
import run  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n", type=int, required=True)
    args = ap.parse_args()
    cfg = config.load_config(args.config)
    dirs = config.run_dirs(cfg)
    f = run._files(cfg, dirs)

    d = run.load_npz(f["inputs"])
    labels = [str(x) for x in d["labels"]]
    z = pd.DataFrame(d["z"], columns=labels)
    events = zscore.event_table(d["Z_vals"], labels)
    res = model.load_result(dirs["sustain"], cfg["name"], args.n)
    seqs = np.asarray(res["samples_sequence"])
    subj = pd.read_csv(dirs["out"] / f"subjects_{args.n}subtypes.csv")
    if len(subj) != len(z):
        sys.exit("subjects file and model inputs differ in length; rerun analyze")

    first = events[events.z == events.z.min()]
    earliness = []
    for s in range(seqs.shape[0]):
        P = analysis.positional_variance(seqs[s], len(events))
        pos = {row.biomarker: (P[e] * np.arange(len(events))).sum() for e, row in first.iterrows()}
        earliness.append(np.array([-pos[b] for b in labels]))
        lead = [b for b, _ in sorted(pos.items(), key=lambda x: x[1])[:3]]
        print(f"Sequence {s + 1} (figure panel {s + 1}) starts with: {', '.join(lead)}")

    print("\nPeople labelled subtype k: most atrophied regions at early stages")
    M = np.zeros((args.n, seqs.shape[0]))
    for k in range(1, args.n + 1):
        mem = subj[(subj.subtype == k) & (subj.stage > 0)]
        early = mem[mem.stage <= mem.stage.median()]
        prof = z.loc[early.index].mean()
        print(f"  label {k} (n early = {len(early)}): {', '.join(prof.sort_values(ascending=False).index[:3])}")
        for s in range(seqs.shape[0]):
            M[k - 1, s] = stats.spearmanr(prof.to_numpy(), earliness[s]).statistic
    print("\nSpearman: rows = people's label, columns = sequence (high = this label's people match this sequence)")
    print(pd.DataFrame(M, index=[f"label {k}" for k in range(1, args.n + 1)],
                       columns=[f"seq {s + 1}" for s in range(seqs.shape[0])]).round(2).to_string())
    best = M.argmax(axis=1) + 1
    ok = all(best[k - 1] == k for k in range(1, args.n + 1))
    print("\nALIGNED: label k matches sequence k" if ok else
          f"\nMISALIGNED: label -> best sequence {dict(zip(range(1, args.n + 1), best.tolist()))}")


if __name__ == "__main__":
    main()
