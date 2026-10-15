"""Figures for the write-up."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import positional_variance

_INK = "#2b2b2b"
_LEVEL_CMAPS = {1: "Reds", 2: "Purples", 3: "Blues"}


def _save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_pvd(res: dict, events: pd.DataFrame, out_dir, features: list[str]):
    """Positional variance diagram per subtype: rows = biomarkers, colour = z level."""
    seqs = np.asarray(res["samples_sequence"])
    fracs = np.asarray(res["samples_f"]).mean(axis=1)
    n_s, n_events = seqs.shape[0], len(events)
    fig, axes = plt.subplots(1, n_s, figsize=(5.5 * n_s, 0.45 * len(features) + 1.5), squeeze=False)
    for s in range(n_s):
        P = positional_variance(seqs[s], n_events)
        ax = axes[0, s]
        rgb = np.ones((len(features), n_events, 3))
        for e, row in events.iterrows():
            colour = np.array(plt.get_cmap(_LEVEL_CMAPS.get(int(row.z), "Greys"))(0.85)[:3])
            b = features.index(row.biomarker)
            rgb[b] -= P[e][:, None] * (1 - colour)[None, :]
        # order biomarkers by expected position of their first event
        first = events[events.z == events.z.min()]
        order = np.argsort([(P[e] * np.arange(n_events)).sum() for e in first.index])
        ax.imshow(np.clip(rgb[order], 0, 1), aspect="auto", interpolation="nearest")
        ax.set_yticks(range(len(features)), [features[i] for i in order], fontsize=8)
        ticks = np.arange(0, n_events, max(1, n_events // 10))
        ax.set_xticks(ticks, ticks + 1)
        ax.set_xlabel("Event position (1 = earliest)")
        ax.set_title(f"Subtype {s + 1}  (f = {fracs[s]:.2f})", color=_INK)
    fig.suptitle("Positional variance: darker = more certain.  Red z=1, purple z=2, blue z=3",
                 fontsize=9, color="#555", y=0.02)
    _save(fig, f"{out_dir}/pvd_{n_s}subtypes.png")


def plot_stage_vs_clinical(df: pd.DataFrame, out_dir, tag: str = ""):
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4))
    cdrs = sorted(df["cdr"].dropna().unique())
    data = [df.loc[df["cdr"] == c, "stage"] for c in cdrs]
    names = [f"{c:g}" for c in cdrs]
    try:
        a.boxplot(data, tick_labels=names)   # matplotlib >= 3.9
    except TypeError:
        a.boxplot(data, labels=names)
    for i, d in enumerate(data, start=1):
        a.scatter(np.random.normal(i, 0.06, len(d)), d, s=8, alpha=0.5, color=_INK)
    a.set_xlabel("CDR"); a.set_ylabel("SuStaIn stage"); a.set_title("Stage by clinical dementia rating")
    for g, colour in [("CDR=0", "#4a7fb5"), ("CDR>0", "#c0504d")]:
        d = df[df["group"] == g]
        b.scatter(d["stage"], d["mmse"], s=12, alpha=0.7, color=colour, label=g)
    b.set_xlabel("SuStaIn stage"); b.set_ylabel("MMSE"); b.legend(frameon=False)
    b.set_title("Stage vs cognition (not a model input)")
    _save(fig, f"{out_dir}/stage_vs_clinical{tag}.png")


def plot_cvic(cvic: np.ndarray, out_dir):
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ks = np.arange(1, len(cvic) + 1)
    ax.plot(ks, cvic, "o-", color=_INK)
    ax.set_xticks(ks); ax.set_xlabel("Number of subtypes"); ax.set_ylabel("CVIC (lower is better)")
    ax.set_title("Cross-validated model selection")
    _save(fig, f"{out_dir}/cvic.png")


def plot_subtype_certainty(df: pd.DataFrame, out_dir, tag: str = ""):
    staged = df[df["subtype"] > 0]
    if staged.empty:
        return
    fig, ax = plt.subplots(figsize=(4.5, 3.5))
    ax.hist(staged["prob_subtype"], bins=np.linspace(0, 1, 21), color="#7a6aa8")
    ax.set_xlabel("P(assigned subtype)"); ax.set_ylabel("Subjects (stage > 0)")
    ax.set_title("Subtype assignment certainty")
    _save(fig, f"{out_dir}/subtype_certainty{tag}.png")


def plot_zscores(z: pd.DataFrame, group: pd.Series, out_dir):
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * z.shape[1]), 4))
    pos = np.arange(z.shape[1])
    for off, g, colour in [(-0.18, "CDR=0", "#4a7fb5"), (0.18, "CDR>0", "#c0504d")]:
        d = z[group.to_numpy() == g]
        bp = ax.boxplot([d[c] for c in z.columns], positions=pos + off, widths=0.3,
                        patch_artist=True, showfliers=False)
        for p in bp["boxes"]:
            p.set_facecolor(colour); p.set_alpha(0.6)
    ax.axhline(0, color="#999", lw=0.8)
    ax.set_xticks(pos, z.columns, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Atrophy z (vs adjusted controls)")
    ax.set_title("Input biomarkers by group  (blue CDR=0, red CDR>0)")
    _save(fig, f"{out_dir}/zscores_by_group.png")
