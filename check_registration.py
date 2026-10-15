"""Is the atlas on the right anatomy? Writes overlay figures and prints space info.

    python check_registration.py --config configs/quick.yaml

Outputs (in outputs/<name>/figures/):
  reg_gm_on_mni.png      mean GM map (red contour) over the MNI152 template
  reg_atlas_on_gm.png    our region masks over the mean GM map
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))
sys.path.insert(0, str(Path(__file__).parent))

from oasis_sustain import config, data, features  # noqa: E402


def main():
    import matplotlib
    matplotlib.use("Agg")
    import nibabel as nib
    from nilearn import datasets, image, plotting

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--n_mean", type=int, default=40, help="subjects averaged for the mean GM map")
    args = ap.parse_args()
    cfg = config.load_config(args.config)
    dirs = config.run_dirs(cfg)

    meta = data.fetch_oasis(dirs["data"], cfg["cohort"]["n_subjects"])
    paths = meta["gm_path"].tolist()
    ref = image.load_img(paths[0])
    print("GM map shape:", ref.shape[:3], " voxel size:", np.round(ref.header.get_zooms()[:3], 2))
    print("GM map orientation:", "".join(nib.aff2axcodes(ref.affine)))
    print("GM map affine:\n", np.round(ref.affine, 2))
    print("GM qform/sform codes:", int(ref.header["qform_code"]), int(ref.header["sform_code"]))

    for name in cfg["atlases"]:
        atl = datasets.fetch_atlas_harvard_oxford(name, data_dir=str(dirs["data"]))
        a = image.load_img(atl.maps)
        print(f"\nAtlas {name}: shape {a.shape[:3]}, orientation {''.join(nib.aff2axcodes(a.affine))}")
        print(np.round(a.affine, 2))

    mean_gm = image.mean_img(paths[: args.n_mean])
    gm_arr = np.asarray(mean_gm.dataobj)
    brain_gm = gm_arr[gm_arr > 0.1]
    print(f"\nMean GM over brain voxels (>0.1): {brain_gm.mean():.3f}")

    atl = features.load_atlases(cfg["atlases"], dirs["data"], paths[0])
    masks = features.build_region_masks(atl, cfg["regions"])
    lab = np.zeros(ref.shape[:3], dtype=np.int16)
    print("\nMean of the mean-GM map inside each region (cortex/hippocampus should be HIGH, ~0.4+):")
    for k, (name, m) in enumerate(masks.items(), start=1):
        lab[m & (lab == 0)] = k
        print(f"  {k}. {name:<22} {gm_arr[m].mean():.3f}   (voxels {int(m.sum())})")
    lab_img = image.new_img_like(ref, lab)

    out = dirs["figures"]
    tmpl = datasets.load_mni152_template()
    disp = plotting.plot_anat(tmpl, display_mode="ortho", cut_coords=(-25, -20, -15),
                              title="Red = mean OASIS GM. Should hug MNI cortex")
    disp.add_contours(mean_gm, levels=[0.3], colors="r", linewidths=1)
    disp.savefig(str(out / "reg_gm_on_mni.png"))
    disp.close()

    disp = plotting.plot_roi(lab_img, bg_img=mean_gm, display_mode="ortho",
                             cut_coords=(-25, -20, -15), cmap="tab10", alpha=0.6,
                             title="Regions on mean GM (crosshair ~ left hippocampus in MNI)")
    disp.savefig(str(out / "reg_atlas_on_gm.png"))
    disp.close()
    print(f"\nFigures written to {out}")


if __name__ == "__main__":
    main()
