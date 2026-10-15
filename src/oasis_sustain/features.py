"""Regional grey-matter density from VBM maps using Harvard-Oxford atlases."""
from __future__ import annotations

import difflib

import numpy as np
import pandas as pd


def _resample_nearest(atlas_img, ref_img):
    from nilearn import image

    try:  # nilearn >= 0.11 warns unless these are given explicitly
        return image.resample_to_img(atlas_img, ref_img, interpolation="nearest",
                                     force_resample=True, copy_header=True)
    except TypeError:
        return image.resample_to_img(atlas_img, ref_img, interpolation="nearest")


def load_atlases(names: list[str], data_dir, ref_path: str):
    """Return [(label_array_in_ref_space, labels_list)] for each Harvard-Oxford atlas."""
    from nilearn import datasets, image

    ref = image.load_img(ref_path)
    out = []
    for name in names:
        atlas = datasets.fetch_atlas_harvard_oxford(name, data_dir=str(data_dir))
        img = _resample_nearest(image.load_img(atlas.maps), ref)
        arr = np.rint(np.asanyarray(img.dataobj)).astype(int)
        out.append((np.squeeze(arr), [str(l) for l in atlas.labels]))
    return out


def build_region_masks(atlases, regions: dict[str, list[str]]) -> dict[str, np.ndarray]:
    """Union of atlas labels per feature. Fails loudly with suggestions on a bad label name."""
    all_labels = sorted({l for _, labels in atlases for l in labels})
    masks = {}
    for feat, label_names in regions.items():
        mask = None
        for lname in label_names:
            hit = False
            for arr, labels in atlases:
                for idx, l in enumerate(labels):
                    if l == lname and idx > 0:
                        m = arr == idx
                        mask = m if mask is None else (mask | m)
                        hit = True
            if not hit:
                close = difflib.get_close_matches(lname, all_labels, n=5, cutoff=0.4)
                raise KeyError(f"Region '{feat}': atlas label '{lname}' not found. "
                               f"Close matches: {close}")
        if mask is None or mask.sum() == 0:
            raise ValueError(f"Region '{feat}' has no voxels in the VBM grid.")
        masks[feat] = mask
    return masks


def extract_regional_means(paths: list[str], masks: dict[str, np.ndarray]) -> pd.DataFrame:
    import nibabel as nib

    shape = next(iter(masks.values())).shape
    rows = []
    for i, p in enumerate(paths):
        data = np.squeeze(np.asanyarray(nib.load(p).dataobj, dtype=np.float32))
        if data.shape != shape:
            raise ValueError(f"{p}: shape {data.shape} != atlas grid {shape}")
        rows.append({f: float(data[m].mean()) for f, m in masks.items()})
        if (i + 1) % 50 == 0:
            print(f"  extracted {i + 1}/{len(paths)}")
    return pd.DataFrame(rows)
