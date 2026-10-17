"""Strict NIfTI geometry and source-HU input handling; no implicit registration."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np

GEOMETRY_ATOL = 1e-5


def load_matched(*paths):
    """Load truly 3-D images on the same grid (shape, affine, spacing).

    Matching grids do not establish anatomical registration. Input phase
    alignment and the tumour segmentation must be checked before this step.
    """
    images = [nib.load(str(path)) for path in paths]
    if not images:
        raise ValueError("At least one image is required")
    first = images[0]
    if len(first.shape) != 3:
        raise ValueError("Every input must be a three-dimensional NIfTI")
    if not np.allclose(first.header.get_zooms(), (1.0, 1.0, 1.0), rtol=0, atol=GEOMETRY_ATOL):
        raise ValueError(
            "The locked methods require 1-mm isotropic inputs; resample CT and masks upstream"
        )
    for image in images:
        if len(image.shape) != 3:
            raise ValueError("Every input must be a three-dimensional NIfTI")
        if not np.isfinite(image.affine).all() or abs(np.linalg.det(image.affine[:3, :3])) <= 0:
            raise ValueError("Invalid NIfTI spatial affine")
        qform, qcode = image.get_qform(coded=True)
        sform, scode = image.get_sform(coded=True)
        if qcode and scode and not np.allclose(qform, sform, rtol=0, atol=GEOMETRY_ATOL):
            raise ValueError(
                "NIfTI qform and sform disagree; resolve the intended geometry explicitly"
            )
        if image.shape != first.shape:
            raise ValueError("Inputs have different voxel shapes; register/resample them first")
        if not np.allclose(image.affine, first.affine, rtol=0, atol=GEOMETRY_ATOL):
            raise ValueError(
                "Inputs have different spatial affines; register/resample them first"
            )
        if not np.allclose(
            image.header.get_zooms(), first.header.get_zooms(), rtol=0, atol=GEOMETRY_ATOL
        ):
            raise ValueError("Inputs have different voxel spacing")
    return images


def label_mask(image, tumor_label=2):
    """Select label 2; a binary label-1 ROI requires an explicit override."""
    if tumor_label not in (1, 2):
        raise ValueError("tumor_label must be explicitly 1 or 2")
    values = image.get_fdata()
    if not np.isfinite(values).all() or not np.equal(values, np.rint(values)).all():
        raise ValueError("ROI labels must be finite integers")
    mask = values == tumor_label
    if not mask.any():
        raise ValueError(
            f"No tumour voxels with label {tumor_label}; binary ROIs need --tumor-label 1"
        )
    return mask


def load_case(case_dir, tumor_label=2):
    """Return arterial HU, venous HU, tumour mask, and arterial NIfTI.

    Files are arterial.nii.gz, venous.nii.gz, tumor_roi.nii.gz. CT inputs
    must retain source HU, not windowed/normalised intensity images.
    """
    case_dir = Path(case_dir)
    a_img, v_img, roi_img = load_matched(
        case_dir / "arterial.nii.gz", case_dir / "venous.nii.gz", case_dir / "tumor_roi.nii.gz"
    )
    mask = label_mask(roi_img, tumor_label)
    arterial = a_img.get_fdata(dtype=np.float32)
    venous = v_img.get_fdata(dtype=np.float32)
    if not np.isfinite(arterial).all() or not np.isfinite(venous).all():
        raise ValueError("CT images contain non-finite values")
    return arterial, venous, mask, a_img


def normalize_hu(image):
    """The locked habitat window: clip((HU + 160) / 400, 0, 1)."""
    return np.clip((image + 160) / 400, 0, 1)


def save_json(value, path):
    """Write finite, portable JSON; no patient identifiers are added."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def save_habitats(labels, reference, path):
    """Save derived labels on the reference grid with a clean NIfTI header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    image = nib.Nifti1Image(np.asarray(labels, dtype=np.uint8), reference.affine)
    image.header.set_xyzt_units(*reference.header.get_xyzt_units())
    nib.save(image, str(path))
