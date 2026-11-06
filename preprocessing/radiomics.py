"""Extract 200 dual-phase Original features with PyRadiomics 3.1.0.

Retain 14 shared shape features and 93 nonshape features from each phase.
"""

from __future__ import annotations

import argparse
import copy
import logging
import warnings
from pathlib import Path

import numpy as np

from .io import load_case, save_json

SETTINGS = {
    "normalize": True,
    "normalizeScale": 255,
    "binWidth": 25,
    "resampledPixelSpacing": [1, 1, 1],
    "interpolator": "sitkBSpline",
    "force2D": False,
    "force2Ddimension": 0,
    "preCrop": False,
    "padDistance": 5,
    "label": 2,
    "additionalInfo": True,
}
CONFIG = {
    "imageType": {"Original": {}},
    "featureClass": {
        name: [] for name in ("shape", "firstorder", "glcm", "glrlm", "glszm", "gldm", "ngtdm")
    },
    "setting": SETTINGS,
}


def _dependencies():
    try:
        import radiomics as pyradiomics
        import SimpleITK as sitk
        from radiomics.featureextractor import RadiomicsFeatureExtractor
    except ImportError as exc:
        raise ImportError(
            "Radiomics extraction needs PyRadiomics 3.1.0 and SimpleITK; use the documented imaging environment"
        ) from exc
    return pyradiomics, sitk, RadiomicsFeatureExtractor


def _sitk_image(array, affine, sitk):
    """Convert nibabel RAS XYZ arrays to SimpleITK LPS ZYX arrays exactly."""
    spacing = np.linalg.norm(affine[:3, :3], axis=0)
    direction_ras = affine[:3, :3] / spacing
    if not np.allclose(direction_ras.T @ direction_ras, np.eye(3), rtol=0, atol=1e-5):
        raise ValueError(
            "Radiomics requires an orthogonal NIfTI grid; remove affine shear explicitly first"
        )
    ras_to_lps = np.diag([-1.0, -1.0, 1.0])
    image = sitk.GetImageFromArray(np.transpose(array, (2, 1, 0)))
    image.SetSpacing(tuple(float(x) for x in spacing))
    image.SetOrigin(tuple(float(x) for x in ras_to_lps @ affine[:3, 3]))
    image.SetDirection(tuple(float(x) for x in (ras_to_lps @ direction_ras).ravel()))
    return image


def extract_radiomics(case_dir, *, tumor_label=2, strict_version=True):
    """Return features for a registered HU-scale case.

    Whole-image normalization depends on the input field of view; preserve
    the study crop. Set strict_version=False only for compatibility runs.
    """
    pyradiomics, sitk, extractor_class = _dependencies()
    actual_version = pyradiomics.__version__.lstrip("v")
    if actual_version != "3.1.0":
        message = f"Source PyRadiomics version is 3.1.0; installed version is {actual_version}"
        if strict_version:
            raise RuntimeError(
                message + "; select the source environment or explicitly allow a version mismatch"
            )
        warnings.warn(message, RuntimeWarning, stacklevel=2)
    arterial, venous, mask, reference = load_case(case_dir, tumor_label)
    mask_image = _sitk_image(mask.astype(np.uint8) * 2, reference.affine, sitk)
    extractor = extractor_class(copy.deepcopy(CONFIG))
    # PyRadiomics diagnostic text may contain local paths or input metadata.
    logger = logging.getLogger("radiomics")
    old_level = logger.level
    logger.setLevel(logging.ERROR)
    result = {}
    try:
        for phase, array in (("A", arterial), ("V", venous)):
            image = _sitk_image(array, reference.affine, sitk)
            for name, value in extractor.execute(image, mask_image).items():
                if name.startswith("original_shape_"):
                    if phase == "A":
                        result[name] = float(value)
                elif name.startswith("original_"):
                    result[f"{phase}_{name}"] = float(value)
    finally:
        logger.setLevel(old_level)
    if len(result) != 200:
        raise RuntimeError(
            f"Expected 200 nondeprecated Original features; extractor returned {len(result)}"
        )
    if not np.isfinite(list(result.values())).all():
        raise ValueError("PyRadiomics returned non-finite feature values")
    return {"features": result}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="Feature JSON for predict.py --input"
    )
    parser.add_argument("--tumor-label", choices=(1, 2), type=int, default=2)
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="Allow extraction with a different PyRadiomics version",
    )
    args = parser.parse_args()
    result = extract_radiomics(
        args.case, tumor_label=args.tumor_label, strict_version=not args.allow_version_mismatch
    )
    save_json(result, args.output)
    print("Saved 200 dual-phase Original radiomics features")


if __name__ == "__main__":
    main()
