"""Dual-phase Original PyRadiomics extraction using audited source settings.

Produces 14 shared shape features + 93 arterial + 93 venous features.
This is a documented extraction implementation, not a claim of exact
reproduction of the historical 200-D CSV or its derived RAD scores.
"""

from __future__ import annotations

import argparse
import copy
import logging
import platform
import warnings
from pathlib import Path

import numpy as np

from .io import load_case, save_json

SOURCE_ENVIRONMENT = {
    "python": "3.9.25",
    "pyradiomics": "3.1.0",
    "numpy": "1.26.4",
    "SimpleITK": "2.5.5",
}
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
        "shape": [],
        "firstorder": [],
        "glcm": [],
        "glrlm": [],
        "glszm": [],
        "gldm": [],
        "ngtdm": [],
    },
    "setting": SETTINGS,
}
PROVENANCE_NOTE = (
    "Source tables record PyRadiomics 3.1.0; legacy 546-case tables record 3.0.1. "
    "Nine historical 200-D CSV SurfaceVolumeRatio values differ materially from their source tables. "
    "Fresh extraction therefore cannot be asserted to reproduce historical RAD scores exactly."
)


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
    """Extract both phases, deduplicate shape, and return features + provenance.

    Uses source HU, not the habitat window. Each whole input image is
    z-normalized and scaled to 255 before radiomics resampling. Input field
    of view therefore affects intensity features and should match the
    intended source preprocessing. Diagnostic paths and image hashes are
    deliberately excluded from the portable output.

    Source PyRadiomics version is required by default; use
    strict_version=False only for an explicitly labelled compatibility run.
    The output never sets historical_reproduction_verified=True.
    """
    pyradiomics, sitk, extractor_class = _dependencies()
    warnings.warn(PROVENANCE_NOTE, RuntimeWarning, stacklevel=2)
    actual_version = pyradiomics.__version__.lstrip("v")
    if actual_version != SOURCE_ENVIRONMENT["pyradiomics"]:
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
    phases = {}
    try:
        for phase, array in (("A", arterial), ("V", venous)):
            image = _sitk_image(array, reference.affine, sitk)
            values = extractor.execute(image, mask_image)
            phases[phase] = {
                key: float(value) for key, value in values.items() if key.startswith("original_")
            }
    finally:
        logger.setLevel(old_level)
    result = {key: value for key, value in phases["A"].items() if key.startswith("original_shape_")}
    for phase in ("A", "V"):
        result.update(
            {
                f"{phase}_{key}": value
                for key, value in phases[phase].items()
                if not key.startswith("original_shape_")
            }
        )
    if len(result) != 200:
        raise RuntimeError(
            f"Expected 200 nondeprecated Original features; extractor returned {len(result)}"
        )
    if not np.isfinite(list(result.values())).all():
        raise ValueError("PyRadiomics returned non-finite feature values")
    return {
        "features": result,
        "provenance": {
            "schema": "pnen-thi.dualphase-radiomics.v1",
            "source_environment": dict(SOURCE_ENVIRONMENT),
            "actual_environment": {
                "python": platform.python_version(),
                "pyradiomics": actual_version,
                "numpy": np.__version__,
                "SimpleITK": sitk.Version_VersionString(),
            },
            "configuration": copy.deepcopy(CONFIG),
            "n_features": len(result),
            "historical_reproduction_verified": False,
            "note": PROVENANCE_NOTE,
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", required=True, type=Path)
    parser.add_argument(
        "--output", required=True, type=Path, help="200-D feature JSON with extraction provenance"
    )
    parser.add_argument("--tumor-label", choices=(1, 2), type=int, default=2)
    parser.add_argument(
        "--allow-version-mismatch",
        action="store_true",
        help="Label a compatibility extraction with a different PyRadiomics version",
    )
    args = parser.parse_args()
    result = extract_radiomics(
        args.case, tumor_label=args.tumor_label, strict_version=not args.allow_version_mismatch
    )
    save_json(result, args.output)
    print(
        "Saved 200 dual-phase Original radiomics features; historical RAD equivalence is not established"
    )


if __name__ == "__main__":
    main()
