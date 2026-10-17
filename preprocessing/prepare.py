"""Validate a phase-based input layout and pack one inference/annotation case.

Read A/<case>.nii.gz, V2A/<case>.nii.gz and ROI/<case>.nii.gz; optionally
include DOT/<case>.nii.gz. V contains the original venous CT and is never
used as a fallback for registered V2A. This command does not segment,
register, resample, anonymize, or create point annotations.
"""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

import numpy as np

from .io import label_mask, load_matched


def pack_case(data_root, case_id, output_dir):
    """Copy a validated case without changing source voxels, HU or geometry.

    ``case_id`` must be a deidentified filename stem without an extension.
    Existing output files are never overwritten. DOT is included when present;
    expert points outside the tumour are ignored later by prior learning.
    """
    if not isinstance(case_id, str) or not re.fullmatch(
        r"[A-Za-z0-9][A-Za-z0-9_-]{0,63}", case_id
    ):
        raise ValueError("case_id must contain 1-64 letters, digits, underscores or hyphens")
    if case_id.upper() in {"CON", "PRN", "AUX", "NUL"} | {
        f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
    }:
        raise ValueError("case_id is a reserved filename; choose a neutral case identifier")

    data_root = Path(data_root).resolve()
    output_dir = Path(output_dir).resolve()
    for folder in ("A", "V", "V2A", "ROI", "DOT"):
        source_folder = (data_root / folder).resolve()
        if output_dir == source_folder or source_folder in output_dir.parents:
            raise ValueError("output_dir must be outside the source phase folders")
    filename = f"{case_id}.nii.gz"
    sources = {
        "arterial.nii.gz": data_root / "A" / filename,
        "venous.nii.gz": data_root / "V2A" / filename,
        "tumor_roi.nii.gz": data_root / "ROI" / filename,
    }
    points_path = data_root / "DOT" / filename
    if points_path.exists():
        sources["points.nii.gz"] = points_path
    for source in sources.values():
        if not source.is_file():
            raise ValueError(
                f"Required {source.parent.name} NIfTI is missing; V2A must be supplied explicitly"
            )
    output_names = (
        "arterial.nii.gz",
        "venous.nii.gz",
        "tumor_roi.nii.gz",
        "points.nii.gz",
    )
    for target in (output_dir / name for name in output_names):
        if target.exists() or target.is_symlink():
            raise FileExistsError(
                "An output file already exists; choose a new output directory"
            )
        if target.resolve() in {source.resolve() for source in sources.values()}:
            raise ValueError("An output path overlaps a source image")

    images = load_matched(*sources.values())
    if not np.any(images[2].get_fdata() == 2):
        raise ValueError(
            "ROI must contain tumour label 2; export a binary tumour mask with label 2 before packing"
        )
    roi = label_mask(images[2], 2)
    for image in images[:2]:
        if not np.isfinite(image.get_fdata(dtype=np.float32)).all():
            raise ValueError("CT images contain non-finite values")
    if "points.nii.gz" in sources:
        points = images[3].get_fdata()
        if not np.isfinite(points).all() or not np.isin(points, (0, 1, 2, 3)).all():
            raise ValueError("DOT must contain integer labels 0, 1, 2 and 3 only")
        if not (roi & np.isin(points, (1, 2, 3))).any():
            raise ValueError("DOT must contain at least one expert point inside the tumour ROI")

    output_dir.mkdir(parents=True, exist_ok=True)
    created = []
    try:
        for name, source in sources.items():
            target = output_dir / name
            # Exclusive creation keeps the no-overwrite guarantee during copying.
            with source.open("rb") as reader, target.open("xb") as writer:
                created.append(target)
                shutil.copyfileobj(reader, writer)
    except Exception:
        for target in created:
            target.unlink(missing_ok=True)
        raise
    return output_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument(
        "--case-id", required=True, help="Deidentified filename stem, e.g. case_001"
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    try:
        destination = pack_case(args.data_root, args.case_id, args.output_dir)
    except (ValueError, OSError) as error:
        parser.exit(2, f"{error}\n")
    contents = ", ".join(
        name
        for name in (
            "arterial.nii.gz",
            "venous.nii.gz",
            "tumor_roi.nii.gz",
            "points.nii.gz",
        )
        if (destination / name).is_file()
    )
    print(f"Validated and packed: {contents}")


if __name__ == "__main__":
    main()
