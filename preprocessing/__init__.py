"""Core methods for registered, segmented dual-phase CT.

Run steps with ``python -m preprocessing.learn_prior --help``,
``python -m preprocessing.cluster --help``,
``python -m preprocessing.heterogeneity --help``, and
``python -m preprocessing.radiomics --help`` from the repository root.
"""


def prepare_case(case_dir, prior, *, include_radiomics=False, output_dir=None, tumor_label=2):
    """Return a flat measured feature dictionary for the shared predict.py.

    Returns the 18 THI candidates, plus the 200 radiomics features when
    include_radiomics=True. Radiomics provenance warnings remain explicit.
    No normalization or score imputation is applied to the returned features.
    """
    from pathlib import Path
    from .cluster import cluster_arrays
    from .heterogeneity import extract_features
    from .io import load_case, save_habitats, save_json

    arterial, venous, mask, image = load_case(case_dir, tumor_label)
    habitats = cluster_arrays(arterial, venous, mask, prior)
    features = extract_features(arterial, venous, habitats, image.affine)
    if output_dir is not None:
        output_dir = Path(output_dir)
        save_habitats(habitats, image, output_dir / "habitats.nii.gz")
        save_json(features, output_dir / "heterogeneity_features.json")
    if include_radiomics:
        from .radiomics import extract_radiomics

        radiomics = extract_radiomics(case_dir, tumor_label=tumor_label)
        features.update(radiomics["features"])
        if output_dir is not None:
            save_json(radiomics, output_dir / "radiomics_features.json")
    return features
