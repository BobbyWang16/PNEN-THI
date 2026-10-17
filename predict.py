"""PNET-ITH: one entry point for the frozen grading models.

Model definitions, score formulas, clinical transforms, and the main expert
prior live in the dictionaries below. Selected historical fitted estimators
are read from one shared root-level weights.joblib. Image methods and feature
extraction live in preprocessing/. No search or evaluation suite is required.
"""

import argparse
import bisect
import json
import math
import numbers
from pathlib import Path

MODELS = {
    "model8": {
        "selection": "training fit; internal model/feature/hyperparameter/ensemble/threshold "
        "selection; locked external evaluation",
        "algorithm": "Weighted nonlinear ensemble",
        "components": [
            {
                "algorithm": "RBF-SVC",
                "weight": 0.4,
                "input_order": ["LDH", "AST", "RAD", "THI"],
                "parameters": {
                    "C": 0.3,
                    "gamma": "scale",
                    "class_weight": None,
                    "max_iter": -1,
                    "probability": True,
                    "random_state": 42,
                },
            },
            {
                "algorithm": "Multilayer perceptron",
                "weight": 0.2,
                "input_order": ["LDH", "AST", "RAD", "THI"],
                "parameters": {
                    "hidden_layer_sizes": [8, 4],
                    "alpha": 1.0,
                    "learning_rate_init": 0.0005,
                    "activation": "tanh",
                    "solver": "lbfgs",
                    "max_iter": 3000,
                    "random_state": 50,
                },
            },
            {
                "algorithm": "Polynomial SVM",
                "weight": 0.39999999999999997,
                "input_order": ["SFI", "AST", "RAD", "THI"],
                "parameters": {
                    "C": 0.1,
                    "degree": 2,
                    "gamma": 1.0,
                    "coef0": 0.0,
                    "class_weight": "balanced",
                    "max_iter": -1,
                    "probability": True,
                    "random_state": 43,
                },
            },
        ],
        "probability_rule": "weighted_average_then_clip_each_entry_to_1e-10_through_1_then_row_normalize",
        "grade_rule": "argmax_class_probability_first_index_on_tie",
        "ordinal_thresholds_applicable": False,
        "source_sha256": "2fdc7a6d44ac0590b3a1e9049986929877b2ce9074456042f55a89f69ecc08c5",
    },
    "model9": {
        "status": "exploratory_post_hoc_external_constrained",
        "selection_rule": "highest external macro-AUC among combinations with internal "
        "macro-AUC >= 0.812 and external macro-AUC > 0.72",
        "external_evaluation_role": "exploratory_external_cohort_contributed_to_selection",
        "boundaries": [
            {
                "name": "G1_vs_G2_G3_NEC",
                "positive_event": "grade_at_least_G2",
                "components": [
                    {
                        "algorithm": "RBF-SVC",
                        "weight": 0.2,
                        "input_order": ["LDH", "AST", "RAD", "THI"],
                        "parameters": {
                            "C": 0.3,
                            "gamma": 0.5,
                            "class_weight": None,
                            "max_iter": -1,
                            "probability": True,
                            "random_state": 142,
                        },
                    },
                    {
                        "algorithm": "Polynomial SVM",
                        "weight": 0.56,
                        "input_order": ["SFI", "AST", "RAD", "THI"],
                        "parameters": {
                            "C": 3.0,
                            "degree": 3,
                            "gamma": 1.0,
                            "coef0": 1.0,
                            "class_weight": "balanced",
                            "max_iter": -1,
                            "probability": True,
                            "random_state": 143,
                        },
                    },
                    {
                        "algorithm": "k-nearest neighbors",
                        "weight": 0.16000000000000003,
                        "input_order": ["LDH", "RIF%", "RAD", "THI"],
                        "parameters": {"n_neighbors": 11, "weights": "uniform", "p": 2},
                    },
                    {
                        "algorithm": "Multilayer perceptron",
                        "weight": 0.07999999999999995,
                        "input_order": ["LDH", "AST", "RAD", "THI"],
                        "parameters": {
                            "hidden_layer_sizes": [8],
                            "alpha": 0.01,
                            "learning_rate_init": 0.0005,
                            "activation": "tanh",
                            "solver": "lbfgs",
                            "max_iter": 3000,
                            "random_state": 150,
                        },
                    },
                ],
            },
            {
                "name": "G1_G2_vs_G3_NEC",
                "positive_event": "G3_NEC",
                "components": [
                    {
                        "algorithm": "Polynomial SVM",
                        "weight": 0.7000000000000001,
                        "input_order": ["AST", "RAD", "THI"],
                        "parameters": {
                            "C": 0.1,
                            "degree": 2,
                            "gamma": 0.3,
                            "coef0": 0.0,
                            "class_weight": "balanced",
                            "max_iter": -1,
                            "probability": True,
                            "random_state": 243,
                        },
                    },
                    {
                        "algorithm": "Multilayer perceptron",
                        "weight": 0.29999999999999993,
                        "input_order": ["size", "AST", "RIF%", "RAD", "THI"],
                        "parameters": {
                            "hidden_layer_sizes": [8],
                            "alpha": 0.01,
                            "learning_rate_init": 0.0005,
                            "activation": "tanh",
                            "solver": "lbfgs",
                            "max_iter": 3000,
                            "random_state": 250,
                        },
                    },
                ],
            },
        ],
        "monotonic_correction": "if_q2_gt_q1_set_both_to_their_mean",
        "class_probabilities": ["1-q1", "q1-q2", "q2"],
        "class_probability_postprocessing": "clip_to_0_through_1_then_row_normalize",
        "risk": "p_G2_plus_2_times_p_G3_NEC",
        "thresholds": {
            "lower_threshold": 0.6772685594841208,
            "upper_threshold": 0.9673987735976841,
        },
        "threshold_equality": "higher_category",
        "source_sha256": "c3012f5fd741615dc36c450110cd7dd1d4438b1d29687028e1754539f133a1bc",
    },
}

SCORES = {
    "THI": {
        "candidate_feature_count": 18,
        "active_feature_count": 7,
        "raw_intercept": 2.662530430455249,
        "standardized_intercept": 8.881784197001252e-16,
        "feature_missing_policy": "use_corresponding_training_median_before_formula_evaluation",
        "terms": [
            {
                "feature": "shi_centroid_spread",
                "coefficient_raw_input": -6.5027376489553514,
                "coefficient_standardized_input": -0.6340821297090068,
                "training_median_for_imputation": 0.2103395805168333,
                "training_mean": 0.2123559191303095,
                "training_sd": 0.0975100279204514,
            },
            {
                "feature": "ddi_a_js_divergence",
                "coefficient_raw_input": -1.8426185821204009,
                "coefficient_standardized_input": -0.2986944442645321,
                "training_median_for_imputation": 0.1907573599087576,
                "training_mean": 0.22593083054561416,
                "training_sd": 0.16210324109551108,
            },
            {
                "feature": "shi_radial_wasserstein",
                "coefficient_raw_input": -3.7895442989992367,
                "coefficient_standardized_input": -0.26064118639181477,
                "training_median_for_imputation": 0.1348020885908512,
                "training_mean": 0.14642107908771398,
                "training_sd": 0.06877903141563652,
            },
            {
                "feature": "ddi_a_between_within_log",
                "coefficient_raw_input": -0.3138136390800163,
                "coefficient_standardized_input": -0.08765349987550977,
                "training_median_for_imputation": 0.2313528123427707,
                "training_mean": 0.3159467168726048,
                "training_sd": 0.27931704986589145,
            },
            {
                "feature": "dehi_js_divergence",
                "coefficient_raw_input": -0.38667416533077614,
                "coefficient_standardized_input": -0.05626225300291323,
                "training_median_for_imputation": 0.5906134012118127,
                "training_mean": 0.5729430161738556,
                "training_sd": 0.14550300497780685,
            },
            {
                "feature": "dehi_phase_correlation_diversity",
                "coefficient_raw_input": -0.30721164206418045,
                "coefficient_standardized_input": -0.016564999208292814,
                "training_median_for_imputation": 0.0787228708670733,
                "training_mean": 0.08973855385441377,
                "training_sd": 0.053920480021496624,
            },
            {
                "feature": "dehi_washout_fraction_diversity",
                "coefficient_raw_input": 0.16383951054237506,
                "coefficient_standardized_input": 0.016438907532156422,
                "training_median_for_imputation": 0.2269958742019876,
                "training_mean": 0.2306970516366539,
                "training_sd": 0.10033542872373696,
            },
        ],
        "normalization": {
            "method": "clip_training_P1_P99_minmax",
            "training_p01": -2.634842991844,
            "training_p99": 1.7849057061880034,
        },
    },
    "RAD": {
        "candidate_feature_count": 200,
        "active_feature_count": 16,
        "raw_intercept": 0.9337086057546836,
        "standardized_intercept": -4.440892098500626e-16,
        "feature_missing_policy": "use_corresponding_training_median_before_formula_evaluation",
        "terms": [
            {
                "feature": "original_shape_Sphericity",
                "coefficient_raw_input": -5.02890940706442,
                "coefficient_standardized_input": -0.5805018802214901,
                "training_median_for_imputation": 0.648159603576838,
                "training_mean": 0.643604569306928,
                "training_sd": 0.1154329563793738,
            },
            {
                "feature": "original_shape_SurfaceVolumeRatio",
                "coefficient_raw_input": -0.9187551034985,
                "coefficient_standardized_input": -0.2192553868635587,
                "training_median_for_imputation": 0.4573824693027917,
                "training_mean": 0.4828176894590387,
                "training_sd": 0.2386439934087578,
            },
            {
                "feature": "V_original_firstorder_Skewness",
                "coefficient_raw_input": 0.1107213199944874,
                "coefficient_standardized_input": 0.210981778334409,
                "training_median_for_imputation": -0.8520010639894218,
                "training_mean": -0.9483047204196328,
                "training_sd": 1.9055208007356972,
            },
            {
                "feature": "A_original_glrlm_LongRunLowGrayLevelEmphasis",
                "coefficient_raw_input": 0.0278944998474423,
                "coefficient_standardized_input": 0.1907306554453034,
                "training_median_for_imputation": 1.586860107765849,
                "training_mean": 3.661894391483228,
                "training_sd": 6.837572155386441,
            },
            {
                "feature": "A_original_gldm_LargeDependenceLowGrayLevelEmphasis",
                "coefficient_raw_input": 0.0029349539739008,
                "coefficient_standardized_input": 0.1792735541338711,
                "training_median_for_imputation": 38.580199768104016,
                "training_mean": 52.79172214767256,
                "training_sd": 61.08223697136777,
            },
            {
                "feature": "A_original_glszm_SizeZoneNonUniformityNormalized",
                "coefficient_raw_input": 1.6999428802643544,
                "coefficient_standardized_input": 0.1580182231654493,
                "training_median_for_imputation": 0.2622222222222222,
                "training_mean": 0.2814338980142251,
                "training_sd": 0.0929550192538682,
            },
            {
                "feature": "original_shape_Maximum2DDiameterColumn",
                "coefficient_raw_input": 0.0055981078313918,
                "coefficient_standardized_input": 0.1513351295991851,
                "training_median_for_imputation": 29.120439557122072,
                "training_mean": 36.78928879102388,
                "training_sd": 27.033264481001986,
            },
            {
                "feature": "A_original_gldm_DependenceNonUniformityNormalized",
                "coefficient_raw_input": 1.7454036262627863,
                "coefficient_standardized_input": 0.106957996192577,
                "training_median_for_imputation": 0.0525517897963043,
                "training_mean": 0.0722003558201393,
                "training_sd": 0.0612798063343049,
            },
            {
                "feature": "V_original_gldm_SmallDependenceLowGrayLevelEmphasis",
                "coefficient_raw_input": -37.64633816187322,
                "coefficient_standardized_input": -0.0945845936287434,
                "training_median_for_imputation": 0.001995121278217,
                "training_mean": 0.0026890542692864,
                "training_sd": 0.0025124513630527,
            },
            {
                "feature": "A_original_glszm_GrayLevelNonUniformityNormalized",
                "coefficient_raw_input": 0.4954078981284099,
                "coefficient_standardized_input": 0.0937791006542188,
                "training_median_for_imputation": 0.379874376024609,
                "training_mean": 0.4180486310275409,
                "training_sd": 0.1892967411470523,
            },
            {
                "feature": "V_original_glcm_Correlation",
                "coefficient_raw_input": -0.5054239043016421,
                "coefficient_standardized_input": -0.0923719775422027,
                "training_median_for_imputation": 0.588255289986905,
                "training_mean": 0.570740911475233,
                "training_sd": 0.1827613944572637,
            },
            {
                "feature": "V_original_glrlm_RunPercentage",
                "coefficient_raw_input": -0.5100287289227232,
                "coefficient_standardized_input": -0.0620629678784012,
                "training_median_for_imputation": 0.3535968284571636,
                "training_mean": 0.3621577637154653,
                "training_sd": 0.121685239201113,
            },
            {
                "feature": "V_original_glcm_Idn",
                "coefficient_raw_input": 2.102654523949822,
                "coefficient_standardized_input": 0.0464831735248177,
                "training_median_for_imputation": 0.960191626715422,
                "training_mean": 0.9591367801126855,
                "training_sd": 0.022106900109058,
            },
            {
                "feature": "V_original_ngtdm_Busyness",
                "coefficient_raw_input": 7.565195768450137e-05,
                "coefficient_standardized_input": 0.0387162663265195,
                "training_median_for_imputation": 18.18547458065475,
                "training_mean": 106.94782985044225,
                "training_sd": 511.7682015313041,
            },
            {
                "feature": "V_original_glrlm_GrayLevelNonUniformityNormalized",
                "coefficient_raw_input": 0.1065002703482386,
                "coefficient_standardized_input": 0.0145773709651819,
                "training_median_for_imputation": 0.4193790318151146,
                "training_mean": 0.4268552807810953,
                "training_sd": 0.1368763752196712,
            },
            {
                "feature": "V_original_glcm_Idmn",
                "coefficient_raw_input": 0.0820485300948441,
                "coefficient_standardized_input": 0.0008689049480185,
                "training_median_for_imputation": 0.9905489092397974,
                "training_mean": 0.9878499711024192,
                "training_sd": 0.0105901342414565,
            },
        ],
        "normalization": {
            "method": "clip_training_P1_P99_minmax",
            "training_p01": -1.6055000053,
            "training_p99": 2.9455285351600007,
        },
    },
}

CLINICAL = {
    "size": {
        "transformation": "log1p",
        "training_mean_transformed": 3.4091682552761373,
        "training_sd_transformed": 0.6364316436615767,
    },
    "SFI": {
        "transformation": "log1p",
        "training_mean_transformed": 5.372254497141533,
        "training_sd_transformed": 0.67460689261669,
    },
    "TFI": {
        "transformation": "none",
        "training_mean_transformed": 500.548318096333,
        "training_sd_transformed": 255.76578342062763,
    },
    "LDH": {
        "transformation": "log1p",
        "training_mean_transformed": 5.202525775270802,
        "training_sd_transformed": 0.31846681068823895,
    },
    "AST": {
        "transformation": "log1p",
        "training_mean_transformed": 3.2315722189353338,
        "training_sd_transformed": 0.6051650878996147,
    },
    "IFI": {
        "transformation": "log1p",
        "training_mean_transformed": 1.8823123314608072,
        "training_sd_transformed": 0.5698124532421897,
    },
    "RIF%": {
        "transformation": "log1p",
        "training_mean_transformed": 0.854447852726564,
        "training_sd_transformed": 0.34078128623178744,
    },
}

PRIOR = {
    "artifact_schema": "pnet-ith.aggregate-prior.v1",
    "source_prior_schema_version": 2,
    "feature_names": ["arterial_intensity", "arterial_local_sd", "arterial_gradient"],
    "centroid_labels": [1, 2, 3],
    "centroids": [
        [0.5099999904632568, 0.04277978464961052, 0.9906561374664307],
        [0.6675000190734863, 0.038093049079179764, 0.8726218342781067],
        [0.9049999713897705, 0.04897499457001686, 1.0915222764015198],
    ],
    "n_training_cases": 100,
    "aggregation": "coordinatewise_median_of_pooled_points_by_phenotype",
    "prior_role": "arterial_centroid_initialization_only",
    "source_sha256": "6eae3fbb96191cc67ab15f073f0a5817a5872d03969cdb07dd7acc1988df7839",
    "image_feature_preprocessing": {
        "arterial_hu_transform": "clip((HU+160)/400,0,1)",
        "local_sd_window": [3, 3, 3],
        "local_sd_ddof": 0,
        "gradient": "sqrt(sum(sobel(normalized_arterial,axis)**2_for_axis_0_1_2))",
        "boundary_mode": "reflect",
        "centroid_space": "fixed_normalized_image_features_before_per_tumor_robust_scaling",
    },
    "clustering_specification": {
        "n_habitats": 3,
        "min_cluster_fraction": 0.05,
        "max_iterations": 100,
        "relative_tolerance": 0.0001,
        "random_state": 42,
        "block_normalization": "robust_scale_then_divide_by_sqrt_retained_dimension",
    },
    "manuscript_reproduction_verified": False,
}

CASE_SPEC = {
    "artifact_schema": "pnet-ith.real-case-contract.v1",
    "purpose": "individual_case_imaging_inference_demonstration",
    "case_count": {"minimum": 1, "maximum": 3},
    "data_origin": "existing_real_cases",
    "case_inclusion_status": "one_deidentified_registered_imaging_case_included",
    "required_files": ["arterial.nii.gz", "venous.nii.gz", "tumor_roi.nii.gz"],
    "public_case_naming": "case_001_to_case_003",
    "image_intensity": "preserve_real_CT_HU_before_pipeline_transform",
    "roi_labels": {"background": 0, "tumor": 2},
    "geometry_requirements": [
        "matching_shape",
        "matching_spacing",
        "consistent_orientation",
        "matching_affine",
        "consistent_qform_sform",
        "visual_anatomical_alignment_review",
    ],
    "deidentification_review": [
        "neutral_filenames",
        "NIfTI_free_text",
        "NIfTI_extensions",
        "sidecars",
        "embedded_image_labels",
    ],
    "source_mapping_location": "local_only_outside_public_repository",
    "clinical_inputs": "not_inferred_from_example_images_and_not_fabricated",
    "training_included": False,
    "experiments_included": False,
    "synthetic_data_included": False,
    "case_performance_claims": False,
}

CLASSES = ("G1", "G2", "G3/NEC")


def score_features(features, formulas=None, names=None):
    """Evaluate locked formulas on actual measured feature values."""
    if not isinstance(features, dict):
        raise ValueError("features must be a dictionary of measured feature names and values")
    output = {}
    formulas = SCORES if formulas is None else formulas
    for name in names or formulas.keys():
        spec = formulas[name]
        terms = [spec["raw_intercept"]]
        for term in spec["terms"]:
            key = term["feature"]
            if key not in features:
                raise ValueError(f"Missing measured feature: {key}")
            value = features[key]
            if value is None:
                value = term["training_median_for_imputation"]
            terms.append(term["coefficient_raw_input"] * finite(value, key))
        raw = math.fsum(terms)
        lower = spec["normalization"]["training_p01"]
        upper = spec["normalization"]["training_p99"]
        output[name] = {
            "raw": raw,
            "scaled": max(0.0, min(1.0, (raw - lower) / (upper - lower))),
        }
    return output


def finite(value, name):
    if (
        isinstance(value, (bool, str))
        or not isinstance(value, numbers.Real)
        or not math.isfinite(value)
    ):
        raise ValueError(f"{name} requires a finite numeric value")
    return float(value)


def clinical_value(name, inputs, transforms=None):
    value = finite(inputs.get(name), name)
    spec = (CLINICAL if transforms is None else transforms)[name]
    if spec["transformation"] == "log1p":
        if value <= -1:
            raise ValueError(f"{name} is outside the log1p domain")
        value = math.log1p(value)
    return (value - spec["training_mean_transformed"]) / spec["training_sd_transformed"]


def probabilities(estimator, component, values, classes):
    order = component.get("input_order")
    if not isinstance(order, list):
        raise RuntimeError("Frozen component input order has not been exported")
    if list(estimator.classes_) != classes:
        raise ValueError("Estimator class order differs from the frozen specification")
    output = estimator.predict_proba([[values[name] for name in order]])[0]
    result = [float(value) for value in output]
    if len(result) != len(classes) or any(
        not math.isfinite(x) or not 0 <= x <= 1 for x in result
    ):
        raise ValueError("Estimator returned invalid probabilities")
    return result


def load_weights(weights=None):
    """Load only the selected-estimator release format, not a research run."""
    path = Path(weights) if weights else Path(__file__).with_name("weights.joblib")
    if not path.is_file():
        raise FileNotFoundError("Missing frozen estimator bundle: weights.joblib")
    import joblib

    bundle = joblib.load(path)
    if (
        not isinstance(bundle, dict)
        or bundle.get("artifact_schema") != "pnet-ith.frozen-estimators.v1"
    ):
        raise ValueError("Expected the selected-estimator inference bundle")
    return bundle


def input_scores(inputs, formulas=None, names=None):
    """Use measured features or explicitly supplied normalized scores."""
    names = names or ("THI", "RAD")
    if "scores" in inputs:
        output = {}
        for name in names:
            value = finite(inputs["scores"].get(name), name)
            if not 0 <= value <= 1:
                raise ValueError(f"Precomputed {name} must be on the clipped 0–1 scale")
            output[name] = {"raw": None, "scaled": value}
        return output
    if "features" not in inputs:
        raise ValueError("Provide measured features or precomputed 0–1 scores")
    return score_features(inputs["features"], formulas, names)


def predict(model, inputs, weights=None):
    """Apply a selected model using one reviewed shared estimator bundle.

    Bundle format: {artifact_schema: 'pnet-ith.frozen-estimators.v1', models:
    {model8: [three fitted estimators], model9: [[four], [two]]}}.
    Original research bundles do not satisfy this inference-only format.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    if not isinstance(inputs, dict):
        raise ValueError("Inference input must be a JSON object")
    bundle = load_weights(weights)
    if model not in bundle["models"]:
        raise ValueError(f"The selected weights do not contain {model}")
    state = bundle["models"][model]
    spec = MODELS[model]
    components = (
        spec["components"]
        if model == "model8"
        else [item for boundary in spec["boundaries"] for item in boundary["components"]]
    )
    for component in components:
        if not isinstance(component.get("input_order"), list):
            raise RuntimeError("The frozen component input order is not yet available")
    scores = input_scores(inputs, bundle.get("score_formulas"))
    values = {name: item["scaled"] for name, item in scores.items()}
    required = {name for component in components for name in component["input_order"]} - set(
        values
    )
    for name in required:
        values[name] = clinical_value(
            name, inputs.get("clinical", {}), bundle.get("clinical_transforms")
        )
    if model == "model8":
        if len(state) != len(components):
            raise ValueError("Wrong number of frozen Model 8 components")
        vectors = [
            probabilities(estimator, component, values, [0, 1, 2])
            for estimator, component in zip(state, components)
        ]
        p = [
            max(
                1e-10,
                min(
                    1.0,
                    math.fsum(
                        component["weight"] * vector[k]
                        for component, vector in zip(components, vectors)
                    ),
                ),
            )
            for k in range(3)
        ]
        total = math.fsum(p)
        p = [value / total for value in p]
        grade = max(range(3), key=lambda k: p[k])
    else:
        if len(state) != 2:
            raise ValueError("Model 9 requires two frozen boundaries")
        cumulative = []
        for estimators, boundary in zip(state, spec["boundaries"]):
            if len(estimators) != len(boundary["components"]):
                raise ValueError("Wrong number of frozen boundary components")
            probability = math.fsum(
                component["weight"] * probabilities(estimator, component, values, [0, 1])[1]
                for estimator, component in zip(estimators, boundary["components"])
            )
            cumulative.append(probability)
        q1, q2 = cumulative
        if q2 > q1:
            q1 = q2 = (q1 + q2) / 2
        p = [max(0.0, min(1.0, value)) for value in (1 - q1, q1 - q2, q2)]
        total = math.fsum(p)
        p = [value / total for value in p]
        grade = bisect.bisect_right(
            [spec["thresholds"]["lower_threshold"], spec["thresholds"]["upper_threshold"]],
            p[1] + 2 * p[2],
        )
    return {
        "model": model,
        "model_status": "exploratory" if model == "model9" else "internally_selected",
        "scores": scores,
        "probabilities": dict(zip(CLASSES, p)),
        "risk": p[1] + 2 * p[2],
        "grade": CLASSES[grade],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--list-models", action="store_true")
    action.add_argument("--describe", choices=MODELS)
    action.add_argument("--model", choices=MODELS)
    action.add_argument("--score-only", action="store_true")
    action.add_argument(
        "--thi-only",
        action="store_true",
        help="Compute THI without requiring radiomics or clinical values",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--input",
        type=Path,
        help="Actual measured features or normalized scores, and complete clinical inputs, in JSON",
    )
    source.add_argument(
        "--case", type=Path, help="Registered arterial/venous CT and ROI case directory"
    )
    parser.add_argument(
        "--clinical",
        type=Path,
        help="Actual complete clinical variables in JSON for image-to-grade inference",
    )
    parser.add_argument("--weights", type=Path, help="Shared frozen estimator bundle")
    parser.add_argument(
        "--output-dir", type=Path, help="Write habitat maps, measured features and result.json"
    )
    args = parser.parse_args()
    try:
        if args.list_models:
            result = {
                name: {
                    "weights_file_present": Path(__file__)
                    .with_name("weights.joblib")
                    .is_file(),
                    "description": spec.get("algorithm", spec.get("status")),
                }
                for name, spec in MODELS.items()
            }
        elif args.describe:
            result = MODELS[args.describe]
        else:
            if args.input is None and args.case is None:
                parser.error("--input or --case is required for inference")
            if args.case is not None and args.model and args.clinical is None:
                parser.error(
                    "Image-to-grade inference requires --clinical with the actual complete clinical inputs"
                )
            if args.case is not None:
                from preprocessing import prepare_case

                features = prepare_case(
                    args.case,
                    PRIOR,
                    include_radiomics=not args.thi_only,
                    output_dir=args.output_dir,
                )
                inputs = {"features": features}
                if args.clinical:
                    inputs["clinical"] = json.loads(args.clinical.read_text(encoding="utf-8"))
            else:
                inputs = json.loads(args.input.read_text(encoding="utf-8"))
                if args.clinical:
                    inputs["clinical"] = json.loads(args.clinical.read_text(encoding="utf-8"))
            if args.score_only or args.thi_only:
                formulas = (
                    load_weights(args.weights).get("score_formulas")
                    if args.weights or Path(__file__).with_name("weights.joblib").is_file()
                    else None
                )
                result = {
                    "scores": input_scores(
                        inputs, formulas, ("THI",) if args.thi_only else None
                    )
                }
            else:
                result = predict(args.model, inputs, args.weights)
            if args.output_dir:
                from preprocessing.io import save_json

                save_json(result, args.output_dir / "result.json")
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
    except (ValueError, KeyError, RuntimeError, FileNotFoundError, ImportError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
