"""PNET-ITH: one entry point for the frozen grading models.

Model definitions, score formulas and clinical transforms live in the
dictionaries below. PRIOR reads the learned aggregate from point_prior_n100.pkl.
Selected historical fitted estimators
are read from one shared root-level weights.joblib. Image methods and feature
extraction live in preprocessing/. No search or evaluation suite is required.
"""

import argparse
import bisect
import json
import math
import numbers
import pickle
from pathlib import Path

MODELS = {
    "model1": {
        "family": "proportional_odds",
        "algorithm": "Unpenalized proportional-odds logistic regression",
        "label": "Clinical",
        "input_order": ["size", "SFI", "TFI", "LDH", "AST", "IFI", "RIF%"],
        "parameters": {"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-07, "threshold_gap": 0.0001},
        "grade_rule": "argmax_class_probability_first_index_on_tie",
        "frozen_state": {
            "family": "proportional_odds",
            "input_order": ["size", "SFI", "TFI", "LDH", "AST", "IFI", "RIF%"],
            "coefficients": [
                0.6781608195151558,
                -0.15921230288398885,
                0.33869110625608123,
                0.20643969459924694,
                0.1607245947342274,
                -0.7323815461570489,
                0.6972593660328262,
            ],
            "thresholds": [-0.5621069133059674, 1.9680039180637459],
        },
    },
    "model2": {
        "family": "proportional_odds",
        "algorithm": "Unpenalized proportional-odds logistic regression",
        "label": "RAD",
        "input_order": ["RAD"],
        "parameters": {"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-07, "threshold_gap": 0.0001},
        "grade_rule": "argmax_class_probability_first_index_on_tie",
        "frozen_state": {
            "family": "proportional_odds",
            "input_order": ["RAD"],
            "coefficients": [4.395102166854081],
            "thresholds": [1.0100855099867476, 3.461802553015413],
        },
    },
    "model3": {
        "family": "proportional_odds",
        "algorithm": "Unpenalized proportional-odds logistic regression",
        "label": "THI",
        "input_order": ["THI"],
        "parameters": {"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-07, "threshold_gap": 0.0001},
        "grade_rule": "argmax_class_probability_first_index_on_tie",
        "frozen_state": {
            "family": "proportional_odds",
            "input_order": ["THI"],
            "coefficients": [3.606759858800513],
            "thresholds": [1.666028937846446, 4.00853111808671],
        },
    },
    "model4": {
        "family": "proportional_odds",
        "algorithm": "Unpenalized proportional-odds logistic regression",
        "label": "Clinical + RAD",
        "input_order": ["size", "SFI", "TFI", "LDH", "AST", "IFI", "RIF%", "RAD"],
        "parameters": {"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-07, "threshold_gap": 0.0001},
        "grade_rule": "argmax_class_probability_first_index_on_tie",
        "frozen_state": {
            "family": "proportional_odds",
            "input_order": ["size", "SFI", "TFI", "LDH", "AST", "IFI", "RIF%", "RAD"],
            "coefficients": [
                0.271688188291463,
                -0.18522002776823485,
                0.3165479887477539,
                0.24825524034605148,
                0.11198847984289897,
                -0.6189092170499066,
                0.5634649212767929,
                2.9528747367728054,
            ],
            "thresholds": [0.44266714831919624, 3.070427429058035],
        },
    },
    "model5": {
        "family": "proportional_odds",
        "algorithm": "Unpenalized proportional-odds logistic regression",
        "label": "Clinical + THI",
        "input_order": ["size", "SFI", "TFI", "LDH", "AST", "IFI", "RIF%", "THI"],
        "parameters": {"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-07, "threshold_gap": 0.0001},
        "grade_rule": "argmax_class_probability_first_index_on_tie",
        "frozen_state": {
            "family": "proportional_odds",
            "input_order": ["size", "SFI", "TFI", "LDH", "AST", "IFI", "RIF%", "THI"],
            "coefficients": [
                0.44006590332303797,
                -0.14707532835047415,
                0.32882757304162097,
                0.20966329059685912,
                0.16316168402199538,
                -0.6594635725909619,
                0.6419375158590858,
                2.082058293747412,
            ],
            "thresholds": [0.6688972937968303, 3.2597352396344177],
        },
    },
    "model6": {
        "family": "proportional_odds",
        "algorithm": "Unpenalized proportional-odds logistic regression",
        "label": "RAD + THI",
        "input_order": ["RAD", "THI"],
        "parameters": {"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-07, "threshold_gap": 0.0001},
        "grade_rule": "argmax_class_probability_first_index_on_tie",
        "frozen_state": {
            "family": "proportional_odds",
            "input_order": ["RAD", "THI"],
            "coefficients": [3.290066310899856, 1.8584905349256808],
            "thresholds": [1.7201446894093844, 4.223197477801283],
        },
    },
    "model7": {
        "family": "proportional_odds",
        "algorithm": "Unpenalized proportional-odds logistic regression",
        "label": "Clinical + RAD + THI",
        "input_order": ["SFI", "TFI", "IFI", "RAD", "THI"],
        "parameters": {"maxiter": 2000, "ftol": 1e-11, "gtol": 1e-07, "threshold_gap": 0.0001},
        "grade_rule": "argmax_class_probability_first_index_on_tie",
    },
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
        "algorithm": "Dual-boundary nonlinear ensemble",
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

PRIOR = pickle.loads(Path(__file__).with_name("point_prior_n100.pkl").read_bytes())

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


def model_inputs(model):
    """Return each model's required inputs in a deterministic order."""
    spec = MODELS[model]
    if "input_order" in spec:
        return list(spec["input_order"])
    components = spec.get("components") or [
        item for boundary in spec["boundaries"] for item in boundary["components"]
    ]
    return list(dict.fromkeys(name for item in components for name in item["input_order"]))


def ordinal_probabilities(state, values):
    """Apply a proportional-odds model stored as portable scalar parameters."""
    order, coefficients = state["input_order"], state["coefficients"]
    if len(order) != len(coefficients):
        raise ValueError("Ordinal coefficient count differs from its input order")
    thresholds = [finite(value, "ordinal threshold") for value in state["thresholds"]]
    if len(thresholds) != 2 or thresholds[0] >= thresholds[1]:
        raise ValueError("Ordinal models require two increasing thresholds")
    eta = math.fsum(
        finite(beta, name) * values[name] for name, beta in zip(order, coefficients)
    )

    def sigmoid(value):
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exponential = math.exp(value)
        return exponential / (1.0 + exponential)

    lower, upper = (sigmoid(cut - eta) for cut in thresholds)
    p = [max(0.0, min(1.0, value)) for value in (lower, upper - lower, 1.0 - upper)]
    total = math.fsum(p)
    return [value / total for value in p]


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

    Models 1–6 have inline scalar states. Refit bundles can supply ordinal
    scalar states or the selected sklearn estimators for Models 8/9 under
    artifact_schema 'pnet-ith.frozen-estimators.v1'.
    """
    if model not in MODELS:
        raise ValueError(f"Unknown model: {model}")
    if not isinstance(inputs, dict):
        raise ValueError("Inference input must be a JSON object")
    spec = MODELS[model]
    bundle = (
        load_weights(weights)
        if weights is not None or Path(__file__).with_name("weights.joblib").is_file()
        else {}
    )
    state = bundle.get("models", {}).get(model)
    if state is None:
        if weights is None and "frozen_state" in spec:
            state, bundle = spec["frozen_state"], {}
        else:
            raise ValueError(
                f"Train {model} with train.py, then supply its bundle with --weights"
            )
    order = model_inputs(model)
    score_names = [name for name in ("THI", "RAD") if name in order]
    scores = (
        input_scores(inputs, bundle.get("score_formulas"), score_names) if score_names else {}
    )
    values = {name: item["scaled"] for name, item in scores.items()}
    required = set(order) - set(values)
    for name in required:
        values[name] = clinical_value(
            name, inputs.get("clinical", {}), bundle.get("clinical_transforms")
        )
    if spec.get("family") == "proportional_odds":
        if state.get("family") != "proportional_odds" or state["input_order"] != order:
            raise ValueError("Ordinal state does not match the selected model specification")
        p = ordinal_probabilities(state, values)
        grade = max(range(3), key=lambda k: p[k])
    elif model == "model8":
        components = spec["components"]
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
        "model_status": (
            "fixed_specification"
            if spec.get("family") == "proportional_odds"
            else "exploratory" if model == "model9" else "internally_selected"
        ),
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
        "--prior",
        type=Path,
        help="Trusted learned prior (.pkl or .json) for THI; defaults to point_prior_n100.pkl",
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Write habitat maps, measured features and result.json"
    )
    args = parser.parse_args()
    try:
        if args.list_models:
            available = (
                set(load_weights(args.weights)["models"])
                if args.weights is not None
                or Path(__file__).with_name("weights.joblib").is_file()
                else set()
            )
            if args.weights is None:
                available.update(
                    name for name, spec in MODELS.items() if "frozen_state" in spec
                )
            result = {
                name: {
                    "weights_available": name in available,
                    "description": spec.get("algorithm", spec.get("status")),
                    "input_order": model_inputs(name),
                }
                for name, spec in MODELS.items()
            }
        elif args.describe:
            result = MODELS[args.describe]
        else:
            needed = set(model_inputs(args.model)) if args.model else {"THI", "RAD"}
            if args.thi_only:
                needed = {"THI"}
            if args.prior is not None and (args.case is None or "THI" not in needed):
                parser.error(
                    "--prior requires --case with a model or scoring action that uses THI"
                )
            if (
                args.input is None
                and args.case is None
                and not (
                    args.model and not needed.intersection({"THI", "RAD"}) and args.clinical
                )
            ):
                parser.error("--input or --case is required for inference")
            if args.case is not None and args.model and not needed.intersection({"THI", "RAD"}):
                parser.error(
                    "This model uses clinical inputs; supply --clinical or --input without --case"
                )
            if (
                args.case is not None
                and args.model
                and needed - {"THI", "RAD"}
                and args.clinical is None
            ):
                parser.error(
                    "Image-to-grade inference requires --clinical with the actual complete clinical inputs"
                )
            if args.case is not None:
                if "THI" in needed:
                    from preprocessing import prepare_case

                    features = prepare_case(
                        args.case,
                        args.prior if args.prior is not None else PRIOR,
                        include_radiomics="RAD" in needed,
                        output_dir=args.output_dir,
                    )
                else:
                    from preprocessing.radiomics import extract_radiomics

                    radiomics = extract_radiomics(args.case)
                    features = radiomics["features"]
                    if args.output_dir:
                        from preprocessing.io import save_json

                        save_json(radiomics, args.output_dir / "radiomics_features.json")
                inputs = {"features": features}
            else:
                inputs = (
                    json.loads(args.input.read_text(encoding="utf-8")) if args.input else {}
                )
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
