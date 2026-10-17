"""Refit the locked THI/RAD and grading specifications on a supplied training CSV.

This script fits one fixed specification, without model search or performance
experiments. --fit-scores constructs five-fold OOF training scores from the
locked 7/16 feature sets, then fits full-training formulas for inference.
Otherwise, supply previously generated OOF THI/RAD scores and their provenance.
Clinical inputs must be complete; the historical MICE state is not recreated.
The resulting weights are a new refit, not the historical study estimators.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


def make_estimator(component):
    """Construct precisely the published component, including its fixed seed."""
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.svm import SVC

    constructors = {
        "RBF-SVC": SVC,
        "Polynomial SVM": SVC,
        "Multilayer perceptron": MLPClassifier,
        "k-nearest neighbors": KNeighborsClassifier,
    }
    parameters = dict(component["parameters"])
    if component["algorithm"] in {"RBF-SVC", "Polynomial SVM"}:
        parameters["kernel"] = "rbf" if component["algorithm"] == "RBF-SVC" else "poly"
    if "hidden_layer_sizes" in parameters:
        parameters["hidden_layer_sizes"] = tuple(parameters["hidden_layer_sizes"])
    return constructors[component["algorithm"]](**parameters)


def fit_clinical(training, features, specifications):
    """Keep locked transform choices; learn new means/SDs on training only."""
    import numpy as np
    import pandas as pd

    design = pd.DataFrame(index=training.index)
    transforms = {}
    for feature in sorted(features):
        values = pd.to_numeric(training[feature], errors="raise").to_numpy(float)
        if not np.isfinite(values).all():
            raise ValueError(f"Complete finite clinical values are required: {feature}")
        operation = specifications[feature]["transformation"]
        if operation == "log1p":
            if (values <= -1).any():
                raise ValueError(f"log1p requires values greater than -1: {feature}")
            values = np.log1p(values)
        elif operation != "none":
            raise ValueError(f"Unsupported clinical transformation: {operation}")
        mean = float(values.mean())
        scale = float(values.std(ddof=0))
        scale = scale if scale > 1e-12 else 1.0
        design[feature] = (values - mean) / scale
        transforms[feature] = {
            "transformation": operation,
            "training_mean_transformed": mean,
            "training_sd_transformed": scale,
        }
    return design, transforms


def fit_linear_score(frame, y, alpha, l1_ratio):
    """Train-only median imputation, variance filtering, scaling and ElasticNet."""
    import numpy as np
    from scipy.stats import spearmanr
    from sklearn.linear_model import ElasticNet

    values = frame.to_numpy(float)
    with np.errstate(invalid="ignore"):
        medians = np.nanmedian(values, axis=0)
    # An all-missing column cannot provide a train-fitted imputation statistic.
    usable = np.isfinite(medians)
    if not usable.any():
        raise ValueError("No observed score features remain in a training fold")
    values = np.where(np.isfinite(values), values, medians)
    keep = usable & (np.nanvar(values, axis=0) > 1e-12)
    if not keep.any():
        raise ValueError("No nonconstant score features remain in a training fold")
    names = frame.columns[keep].tolist()
    values = values[:, keep]
    medians = medians[keep]
    mean, scale = values.mean(axis=0), values.std(axis=0, ddof=0)
    scale = np.where(scale > 1e-12, scale, 1.0)
    estimator = ElasticNet(
        alpha=alpha,
        l1_ratio=l1_ratio,
        max_iter=20000,
        tol=1e-6,
        selection="cyclic",
    ).fit((values - mean) / scale, y)
    raw = estimator.predict((values - mean) / scale)
    rho = spearmanr(raw, y).statistic if np.std(raw) > 1e-12 else 0.0
    sign = -1 if np.isfinite(rho) and rho < 0 else 1
    score_mean = float((sign * raw).mean())
    score_sd = float((sign * raw).std(ddof=0))
    score_sd = score_sd if score_sd > 1e-12 else 1.0
    coefficients = sign * estimator.coef_ / (scale * score_sd)
    intercept = float(
        (sign * estimator.intercept_ - score_mean) / score_sd - mean @ coefficients
    )
    return {
        "raw_intercept": intercept,
        "standardized_intercept": float((sign * estimator.intercept_ - score_mean) / score_sd),
        "feature_missing_policy": "use_corresponding_training_median_before_formula_evaluation",
        "terms": [
            {
                "feature": name,
                "coefficient_raw_input": float(coefficient),
                "coefficient_standardized_input": float(coefficient * sd),
                "training_median_for_imputation": float(median),
                "training_mean": float(mu),
                "training_sd": float(sd),
            }
            for name, coefficient, median, mu, sd in zip(
                names, coefficients, medians, mean, scale
            )
        ],
        "active_feature_count": int(np.count_nonzero(np.abs(estimator.coef_) > 1e-10)),
    }


def apply_linear_score(frame, formula):
    import numpy as np

    score = np.full(len(frame), formula["raw_intercept"], dtype=float)
    for term in formula["terms"]:
        values = frame[term["feature"]].to_numpy(float)
        values = np.where(np.isfinite(values), values, term["training_median_for_imputation"])
        score += term["coefficient_raw_input"] * values
    return score


def fit_scores(training, y, specifications):
    """Fixed-feature OOF refit; no feature selection or hyperparameter tuning."""
    import numpy as np
    import pandas as pd
    from sklearn.model_selection import StratifiedKFold

    if np.bincount(y, minlength=3).min() < 5:
        raise ValueError("--fit-scores requires at least 5 training cases per grade")
    splits = list(StratifiedKFold(5, shuffle=True, random_state=20260720).split(training, y))
    design = pd.DataFrame(index=training.index)
    formulas = {}
    for name, (alpha, ratio) in {"THI": (0.3, 0.2), "RAD": (0.1, 0.5)}.items():
        names = [term["feature"] for term in specifications[name]["terms"]]
        missing = sorted(set(names) - set(training.columns))
        if missing:
            raise ValueError(f"Missing {name} feature columns: {missing}")
        frame = (
            training[names]
            .apply(pd.to_numeric, errors="raise")
            .replace([np.inf, -np.inf], np.nan)
        )
        oof = np.full(len(training), np.nan)
        for fit, held_out in splits:
            formula = fit_linear_score(frame.iloc[fit], y[fit], alpha, ratio)
            oof[held_out] = apply_linear_score(frame.iloc[held_out], formula)
        if not np.isfinite(oof).all():
            raise ValueError(f"Non-finite OOF {name} scores")
        lower, upper = map(float, np.quantile(oof, [0.01, 0.99]))
        if upper - lower <= 1e-12:
            raise ValueError(f"The training {name} score is constant")
        design[name] = np.clip((oof - lower) / (upper - lower), 0, 1)
        formula = fit_linear_score(frame, y, alpha, ratio)
        formula.update(
            {
                "candidate_feature_count": len(names),
                "training_strategy": "locked_feature_refit_with_training_only_five_fold_oof",
                "elastic_net": {"alpha": alpha, "l1_ratio": ratio},
                "normalization": {
                    "method": "clip_training_P1_P99_minmax",
                    "training_p01": lower,
                    "training_p99": upper,
                },
            }
        )
        formulas[name] = formula
    return design, formulas


def supplied_scores(training):
    """Require explicit OOF provenance, avoiding use of in-sample fitted scores."""
    import numpy as np
    import pandas as pd

    design = pd.DataFrame(index=training.index)
    for name, count in (("THI", 7), ("RAD", 16)):
        column = name if name in training.columns else f"{name}_score_01"
        source = f"{name}_score_source"
        if column not in training or source not in training:
            raise ValueError(
                f"Supply {name} or {name}_score_01 and {source}; alternatively use --fit-scores"
            )
        accepted = {
            "out_of_fold",
            "nested_oof",
            f"training_locked_{count}_feature_5fold_OOF_refit",
        }
        if not training[source].isin(accepted).all():
            raise ValueError(f"{source} must explicitly identify training OOF predictions")
        values = pd.to_numeric(training[column], errors="raise").to_numpy(float)
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError(f"Training {name} scores must be finite and in [0, 1]")
        design[name] = values
    return design


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, required=True, help="User-supplied merged feature/clinical CSV"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="One fitted weights.joblib output"
    )
    parser.add_argument("--models", nargs="+", choices=("model8", "model9"), default=["model8"])
    parser.add_argument(
        "--fit-scores",
        action="store_true",
        help="Fit locked 7/16-feature THI/RAD with 5-fold OOF",
    )
    parser.add_argument("--cohort-column", default="Cohort")
    parser.add_argument("--training-label", default="training")
    parser.add_argument("--grade-column", default="Grade_ordinal", help="G1=0, G2=1, G3/NEC=2")
    args = parser.parse_args()

    import joblib
    import numpy as np
    import pandas as pd
    import sklearn
    from predict import CLINICAL, MODELS, SCORES

    try:
        if args.input.resolve() == args.output.resolve():
            raise ValueError("Input and output paths must differ")
        if args.output.exists():
            raise ValueError(
                "Output already exists; choose a new path to preserve frozen weights"
            )
        data = pd.read_csv(args.input)
        if args.cohort_column not in data or args.grade_column not in data:
            raise ValueError("Input requires cohort and grade columns")
        training = data.loc[data[args.cohort_column].eq(args.training_label)].reset_index(
            drop=True
        )
        if training.empty:
            raise ValueError("No training rows matched --training-label")
        grades = pd.to_numeric(training[args.grade_column], errors="raise").to_numpy(float)
        if not np.isfinite(grades).all() or not np.isin(grades, [0, 1, 2]).all():
            raise ValueError("Grade encoding must be G1=0, G2=1, G3/NEC=2")
        y = grades.astype(int)
        if set(y) != {0, 1, 2}:
            raise ValueError("All three grades must be present in training")
        model_names = list(dict.fromkeys(args.models))
        components = {
            name: (
                [MODELS[name]["components"]]
                if name == "model8"
                else [boundary["components"] for boundary in MODELS[name]["boundaries"]]
            )
            for name in model_names
        }
        clinical = {
            feature
            for groups in components.values()
            for group in groups
            for item in group
            for feature in item["input_order"]
            if feature not in {"THI", "RAD"}
        }
        if missing := sorted(clinical - set(training.columns)):
            raise ValueError(f"Missing clinical columns: {missing}")
        design, transforms = fit_clinical(training, clinical, CLINICAL)
        if args.fit_scores:
            scores, formulas = fit_scores(training, y, SCORES)
        else:
            scores, formulas = supplied_scores(training), SCORES
        design = pd.concat([design, scores], axis=1)
        state = {}
        for name, groups in components.items():
            fitted_groups = []
            for index, group in enumerate(groups):
                target = y if name == "model8" else (y >= index + 1).astype(int)
                estimators = []
                for item in group:
                    estimator = make_estimator(item)
                    if getattr(estimator, "n_neighbors", 1) > len(training):
                        raise ValueError(
                            "Training cohort is too small for the locked kNN component"
                        )
                    estimator.fit(design[item["input_order"]].to_numpy(float), target)
                    if item["algorithm"] == "Multilayer perceptron":
                        # Forward inference reads coefs_/intercepts_, not this fit-only RNG.
                        # Omitting it avoids NumPy-version-specific RandomState pickle state.
                        estimator.__dict__.pop("_random_state", None)
                    estimators.append(estimator)
                fitted_groups.append(estimators)
            state[name] = fitted_groups[0] if name == "model8" else fitted_groups
        bundle = {
            "artifact_schema": "pnet-ith.frozen-estimators.v1",
            "models": state,
            "clinical_transforms": transforms,
            "score_formulas": formulas,
            "provenance": {
                "status": "locked_spec_refit_not_historical_frozen",
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "class_order": ["G1", "G2", "G3/NEC"],
                "training_rows": len(training),
                "fit_scope": "training cohort only; validation rows excluded",
                "clinical_missing_values": "complete inputs required; no recovered historical MICE state",
                "scores": (
                    "locked-feature five-fold OOF refit"
                    if args.fit_scores
                    else "supplied OOF scores with bundled published inference formulas"
                ),
                "model_selection": "fixed published configuration; no search or retuning",
                "model9_status": (
                    "exploratory_external_constrained_specification"
                    if "model9" in state
                    else "not_fitted"
                ),
                "scikit_learn": sklearn.__version__,
            },
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(bundle, args.output, compress=3)
        print(f"Saved {', '.join(model_names)} refit to {args.output}")
    except (ValueError, KeyError, RuntimeError, OSError) as error:
        parser.exit(2, f"{error}\n")


if __name__ == "__main__":
    main()
