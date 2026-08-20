"""Fold-local sklearn pipelines and fixed Step 4 parameter grids."""

from __future__ import annotations

from typing import Any, Mapping

from sklearn.base import RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


MODEL_NAMES = ("OLS", "Ridge", "Lasso", "RandomForest", "XGBoost")
MODEL_NAME_MAP = {
    "ols": "OLS",
    "ridge": "Ridge",
    "lasso": "Lasso",
    "random_forest": "RandomForest",
    "randomforest": "RandomForest",
    "xgboost": "XGBoost",
}


def configured_model_names(config: Mapping[str, Any]) -> list[str]:
    """Normalize the existing candidate_models config without adding models."""
    names = [MODEL_NAME_MAP[str(name).casefold()] for name in config["candidate_models"]]
    if tuple(names) != MODEL_NAMES:
        raise ValueError(f"candidate_models must resolve exactly to {MODEL_NAMES}")
    return names


def build_model_pipeline(
    model_name: str,
    parameters: Mapping[str, Any],
    *,
    imputer_strategy: str,
    random_seed: int,
    n_jobs: int,
    lasso_max_iter: int,
) -> Pipeline:
    """Create a fresh, unfitted pipeline for one inner or outer fold."""
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unsupported model: {model_name}")
    if model_name == "OLS":
        estimator: RegressorMixin = LinearRegression()
    elif model_name == "Ridge":
        estimator = Ridge(alpha=float(parameters["alpha"]))
    elif model_name == "Lasso":
        estimator = Lasso(
            alpha=float(parameters["alpha"]),
            max_iter=lasso_max_iter,
            random_state=random_seed,
        )
    elif model_name == "RandomForest":
        estimator = RandomForestRegressor(
            n_estimators=int(parameters["n_estimators"]),
            max_depth=(
                None if parameters.get("max_depth") is None else int(parameters["max_depth"])
            ),
            min_samples_leaf=int(parameters["min_samples_leaf"]),
            max_features=parameters["max_features"],
            random_state=random_seed,
            n_jobs=n_jobs,
        )
    else:
        try:
            from xgboost import XGBRegressor
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "XGBoost is required for Step 4; install project requirements"
            ) from exc
        estimator = XGBRegressor(
            n_estimators=int(parameters["n_estimators"]),
            max_depth=int(parameters["max_depth"]),
            learning_rate=float(parameters["learning_rate"]),
            subsample=float(parameters["subsample"]),
            colsample_bytree=float(parameters["colsample_bytree"]),
            reg_lambda=float(parameters["reg_lambda"]),
            objective="reg:squarederror",
            eval_metric="rmse",
            tree_method="hist",
            random_state=random_seed,
            n_jobs=n_jobs,
            verbosity=0,
        )

    steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=imputer_strategy))
    ]
    if model_name in {"OLS", "Ridge", "Lasso"}:
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", estimator))
    return Pipeline(steps)


def parameter_grid(
    model_name: str,
    config: Mapping[str, Any],
    *,
    debug: bool,
) -> list[dict[str, Any]]:
    """Return the predeclared formal grid or the predeclared debug preset."""
    if model_name == "OLS":
        return [{}]
    if debug:
        return [dict(config["step4_debug_parameters"][model_name])]
    if model_name == "Ridge":
        grid = {"alpha": list(config["ridge_alpha_grid"])}
    elif model_name == "Lasso":
        grid = {"alpha": list(config["lasso_alpha_grid"])}
    elif model_name == "RandomForest":
        grid = dict(config["random_forest_grid"])
    elif model_name == "XGBoost":
        grid = dict(config["xgboost_grid"])
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    return [dict(parameters) for parameters in ParameterGrid(grid)]


def default_parameters(model_name: str, config: Mapping[str, Any]) -> dict[str, Any]:
    """Fixed fallback chosen before any validation results are observed."""
    if model_name == "OLS":
        return {}
    return dict(config["step4_debug_parameters"][model_name])
