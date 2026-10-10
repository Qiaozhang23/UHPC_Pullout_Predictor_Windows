from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


STATIC_INPUT_COLS = [
    "Fiber shape",
    "Embedded length",
    "Angle of inclination",
    "Tensile strength",
    "Length",
    "Diameter",
    "Aspect ratio Lf/Df",
    "Fiber pull-out Specimen shape",
    "Loading rate",
    "W/B",
    "Compressive strength",
    "Curing age",
]
CATEGORICAL_COLS = ["Fiber shape", "Fiber pull-out Specimen shape"]
NUMERIC_COLS = [column for column in STATIC_INPUT_COLS if column not in CATEGORICAL_COLS]


def _one_hot_encoder() -> OneHotEncoder:
    try:
        return OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    except TypeError:
        return OneHotEncoder(sparse=False, handle_unknown="ignore")


def build_features(data: pd.DataFrame, peak_predictions: np.ndarray | None = None) -> pd.DataFrame:
    """Reproduce the feature engineering in the manuscript notebook."""
    x = data[STATIC_INPUT_COLS].copy()
    eps = 1e-8
    for column in NUMERIC_COLS:
        x[column] = pd.to_numeric(x[column], errors="coerce")

    angle = np.deg2rad(x["Angle of inclination"].astype(float))
    diameter = x["Diameter"].astype(float)
    embedded = x["Embedded length"].astype(float)
    length = x["Length"].astype(float)
    tensile = x["Tensile strength"].astype(float)
    loading_rate = x["Loading rate"].astype(float)
    compressive_strength = x["Compressive strength"].astype(float)
    water_binder = x["W/B"].astype(float)
    area = np.pi * diameter**2 / 4.0
    surface = np.pi * diameter * embedded

    x["log_loading_rate"] = np.log10(loading_rate + eps)
    x["sqrt_loading_rate"] = np.sqrt(loading_rate + eps)
    x["log_curing_age"] = np.log1p(x["Curing age"].astype(float))
    x["embedded_over_diameter"] = embedded / (diameter + eps)
    x["length_over_diameter_calc"] = length / (diameter + eps)
    x["embedded_over_length"] = embedded / (length + eps)
    x["aspect_x_strength"] = x["Aspect ratio Lf/Df"].astype(float) * tensile
    x["embedded_x_strength"] = embedded * tensile
    x["diameter_x_strength"] = diameter * tensile
    x["fiber_area"] = area
    x["fiber_area_x_strength"] = area * tensile
    x["surface_proxy"] = surface
    x["volume_proxy"] = area * embedded
    x["fc_x_wb"] = compressive_strength * water_binder
    x["fc_x_embedded"] = compressive_strength * embedded
    x["bond_capacity_proxy"] = surface * np.sqrt(np.maximum(compressive_strength, eps))
    x["tensile_area_proxy"] = area * tensile
    x["sin_angle"] = np.sin(angle)
    x["cos_angle"] = np.cos(angle)
    x["tan_angle"] = np.tan(angle)

    if peak_predictions is not None:
        peaks = np.asarray(peak_predictions, dtype=float)
        predicted_pmax = np.maximum(peaks[:, 0], eps)
        predicted_slip = np.maximum(peaks[:, 1], eps)
        x["pred_Pmax"] = predicted_pmax
        x["pred_Slip"] = predicted_slip
        x["log_pred_Pmax"] = np.log1p(predicted_pmax)
        x["log_pred_Slip"] = np.log1p(predicted_slip)
        x["pred_Pmax_over_surface"] = predicted_pmax / (surface + eps)
        x["pred_Slip_over_embedded"] = predicted_slip / (embedded + eps)
    return x


def _preprocessor(features: pd.DataFrame) -> ColumnTransformer:
    categorical = [column for column in features.columns if features[column].dtype == "object"]
    numeric = [column for column in features.columns if column not in categorical]
    return ColumnTransformer(
        [("num", StandardScaler(), numeric), ("cat", _one_hot_encoder(), categorical)],
        remainder="drop",
    )


def _physics_scale(data: pd.DataFrame) -> np.ndarray:
    diameter = pd.to_numeric(data["Diameter"], errors="coerce").to_numpy(dtype=float)
    embedded = pd.to_numeric(data["Embedded length"], errors="coerce").to_numpy(dtype=float)
    strength = pd.to_numeric(data["Compressive strength"], errors="coerce").to_numpy(dtype=float)
    eps = 1e-8
    return np.maximum(
        np.pi * np.maximum(diameter, eps) * np.maximum(embedded, eps) ** 0.25 * np.maximum(strength, eps) ** 0.75,
        eps,
    )


def _smooth_normalized_curves(curves: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    from scipy.signal import savgol_filter

    normalized = curves / (peaks[:, [0]] + 1e-8)
    normalized = np.maximum(normalized, -0.08)
    if normalized.shape[1] >= 11:
        normalized = savgol_filter(normalized, window_length=11, polyorder=3, axis=1, mode="interp")
    return np.clip(normalized, -0.08, 1.35)


def _peak_regressors() -> list:
    return [
        ExtraTreesRegressor(n_estimators=450, max_features=0.90, min_samples_leaf=1, random_state=101, n_jobs=-1),
        RandomForestRegressor(n_estimators=450, max_features=0.85, min_samples_leaf=1, random_state=102, n_jobs=-1),
        MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=520,
                learning_rate=0.025,
                max_depth=3,
                min_samples_leaf=2,
                subsample=0.90,
                random_state=103,
            )
        ),
        MultiOutputRegressor(KernelRidge(alpha=0.02, kernel="rbf", gamma=0.025)),
    ]


@dataclass
class Prediction:
    pmax: float
    slip: float
    displacement: np.ndarray
    load: np.ndarray
    warnings: list[str]


class TwoStagePulloutPredictor:
    """Physics-scaled peak ensemble plus Route-D PCA curve reconstruction."""

    model_name = "Physics-scaled ensemble + Route-D PCA"

    def fit(self, dataset_path: str | Path) -> "TwoStagePulloutPredictor":
        dataset_path = Path(dataset_path)
        static = pd.read_excel(dataset_path, sheet_name="dataset")
        curve_sheet = pd.read_excel(dataset_path, sheet_name="P-s-interpolation")
        self.displacement_ = curve_sheet.iloc[:, 0].to_numpy(dtype=float)
        curves = curve_sheet.iloc[:, 1:].to_numpy(dtype=float).T
        peaks = static[["Pmax", "Slip"]].to_numpy(dtype=float)
        if len(static) != curves.shape[0]:
            raise ValueError("The dataset and P-s-interpolation sheets contain different sample counts.")

        self.domain_ = {
            column: (float(static[column].min()), float(static[column].max())) for column in NUMERIC_COLS
        }
        self.categories_ = {
            column: sorted(static[column].dropna().astype(str).unique().tolist()) for column in CATEGORICAL_COLS
        }
        base_features = build_features(static)
        scale = _physics_scale(static)
        target = np.column_stack([np.log1p(peaks[:, 0] / scale), np.log1p(peaks[:, 1])])

        self.peak_models_ = []
        peak_components = []
        for regressor in _peak_regressors():
            model = Pipeline([("preprocessor", _preprocessor(base_features)), ("regressor", regressor)])
            model.fit(base_features, target)
            transformed = np.expm1(model.predict(base_features))
            transformed[:, 0] *= scale
            peak_components.append(transformed)
            self.peak_models_.append(model)
        fitted_peaks = np.maximum(np.mean(peak_components, axis=0), 1e-6)

        shape_features = build_features(static, fitted_peaks)
        normalized_curves = _smooth_normalized_curves(curves, fitted_peaks)
        self.shape_components_ = []
        for component_count, kind in [(8, "ETR"), (12, "ETR"), (16, "ETR"), (12, "KNN")]:
            pca = PCA(n_components=min(component_count, len(static) - 1, curves.shape[1]), random_state=230 + component_count)
            coefficients = pca.fit_transform(normalized_curves)
            if kind == "ETR":
                regressor = ExtraTreesRegressor(
                    n_estimators=420, max_features=0.80, min_samples_leaf=1, random_state=301, n_jobs=-1
                )
            else:
                regressor = KNeighborsRegressor(n_neighbors=3, weights="distance", p=1)
            model = Pipeline([("preprocessor", _preprocessor(shape_features)), ("regressor", regressor)])
            model.fit(shape_features, coefficients)
            self.shape_components_.append((pca, model))
        return self

    def _predict_peaks(self, data: pd.DataFrame) -> np.ndarray:
        features = build_features(data)
        scale = _physics_scale(data)
        predictions = []
        for model in self.peak_models_:
            prediction = np.expm1(model.predict(features))
            prediction[:, 0] *= scale
            predictions.append(prediction)
        return np.maximum(np.mean(predictions, axis=0), 1e-6)

    def domain_warnings(self, data: pd.DataFrame) -> list[str]:
        warnings = []
        for column in NUMERIC_COLS:
            value = float(data.iloc[0][column])
            low, high = self.domain_[column]
            if value < low or value > high:
                warnings.append(f"{column}: {value:g} is outside the training range [{low:g}, {high:g}].")
        for column in CATEGORICAL_COLS:
            value = str(data.iloc[0][column])
            if value not in self.categories_[column]:
                warnings.append(f"{column}: {value!r} was not present in the training dataset.")
        return warnings

    def predict(self, values: dict) -> Prediction:
        data = pd.DataFrame([{column: values[column] for column in STATIC_INPUT_COLS}])
        peaks = self._predict_peaks(data)
        shape_features = build_features(data, peaks)
        normalized_components = []
        for pca, model in self.shape_components_:
            coefficients = model.predict(shape_features)
            normalized_components.append(np.clip(pca.inverse_transform(coefficients), -0.08, 1.35))
        normalized = np.mean(normalized_components, axis=0)
        load = np.clip(normalized * peaks[:, [0]], 0.0, None)[0]
        return Prediction(
            pmax=float(peaks[0, 0]),
            slip=float(peaks[0, 1]),
            displacement=self.displacement_.copy(),
            load=load,
            warnings=self.domain_warnings(data),
        )

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, path: str | Path) -> "TwoStagePulloutPredictor":
        with Path(path).open("rb") as handle:
            model = pickle.load(handle)
        if not isinstance(model, cls):
            raise TypeError("The selected file is not a TwoStagePulloutPredictor model.")
        return model
