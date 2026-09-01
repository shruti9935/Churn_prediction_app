"""
Prediction logic, kept free of Streamlit so it can be unit-tested directly.

The rule this module enforces: the app never transforms features. It only
selects and renames columns, then hands raw values to the pipeline, which owns
imputation, encoding and scaling. Anything else reintroduces train/serve skew.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

PIPELINE_PATH = "churn_pipeline.pkl"
SCHEMA_PATH = "feature_schema.json"
METRICS_PATH = "model_metrics.json"

# Accepted spellings for the two-way Gender column, normalised to the labels the
# model was trained on. Anything else is left alone and handled by the
# pipeline's handle_unknown="ignore".
GENDER_ALIASES = {
    "f": "Female", "female": "Female", "woman": "Female", "w": "Female",
    "m": "Male", "male": "Male", "man": "Male",
    "0": "Female", "1": "Male",
}


def load_artifacts(directory="."):
    """Load pipeline + schema + metrics, with an actionable error if absent."""
    d = Path(directory)
    pipe_path = d / PIPELINE_PATH
    schema_path = d / SCHEMA_PATH
    if not pipe_path.exists() or not schema_path.exists():
        raise FileNotFoundError(
            "Model artifacts not found. Run `python train_model.py` first "
            f"(expected {PIPELINE_PATH} and {SCHEMA_PATH} in {d.resolve()})."
        )
    pipeline = joblib.load(pipe_path)
    schema = json.loads(schema_path.read_text())
    metrics_path = d / METRICS_PATH
    metrics = json.loads(metrics_path.read_text()) if metrics_path.exists() else {}
    return pipeline, schema, metrics


def _normalise(name: str) -> str:
    """Fold a column name to a comparable key: 'Satisfaction Score' -> 'satisfactionscore'."""
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


def match_columns(df_columns, targets, threshold=88):
    """
    Map each expected feature to a column in the uploaded CSV.

    Exact and normalised matches win outright; fuzzy matching is only a
    last resort and each source column can be claimed once, so 'Age' can't
    silently steal the column that 'Average Balance' should have matched.
    """
    columns = list(df_columns)
    normalised = {_normalise(c): c for c in columns}
    mapping, claimed = {}, set()

    # Pass 1: exact, then case/punctuation-insensitive.
    for target in targets:
        if target in columns and target not in claimed:
            mapping[target] = target
            claimed.add(target)
            continue
        hit = normalised.get(_normalise(target))
        if hit is not None and hit not in claimed:
            mapping[target] = hit
            claimed.add(hit)

    # Pass 2: fuzzy, only for what is still unmatched.
    unresolved = [t for t in targets if t not in mapping]
    if unresolved:
        available = [c for c in columns if c not in claimed]
        scorer = _get_scorer()
        if scorer is not None and available:
            for target in unresolved:
                best, best_score = None, 0
                for col in available:
                    score = scorer(_normalise(target), _normalise(col))
                    if score > best_score:
                        best, best_score = col, score
                if best is not None and best_score >= threshold:
                    mapping[target] = best
                    available.remove(best)

    missing = [t for t in targets if t not in mapping]
    return mapping, missing


def _get_scorer():
    """rapidfuzz if available, else fuzzywuzzy, else no fuzzy matching at all."""
    try:
        from rapidfuzz.fuzz import ratio

        return ratio
    except ImportError:
        pass
    try:
        from fuzzywuzzy.fuzz import ratio

        return ratio
    except ImportError:
        return None


def normalise_gender(series: pd.Series, categories) -> pd.Series:
    """Map assorted gender spellings onto the trained category labels."""
    valid = {str(c).lower(): str(c) for c in categories}

    def convert(v):
        if pd.isna(v):
            return np.nan
        key = str(v).strip().lower()
        if key in valid:
            return valid[key]
        if key.endswith(".0"):  # 1.0 -> "1"
            key = key[:-2]
        return GENDER_ALIASES.get(key, str(v).strip())

    return series.map(convert)


def build_features(data: pd.DataFrame, schema: dict):
    """
    Select the model's features out of an arbitrary uploaded dataframe.

    Returns (X, missing, mapping). Columns absent from the upload are filled
    with training-set defaults from the schema -- never with statistics
    computed from the uploaded file itself, which would make a prediction
    depend on whichever other rows happened to be in the same batch.
    """
    features = schema["features"]
    mapping, missing = match_columns(data.columns, features)

    X = pd.DataFrame(index=data.index)
    for col in features:
        if col in mapping:
            X[col] = data[mapping[col]]
        else:
            X[col] = schema["defaults"][col]

    for col in schema["numeric"]:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in schema["categorical"]:
        cats = schema["categories"].get(col, [])
        if col.lower() == "gender":
            X[col] = normalise_gender(X[col], cats)
        else:
            X[col] = X[col].astype("object")

    return X[features], missing, mapping


def predict(pipeline, X: pd.DataFrame, threshold=0.5):
    """Return (labels, probabilities) using the tuned decision threshold."""
    probs = pipeline.predict_proba(X)[:, 1]
    return (probs >= threshold).astype(int), probs


def risk_band(prob: float) -> str:
    if prob >= 0.60:
        return "High"
    if prob >= 0.35:
        return "Medium"
    return "Low"


def balance_tier(series: pd.Series) -> pd.Series:
    """
    Bucket balances for the insight chart.

    A zero balance is a real and very common value in this dataset, so the
    lowest bin must be closed on the left, and the top bin must be unbounded
    or every large balance silently drops out of the chart.
    """
    return pd.cut(
        series,
        bins=[-np.inf, 0, 50_000, 100_000, np.inf],
        labels=["Zero", "Low", "Mid", "High"],
    )
