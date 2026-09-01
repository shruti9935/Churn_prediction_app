"""
Train the churn model.

This is the TRAINING script. The Streamlit app lives in churn_app.py.

Everything the model needs at prediction time -- imputation, categorical
encoding and scaling -- is baked into a single sklearn Pipeline. The app never
transforms anything itself, so training and serving cannot drift apart.

Usage:
    python train_model.py                       # basic feature set (default)
    python train_model.py --rich                # adds the stronger columns
    python train_model.py --data path/to.csv
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Exited"

# 'Complain' is recorded *after* a customer churns: it correlates with the
# target at r=0.996 and any model trained on it scores ~99.8% while being
# useless in production. 'CustomerId'/'RowNumber'/'Surname' are identifiers.
# Never move anything from this list into a feature set.
LEAKY_OR_ID_COLUMNS = ["Complain", "CustomerId", "RowNumber", "Surname"]

# Kept identical to the CSV contract documented in the README.
BASIC_NUMERIC = ["Age", "Tenure", "Balance", "Satisfaction Score", "EstimatedSalary"]
BASIC_CATEGORICAL = ["Gender"]

# Genuinely predictive columns the original feature set left on the table.
RICH_NUMERIC = BASIC_NUMERIC + [
    "CreditScore",
    "NumOfProducts",
    "IsActiveMember",
    "HasCrCard",
]
RICH_CATEGORICAL = BASIC_CATEGORICAL + ["Geography"]


def build_pipeline(estimator, numeric, categorical, seed=42):
    """Impute -> encode -> scale -> SMOTE -> estimator, as one fitted object."""
    preprocessor = ColumnTransformer(
        [
            (
                "num",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", drop="if_binary"),
                        ),
                    ]
                ),
                categorical,
            ),
        ],
        remainder="drop",
    )
    # SMOTE sits inside the pipeline so it only ever sees training folds.
    return ImbPipeline(
        [
            ("prep", preprocessor),
            ("smote", SMOTE(random_state=seed)),
            ("clf", estimator),
        ]
    )


def evaluate(pipe, X_test, y_test, threshold=0.5):
    probs = pipe.predict_proba(X_test)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "Accuracy": round(float(accuracy_score(y_test, preds)), 4),
        "Precision": round(float(precision_score(y_test, preds, zero_division=0)), 4),
        "Recall": round(float(recall_score(y_test, preds, zero_division=0)), 4),
        "F1 Score": round(float(f1_score(y_test, preds, zero_division=0)), 4),
        "ROC AUC": round(float(roc_auc_score(y_test, probs)), 4),
    }


def best_threshold(pipe, X, y):
    """Pick the probability cut-off that maximises F1 on held-out data."""
    probs = pipe.predict_proba(X)[:, 1]
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y, (probs >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="Customer-Churn-Records.csv")
    parser.add_argument("--rich", action="store_true", help="use the extended feature set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    csv = Path(args.data)
    if not csv.exists():
        raise SystemExit(
            f"Dataset not found: {csv}\n"
            "Download 'Customer-Churn-Records.csv' (Kaggle: Bank Customer Churn) "
            "into this folder, or pass --data /path/to/file.csv"
        )

    df = pd.read_csv(csv)
    if TARGET not in df.columns:
        raise SystemExit(f"'{TARGET}' column missing from {csv}")

    numeric = RICH_NUMERIC if args.rich else BASIC_NUMERIC
    categorical = RICH_CATEGORICAL if args.rich else BASIC_CATEGORICAL
    features = numeric + categorical

    missing = [c for c in features if c not in df.columns]
    if missing:
        raise SystemExit(f"Dataset is missing required columns: {missing}")

    leaked = [c for c in features if c in LEAKY_OR_ID_COLUMNS]
    if leaked:  # guard against a future edit quietly reintroducing leakage
        raise SystemExit(
            f"Refusing to train: leaky/identifier columns in feature set: {leaked}"
        )

    X = df[features].copy()
    y = df[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=args.seed, stratify=y
    )
    # A slice of train is held back purely for choosing the threshold, so the
    # test set stays untouched until the final report.
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=args.seed, stratify=y_train
    )

    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=args.seed),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, min_samples_leaf=2, random_state=args.seed, n_jobs=-1
        ),
    }
    try:
        from xgboost import XGBClassifier

        candidates["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.1,
            eval_metric="logloss",
            random_state=args.seed,
        )
    except ImportError:
        print("[info] xgboost not installed -- skipping that candidate.\n")

    print(f"Rows: {len(df)}   Churn rate: {y.mean():.1%}")
    print(f"Features: {features}\n")

    results, fitted = {}, {}
    for name, est in candidates.items():
        pipe = build_pipeline(est, numeric, categorical, args.seed)
        pipe.fit(X_fit, y_fit)
        thr = best_threshold(pipe, X_val, y_val)
        pipe.fit(X_train, y_train)  # refit on all training data
        results[name] = evaluate(pipe, X_test, y_test, thr)
        results[name]["Threshold"] = round(thr, 3)
        fitted[name] = pipe
        print(f"{name:<22} {results[name]}")

    winner = max(results, key=lambda k: results[k]["ROC AUC"])
    print(f"\nSelected: {winner} (highest ROC AUC)")

    pipe = fitted[winner]
    threshold = results[winner]["Threshold"]

    # Defaults the app falls back to when an uploaded CSV omits a column.
    # Medians/modes from the TRAINING split only -- never from user uploads.
    defaults = {}
    for col in numeric:
        defaults[col] = float(np.round(X_train[col].median(), 4))
    for col in categorical:
        defaults[col] = str(X_train[col].mode().iloc[0])

    schema = {
        "features": features,
        "numeric": numeric,
        "categorical": categorical,
        "categories": {
            c: sorted(X_train[c].dropna().astype(str).unique()) for c in categorical
        },
        "defaults": defaults,
        "ranges": {c: [float(X_train[c].min()), float(X_train[c].max())] for c in numeric},
        "threshold": threshold,
        "model": winner,
        "trained_rows": int(len(X_train)),
        "excluded_columns": LEAKY_OR_ID_COLUMNS,
    }

    joblib.dump(pipe, "churn_pipeline.pkl")
    Path("model_metrics.json").write_text(json.dumps(results, indent=2))
    Path("feature_schema.json").write_text(json.dumps(schema, indent=2))
    print("\nWrote churn_pipeline.pkl, model_metrics.json, feature_schema.json")


if __name__ == "__main__":
    main()
