"""
Tests for the churn app.

Run everything:      python -m pytest test_churn.py -v
Or without pytest:   python test_churn.py

The regression tests at the bottom pin the specific bugs that were in the
original code, so they cannot silently come back.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import churn_core as core

pipeline, schema, metrics = core.load_artifacts()
THRESHOLD = schema["threshold"]


def sample_frame(n=4):
    """A small upload using the exact column names from the README."""
    rows = [
        {"Gender": "Female", "Age": 42, "CustomerId": 15634602, "Tenure": 2,
         "Balance": 1115.0, "Satisfaction Score": 2, "EstimatedSalary": 101348.88},
        {"Gender": "Male", "Age": 36, "CustomerId": 15647311, "Tenure": 1,
         "Balance": 0.0, "Satisfaction Score": 1, "EstimatedSalary": 112542.58},
        {"Gender": "Male", "Age": 25, "CustomerId": 15619304, "Tenure": 8,
         "Balance": 125510.0, "Satisfaction Score": 5, "EstimatedSalary": 79084.10},
        {"Gender": "Female", "Age": 70, "CustomerId": 15701354, "Tenure": 4,
         "Balance": 0.0, "Satisfaction Score": 1, "EstimatedSalary": 93826.63},
    ]
    return pd.DataFrame(rows[:n])


# ---------------------------------------------------------------------------
# Core behaviour
# ---------------------------------------------------------------------------
def test_pipeline_is_self_contained():
    """The pipeline must own its own scaling -- the app must not pre-scale."""
    names = [step[0] for step in pipeline.steps]
    assert "prep" in names, f"no preprocessing step in pipeline: {names}"
    assert "clf" in names


def test_predictions_are_valid_probabilities():
    X, missing, _ = core.build_features(sample_frame(), schema)
    assert missing == [], f"README's own sample CSV should map cleanly, missing={missing}"
    preds, probs = core.predict(pipeline, X, THRESHOLD)
    assert len(preds) == len(probs) == 4
    assert np.all((probs >= 0) & (probs <= 1))
    assert set(np.unique(preds)) <= {0, 1}


def test_predictions_are_not_all_identical():
    X, _, _ = core.build_features(sample_frame(), schema)
    _, probs = core.predict(pipeline, X, THRESHOLD)
    assert probs.std() > 0.01, f"model gives near-identical output for everyone: {probs}"


def test_labels_respect_the_threshold():
    X, _, _ = core.build_features(sample_frame(), schema)
    preds, probs = core.predict(pipeline, X, THRESHOLD)
    assert np.array_equal(preds, (probs >= THRESHOLD).astype(int))


# ---------------------------------------------------------------------------
# Regression: every feature must actually influence the prediction.
# The original model returned 50.62% for a balance of 0 and of 250,000.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "feature,values",
    [
        ("Age", [18, 30, 45, 60, 80]),
        ("Balance", [0.0, 50_000.0, 120_000.0, 200_000.0]),
        ("Tenure", [0, 3, 6, 10]),
        ("EstimatedSalary", [12_000.0, 60_000.0, 150_000.0]),
    ],
)
def test_feature_moves_the_prediction(feature, values):
    if feature not in schema["features"]:
        pytest.skip(f"{feature} not in the trained feature set")
    base = sample_frame(1)
    frame = pd.concat([base] * len(values), ignore_index=True)
    frame[feature] = values
    X, _, _ = core.build_features(frame, schema)
    _, probs = core.predict(pipeline, X, THRESHOLD)
    spread = probs.max() - probs.min()
    assert spread > 0.01, (
        f"changing {feature} across {values} moved the probability by only "
        f"{spread:.4%} -- the feature is being ignored"
    )


def test_gender_changes_the_prediction():
    frame = pd.concat([sample_frame(1)] * 2, ignore_index=True)
    frame["Gender"] = ["Female", "Male"]
    X, _, _ = core.build_features(frame, schema)
    _, probs = core.predict(pipeline, X, THRESHOLD)
    assert abs(probs[0] - probs[1]) > 1e-6


# ---------------------------------------------------------------------------
# Regression: column matching
# ---------------------------------------------------------------------------
def test_exact_names_match():
    mapping, missing = core.match_columns(
        ["Gender", "Age", "Tenure", "Balance", "Satisfaction Score", "EstimatedSalary"],
        schema["features"],
    )
    assert missing == []
    assert all(mapping[k] == k for k in mapping)


def test_case_and_punctuation_variants_match():
    mapping, missing = core.match_columns(
        ["gender", "AGE", "tenure", "balance", "satisfaction_score", "estimated salary"],
        schema["features"],
    )
    assert missing == [], f"failed to match tolerant spellings: {missing}"
    assert mapping["Age"] == "AGE"
    assert mapping["Satisfaction Score"] == "satisfaction_score"


def test_no_two_features_claim_the_same_column():
    mapping, _ = core.match_columns(
        ["Age", "Average Balance", "Gender", "Tenure"], schema["features"]
    )
    assert len(set(mapping.values())) == len(mapping), f"duplicate source column: {mapping}"


def test_unrelated_columns_do_not_match():
    """A CSV of nonsense should report missing, not invent matches."""
    _, missing = core.match_columns(
        ["foo", "bar", "baz", "quux"], schema["features"]
    )
    assert set(missing) == set(schema["features"]), (
        f"fuzzy matcher accepted unrelated columns; only missing={missing}"
    )


# ---------------------------------------------------------------------------
# Regression: missing columns use TRAINING defaults, not batch statistics.
# ---------------------------------------------------------------------------
def test_missing_column_uses_training_default():
    frame = sample_frame().drop(columns=["Balance"])
    X, missing, _ = core.build_features(frame, schema)
    assert "Balance" in missing
    assert (X["Balance"] == schema["defaults"]["Balance"]).all()


def test_prediction_does_not_depend_on_other_rows_in_the_batch():
    """
    The original app fit a SimpleImputer on the uploaded file, so one row's
    prediction changed depending on which other rows were uploaded with it.
    """
    one = sample_frame(1)
    one_with_nan = one.copy()
    one_with_nan.loc[0, "Balance"] = np.nan

    batch = sample_frame(4)
    batch.loc[0, "Balance"] = np.nan
    batch.loc[1, "Balance"] = 999_999.0  # would drag a batch-fitted mean

    X_solo, _, _ = core.build_features(one_with_nan, schema)
    X_batch, _, _ = core.build_features(batch, schema)

    _, p_solo = core.predict(pipeline, X_solo, THRESHOLD)
    _, p_batch = core.predict(pipeline, X_batch, THRESHOLD)
    assert p_solo[0] == pytest.approx(p_batch[0], abs=1e-9), (
        "row 0's prediction changed because of other rows in the same upload"
    )


def test_all_nan_column_does_not_crash():
    """SimpleImputer used to drop an all-NaN column, breaking the feature count."""
    frame = sample_frame()
    frame["Balance"] = np.nan
    X, _, _ = core.build_features(frame, schema)
    preds, probs = core.predict(pipeline, X, THRESHOLD)
    assert len(preds) == len(frame)
    assert np.all(np.isfinite(probs))


def test_single_row_upload_works():
    X, _, _ = core.build_features(sample_frame(1), schema)
    preds, probs = core.predict(pipeline, X, THRESHOLD)
    assert len(preds) == 1 and np.isfinite(probs[0])


# ---------------------------------------------------------------------------
# Regression: gender spellings
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "written,expected", [("female", "Female"), ("F", "Female"), ("MALE", "Male"),
                         ("m", "Male"), ("0", "Female"), ("1", "Male"), (0, "Female")],
)
def test_gender_spellings_normalise(written, expected):
    out = core.normalise_gender(pd.Series([written]), schema["categories"]["Gender"])
    assert out.iloc[0] == expected


def test_lowercase_gender_predicts_same_as_titlecase():
    """The original mapped only 'Female'/'Male'; 'female' silently became NaN."""
    a = sample_frame(2)
    b = a.copy()
    b["Gender"] = b["Gender"].str.lower()
    Xa, _, _ = core.build_features(a, schema)
    Xb, _, _ = core.build_features(b, schema)
    _, pa = core.predict(pipeline, Xa, THRESHOLD)
    _, pb = core.predict(pipeline, Xb, THRESHOLD)
    assert np.allclose(pa, pb)


# ---------------------------------------------------------------------------
# Regression: balance tiers
# ---------------------------------------------------------------------------
def test_zero_balance_is_kept_in_a_tier():
    """pd.cut(bins=[0, ...]) used to turn every zero balance into NaN."""
    tiers = core.balance_tier(pd.Series([0.0, 10.0, 60_000.0, 150_000.0]))
    assert tiers.notna().all(), f"balances dropped out of the chart: {list(tiers)}"
    assert tiers.iloc[0] == "Zero"


def test_very_large_balance_is_kept():
    tiers = core.balance_tier(pd.Series([250_000.0, 1_000_000.0]))
    assert tiers.notna().all(), "balances above the top bin were dropped"


# ---------------------------------------------------------------------------
# Regression: metrics are measured, not invented
# ---------------------------------------------------------------------------
def test_metrics_are_real_and_complete():
    assert metrics, "model_metrics.json missing -- run train_model.py"
    for name, row in metrics.items():
        for key in ("Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"):
            assert key in row, f"{name} missing {key}"
            assert 0.0 <= row[key] <= 1.0


def test_no_leaky_columns_in_feature_set():
    leaky = {"Complain", "CustomerId", "RowNumber", "Surname"}
    overlap = leaky & set(schema["features"])
    assert not overlap, f"leaky/identifier columns in the model: {overlap}"


def test_model_beats_predicting_the_majority_class():
    best_auc = max(row["ROC AUC"] for row in metrics.values())
    assert best_auc > 0.65, f"model barely beats chance (AUC={best_auc})"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "--tb=short"]))
