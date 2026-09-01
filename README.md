# Customer Churn Prediction App

An interactive machine learning web app that predicts customer churn, built with
Python, Streamlit and scikit-learn.

## Quick start

```bash
pip install -r requirements.txt

# one-off: fetch the dataset (not vendored in the repo)
curl -L -o Customer-Churn-Records.csv \
  https://raw.githubusercontent.com/241111-Python/team9-p2/main/datasets/Customer-Churn-Records.csv

python train_model.py          # writes churn_pipeline.pkl + metrics + schema
streamlit run churn_app.py     # launch the app
```

Then verify everything works:

```bash
python -m pytest test_churn.py -v      # 30 tests
```

## Project structure

```
churn_app.py            # Streamlit UI (the app entry point)
churn_core.py           # prediction logic, no Streamlit -- this is what the tests import
train_model.py          # trains the model, writes the artifacts below
test_churn.py           # test suite, incl. regressions for every fixed bug

churn_pipeline.pkl      # fitted Pipeline: impute -> encode -> scale -> model
feature_schema.json     # feature list, training defaults, valid ranges, threshold
model_metrics.json      # measured test-set scores

sample_customers.csv    # 25 rows you can upload to try the app
notebooks/              # original Colab exploration notebooks
```

## How it works

All preprocessing lives **inside** the saved pipeline. The app selects and
renames columns, then hands raw values straight to `pipeline.predict_proba`.
It never scales or encodes anything itself, which is what makes training and
serving impossible to drift apart.

`feature_schema.json` drives the UI: the single-customer form, the expected
column list and the fallback defaults are all generated from it. Retrain with a
different feature set and the app follows automatically.

## Model performance

Measured on a held-out 20% test split by `train_model.py` — the app reads these
from `model_metrics.json` rather than displaying hard-coded numbers.

| Model | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.725 | 0.403 | 0.696 | 0.510 | **0.762** |
| Random Forest | 0.706 | 0.380 | 0.675 | 0.486 | 0.744 |
| XGBoost | 0.742 | 0.413 | 0.608 | 0.492 | 0.758 |

### Getting a much better model

The default feature set is limited to the six columns in the CSV contract
below. The dataset has stronger predictors available:

```bash
python train_model.py --rich
```

| Feature set | Best model | ROC AUC | F1 |
|---|---|---|---|
| basic (default) | Logistic Regression | 0.762 | 0.510 |
| `--rich` | XGBoost | **0.880** | **0.648** |

`--rich` adds `CreditScore`, `NumOfProducts`, `IsActiveMember`, `HasCrCard` and
`Geography`. The app picks up the new schema with no code change, but uploaded
CSVs then need those columns to get the full benefit.

### Columns deliberately excluded

`Complain` correlates with the target at **r = 0.996** — it is recorded *after*
a customer churns. Training on it yields ~99.8% accuracy and a model that is
worthless in production. `CustomerId`, `RowNumber` and `Surname` are
identifiers. `train_model.py` refuses to start if any of them appear in the
feature set.

## CSV format for batch prediction

```csv
Gender,Age,Tenure,Balance,Satisfaction Score,EstimatedSalary
Female,42,2,1115.0,2,101348.88
Male,36,1,0.0,1,112542.58
```

Column matching is tolerant of case and punctuation (`estimated salary`,
`AGE`, `satisfaction_score` all match). Missing columns fall back to
training-set medians and the app warns you before showing any predictions.
Extra columns such as `CustomerId` are carried through to the output but not
used as features.

## Deployment note

The Streamlit Cloud entry point must be set to **`churn_app.py`**.
(`train_model.py` used to contain the app; it is now the training script.)

## License

MIT License © SHRUTI
