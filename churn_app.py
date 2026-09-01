"""
Customer Churn Prediction -- Streamlit app.

Run with:  streamlit run churn_app.py
Train first with:  python train_model.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

import churn_core as core

st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.markdown(
    "<h1 style='text-align:center;color:teal;'>Customer Churn Prediction</h1>",
    unsafe_allow_html=True,
)


@st.cache_resource
def _load():
    return core.load_artifacts()


try:
    pipeline, schema, metrics = _load()
except FileNotFoundError as exc:
    st.error(str(exc))
    st.stop()

THRESHOLD = schema.get("threshold", 0.5)
FEATURES = schema["features"]

st.caption(
    f"Model: **{schema.get('model', 'unknown')}** · trained on "
    f"{schema.get('trained_rows', 0):,} customers · decision threshold "
    f"{THRESHOLD:.2f}"
)

batch_tab, single_tab, about_tab = st.tabs(
    ["Batch prediction", "Single customer", "Model performance"]
)


# --------------------------------------------------------------------------
# Batch prediction from an uploaded CSV
# --------------------------------------------------------------------------
with batch_tab:
    st.write("Upload customer data to score every row.")
    st.caption("Expected columns: " + ", ".join(FEATURES))

    uploaded = st.file_uploader("CSV file", type=["csv"])

    if uploaded is None:
        st.info("Upload a CSV to continue, or use the **Single customer** tab.")
    else:
        try:
            data = pd.read_csv(uploaded, encoding="utf-8-sig")
        except Exception as exc:
            st.error(f"Could not read that CSV: {exc}")
            st.stop()

        if data.empty:
            st.error("That file has no rows.")
            st.stop()

        st.write("Uploaded data preview:")
        st.dataframe(data.head())

        X, missing, mapping = core.build_features(data, schema)

        # Warn *before* showing any numbers, so nobody reads a prediction
        # built on defaults without knowing it.
        if missing:
            st.warning(
                "These columns were not found and were filled with training "
                "defaults, so the predictions below are less reliable: "
                + ", ".join(missing)
            )
        renamed = {t: s for t, s in mapping.items() if t != s}
        if renamed:
            st.caption(
                "Matched columns: "
                + ", ".join(f"`{s}` -> `{t}`" for t, s in renamed.items())
            )
        if len(missing) == len(FEATURES):
            st.error("None of the expected columns were found. Nothing to predict on.")
            st.stop()

        try:
            preds, probs = core.predict(pipeline, X, THRESHOLD)
        except Exception as exc:
            st.error(f"Prediction failed: {type(exc).__name__}: {exc}")
            st.stop()

        results = data.copy()
        results["Churn Prediction"] = preds
        results["Churn Probability (%)"] = (probs * 100).round(2)
        results["Risk"] = [core.risk_band(p) for p in probs]

        st.success(f"Scored {len(results):,} customers.")

        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted to churn", f"{int(preds.sum()):,}")
        c2.metric("Churn rate", f"{preds.mean():.1%}")
        c3.metric("Avg probability", f"{probs.mean() * 100:.1f}%")

        display = ["Churn Prediction", "Churn Probability (%)", "Risk"]
        for ident in ("CustomerId", "Surname"):
            if ident in results.columns:
                display.insert(0, ident)
        st.dataframe(results[display])

        st.download_button(
            "Download predictions as CSV",
            results.to_csv(index=False).encode("utf-8"),
            file_name="churn_predictions.csv",
            mime="text/csv",
        )

        # ---- Insights -----------------------------------------------------
        if "Gender" not in missing:
            st.markdown("### Churn rate by gender")
            by_gender = (
                pd.DataFrame({"Gender": X["Gender"], "Churn": preds})
                .dropna(subset=["Gender"])
                .groupby("Gender", as_index=False)["Churn"]
                .mean()
            )
            if not by_gender.empty:
                fig, ax = plt.subplots()
                sns.barplot(data=by_gender, x="Gender", y="Churn", ax=ax)
                ax.set_ylabel("Churn rate")
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                plt.close(fig)

        if "Balance" not in missing:
            st.markdown("### Churn rate by balance tier")
            tiers = pd.DataFrame(
                {"Tier": core.balance_tier(X["Balance"]), "Churn": preds}
            ).dropna(subset=["Tier"])
            by_tier = tiers.groupby("Tier", as_index=False, observed=True)["Churn"].mean()
            if not by_tier.empty:
                fig, ax = plt.subplots()
                sns.barplot(data=by_tier, x="Tier", y="Churn", ax=ax)
                ax.set_ylabel("Churn rate")
                ax.set_ylim(0, 1)
                st.pyplot(fig)
                plt.close(fig)


# --------------------------------------------------------------------------
# Single-customer prediction
# --------------------------------------------------------------------------
with single_tab:
    st.write("Enter one customer's details.")
    values = {}
    columns = st.columns(2)

    for i, feature in enumerate(FEATURES):
        target = columns[i % 2]
        default = schema["defaults"][feature]
        if feature in schema["categorical"]:
            options = schema["categories"].get(feature, [])
            index = options.index(default) if default in options else 0
            values[feature] = target.selectbox(feature, options, index=index)
        else:
            low, high = schema["ranges"].get(feature, [0.0, float(default) * 4 + 1])
            step = 1.0 if float(high - low) > 20 else 0.1
            values[feature] = target.number_input(
                feature,
                min_value=float(low),
                max_value=float(high),
                value=float(default),
                step=step,
            )

    if st.button("Predict churn risk", type="primary"):
        X_one = pd.DataFrame([values])[FEATURES]
        preds, probs = core.predict(pipeline, X_one, THRESHOLD)
        prob = float(probs[0])
        band = core.risk_band(prob)

        st.metric("Churn probability", f"{prob * 100:.1f}%")
        st.progress(min(max(prob, 0.0), 1.0))
        message = f"**{band} risk** — predicted to {'churn' if preds[0] else 'stay'}."
        (st.error if band == "High" else st.warning if band == "Medium" else st.success)(
            message
        )


# --------------------------------------------------------------------------
# Real, measured model performance
# --------------------------------------------------------------------------
with about_tab:
    st.markdown("### Model comparison")
    if metrics:
        st.caption(
            "Measured on the held-out 20% test split by `train_model.py` — "
            "not hard-coded."
        )
        st.dataframe(pd.DataFrame(metrics).T)
    else:
        st.info("Run `python train_model.py` to generate model_metrics.json.")

    st.markdown("### Excluded columns")
    st.caption(
        "`Complain` is recorded after a customer churns (r=0.996 with the "
        "target) and would inflate accuracy to ~99.8% while being useless in "
        "production. `CustomerId`, `RowNumber` and `Surname` are identifiers. "
        "All are deliberately excluded from the feature set."
    )
