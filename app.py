import json

import pandas as pd
import streamlit as st

from fraudlens.data import simulate
from fraudlens.schema import EXAMPLES, Transaction
from fraudlens.service import FraudService, parse_csv
from fraudlens.train import ROOT

st.set_page_config(page_title="FraudLens · Risk Lab", page_icon="🔎", layout="wide")
st.markdown(
    """<style>.stApp{background:#0c1320}
[data-testid="stMetric"]{background:#162238;border-radius:12px;padding:18px}
.eyebrow{color:#65d6b3;letter-spacing:.18em;font-size:12px}</style>""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load():
    return FraudService()


service = load()
report = json.loads((ROOT / "artifacts/report.json").read_text())
st.markdown('<div class="eyebrow">FRAUDLENS / MODEL RISK LAB</div>', unsafe_allow_html=True)
st.title("Spot suspicious transactions. Explain the signal.")
st.caption(
    "A trained CatBoost classifier · Synthetic data only · Independent portfolio by Jaber Mobarak"
)
metrics = report["catboost"]
cols = st.columns(4)
for col, label, value in zip(
    cols,
    ["TEST AVERAGE PRECISION", "TEST RECALL", "TEST ALERT RATE", "TEST FRAUD RATE"],
    [
        metrics["average_precision"],
        metrics["recall"],
        metrics["alert_rate"],
        report["splits"]["test"]["fraud_rate"],
    ],
    strict=True,
):
    col.metric(label, f"{value:.1%}")
st.caption(
    "Measured on the final chronological synthetic test window; these are not real-bank results."
)
score_tab, batch_tab, eval_tab, drift_tab = st.tabs(
    ["Score a transaction", "Batch scoring", "Model evaluation", "Drift monitor"]
)
with score_tab:
    example = st.selectbox("Load an example", list(EXAMPLES))
    defaults = EXAMPLES[example]
    with st.form("transaction"):
        left, right = st.columns(2)
        values = {}
        with left:
            values["amount_rub"] = st.number_input(
                "Amount · RUB",
                min_value=1.0,
                max_value=5e6,
                value=float(defaults["amount_rub"]),
                key=example + "amount",
            )
            for key, title, maximum in [
                ("account_age_days", "Account age · days", 20000),
                ("device_age_days", "Device age · days", 20000),
                ("transactions_last_hour", "Previous transactions · past hour", 200),
            ]:
                values[key] = st.number_input(
                    title, 0, maximum, int(defaults[key]), key=example + key
                )
        with right:
            values["distance_km"] = st.number_input(
                "Distance from usual location · km",
                0.0,
                20000.0,
                float(defaults["distance_km"]),
                key=example + "distance",
            )
            values["hour"] = st.slider(
                "Hour · UTC", 0, 23, int(defaults["hour"]), key=example + "hour"
            )
            values["new_payee"] = int(
                st.checkbox("New payee", bool(defaults["new_payee"]), key=example + "payee")
            )
            for key, options in [
                ("channel", ["mobile", "web", "pos"]),
                ("merchant_category", ["retail", "travel", "digital", "services"]),
            ]:
                values[key] = st.selectbox(
                    key.replace("_", " ").title(),
                    options,
                    index=options.index(defaults[key]),
                    key=example + key,
                )
        submitted = st.form_submit_button("Score transaction", type="primary")
    if submitted:
        result = service.score([Transaction(**values)])[0]
        st.session_state["result"] = result
    if "result" in st.session_state:
        r = st.session_state["result"]
        st.subheader(
            f"{r['fraud_probability']:.1%} estimated fraud probability · {r['recommendation'].replace('_', ' ').upper()}"
        )
        st.caption(
            f"Policy threshold: {r['threshold']:.4f}. Recommendation only; no transaction is blocked."
        )
        contributions = pd.DataFrame(r["contributions"]).set_index("feature")
        st.bar_chart(contributions, horizontal=True, color="#65d6b3")
        st.caption(
            "SHAP values explain the raw model margin, not percentage-point changes in calibrated probability. Associations are not causes."
        )
        with st.expander("Inspect complete prediction"):
            st.json(r)
with batch_tab:
    st.write("Upload 1–500 synthetic transactions with exactly the documented feature columns.")
    st.download_button(
        "Download sample CSV",
        (ROOT / "artifacts/example_batch.csv").read_bytes(),
        "example_batch.csv",
        "text/csv",
    )
    uploaded = st.file_uploader("Synthetic transaction CSV", type="csv")
    if uploaded is not None:
        if uploaded.size > 1_000_000:
            st.error("Maximum upload size is 1 MB.")
        else:
            try:
                rows = parse_csv(uploaded)
                scored = service.score(rows, explain=False)
                st.dataframe(scored)
                st.download_button(
                    "Download predictions",
                    pd.DataFrame(scored).to_csv(index=False),
                    "predictions.csv",
                    "text/csv",
                )
            except (ValueError, TypeError):
                st.error(
                    "Invalid CSV. Use the sample's columns and valid values; no labels or IDs."
                )
with eval_tab:
    st.subheader("Chronological, separate evaluation stages")
    st.dataframe(report["splits"])
    st.dataframe(
        {
            "CatBoost + sigmoid calibration": report["catboost"],
            "Logistic regression baseline": report["logistic_baseline_uncalibrated"],
        }
    )
    st.write(
        "Train → calibration → policy threshold selection → final test. CatBoost was predeclared as the primary model."
    )
    st.caption(
        "Costs use fictional assumptions: ₽300 per review; missed fraud loses 60% of amount. Review is assumed to prevent all loss. This is not measured savings."
    )
with drift_tab:
    st.subheader("Feature distribution monitoring")
    scenario = st.radio(
        "Monitoring window", ["Held-out test", "Simulated distribution shift"], horizontal=True
    )
    key = "held_out_drift" if scenario == "Held-out test" else "shifted_simulation_drift"
    st.dataframe(report[key])
    st.caption(
        "PSI ≥ 0.2 triggers an illustrative distribution-review flag. PSI does not measure fraud quality or concept drift. Numeric features only."
    )
    if st.button("Generate a fresh shifted window"):
        frame = simulate(300, seed=123, shifted=True)
        rows = [
            Transaction(**item)
            for item in frame[service.manifest["features"]].to_dict(orient="records")
        ]
        st.dataframe(service.drift(rows))
