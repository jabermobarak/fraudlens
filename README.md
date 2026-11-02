# FraudLens

**A trained transaction fraud classifier with temporal evaluation, calibration, SHAP explanations and drift monitoring.**

Python · CatBoost · scikit-learn · FastAPI · Streamlit · pytest · Docker

Independent portfolio project by **Jaber Mobarak**. All transactions and fraud labels are simulated.
No Sber/customer data, bank connection or automatic payment blocking is used.

## What you can try

- Score one transaction and inspect its estimated fraud probability.
- Compare a routine purchase with an unusual transfer.
- Read signed SHAP feature contributions, including the additive base value.
- Upload a synthetic CSV for batch predictions and download the results.
- Inspect a logistic-regression baseline, test metrics and confusion matrices.
- Compare stable and shifted feature distributions with Population Stability Index (PSI).
- Retrain the complete model, calibration and threshold pipeline on CPU.

The trained `artifacts/model.cbm` is included. No API key, GPU or retraining is needed to use the dashboard.

## Start on Windows

1. Install **Python 3.12**.
2. Extract the archive and open the inner `fraudlens` folder.
3. Double-click **`start_windows.bat`**. The first run installs dependencies.
4. Open **http://localhost:8501** and keep the terminal open.
5. Click **Score transaction**, then load **Unusual transfer** and score again.

Manual PowerShell commands:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m streamlit run app.py --server.address 127.0.0.1
```

## macOS / Linux

Run `bash start_unix.sh` with Python 3.12 installed, or:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m streamlit run app.py --server.address 127.0.0.1
```

After dependency installation, scoring and training work offline. If BankOps is already using port 8501,
stop that app or add `--server.port 8502` and open localhost:8502.

## Learning pipeline

The simulator produces 40,000 independent transactions with stochastic labels and approximately 3% fraud.
Labels depend on deliberately invented risk relationships plus unobserved noise. They are not real fraud labels.

| Stage | Share | What it is used for |
| --- | --- | --- |
| Earliest training window | 55% | CatBoost and logistic-regression fitting; numeric drift reference |
| Later calibration window | 15% | Sigmoid calibration of CatBoost raw margins |
| Later policy window | 10% | Select a cost-based review threshold with at most 10% alerts |
| Final test window | 20% | Final metrics only |

All splits are disjoint and chronological. IDs, timestamps and labels are excluded from model features.
CatBoost is the predeclared primary model; no model selection or threshold fitting uses final-test labels.
Iterations are fixed; there is no early stopping against the test set.

The simulator is stationary by default, so time ordering alone is not a realistic temporal stress test.
It does not model delayed labels, shared customer histories, fraud rings or label censoring. A production
pipeline would need point-in-time feature engineering, label-availability gaps and group/entity checks.
The past-hour count is an input feature, not a deployed feature-store implementation.

## Actual delivered results

Final chronological test: **8,000 synthetic transactions, 248 fraud labels (3.1%)**.

| Metric | CatBoost + sigmoid calibration | Logistic baseline |
| --- | ---: | ---: |
| Average precision (AP) | 0.3866 | 0.3089 |
| ROC-AUC | 0.8156 | 0.8149 |
| Brier score, lower is better | 0.02345 | 0.02464 |
| Precision at selected threshold | 38.85% | 31.89% |
| Recall at selected threshold | 46.37% | 50.40% |
| Alert rate | 3.70% | 4.90% |
| Illustrative test cost · RUB | 904,281 | 896,527 |

CatBoost ranks better here; the baseline has slightly lower illustrative cost and higher recall at its own
validation-selected threshold. Both results are retained. The baseline is standard logistic regression without
a separate post-hoc calibrator; each model gets its own policy threshold using the same validation window.

AP is average precision, not trapezoidal PR-AUC or classification accuracy. A random ranking's expected AP
is roughly the positive prevalence. These metrics describe the simulator only, not real-bank effectiveness.
See `artifacts/report.json` for exact results, split boundaries, costs and monitoring outputs.

## Review policy and calibration

The illustrative cost is ₽300 for every review plus 60% of the amount for each missed fraudulent transaction.
It assumes review prevents all fraud loss and ignores delay, reviewer errors and customer-friction costs.
These are fictional assumptions, not demonstrated savings or bank policies.

A finite threshold grid is searched on the policy window, subject to the alert budget. It is a global probability
threshold, not an optimal transaction-specific expected-loss policy. The test alert rate is measured, not constrained
or retuned. Calibration uses a separate later window because class weighting changes raw probability behavior.

## Explainability

CatBoost's native Tree SHAP values sum with the base value to the **raw model margin**. The calibrated
probability is `sigmoid(slope * raw_margin + intercept)`. SHAP bars are not probability-point changes and do
not establish causal relationships. The API exposes all contributions and the calibration parameters.

## Monitoring

PSI compares numeric feature bins with training-reference proportions. A PSI of 0.2 is an illustrative
review flag, not a statistical significance threshold or automatic retraining trigger. PSI cannot establish
model degradation or concept drift; those require suitable labels and outcome metrics. Categorical drift
and missing-value monitoring are not implemented; input schema rejects missing/non-finite values.

## REST API

```bash
python -m uvicorn fraudlens.api:app --host 127.0.0.1 --port 8001
```

Interactive docs: http://localhost:8001/docs

- `POST /v1/score`: one transaction with SHAP explanation.
- `POST /v1/batch`: `{"transactions":[...]}`, 1–500 rows, predictions only.
- `POST /v1/drift`: same schema, 100–500 rows for a numeric drift report.
- `GET /health`: readiness after startup/model loading.

Optional `FRAUDLENS_API_KEY` environment variable requires `X-API-Key` on protected routes.
The dashboard loads the model directly; this API key does not authenticate the dashboard. Keep both local.
There is no user/tenant identity, distributed rate limiting or production authorization.

## Retrain and verify

From the repository root with the environment active:

```bash
python -m fraudlens.train
pip install -r requirements-dev.txt
ruff check .
ruff format --check .
pytest -q
python scripts/evaluate.py
```

Training regenerates data from seed 42 and overwrites the model, manifest, report and example CSV.
Restart Streamlit after retraining to clear the cached model. Retraining a different configuration requires
updating the README's reported metrics; the JSON report is the authoritative machine-readable output.
The evaluator re-scores the final test and checks it against the saved primary-model report without fitting.

Tests cover chronological isolation, schema rejection, hand-calculated costs, alert-budget selection,
SHAP additivity, batch/single consistency, drift identity/shift, model checksum validation, API behavior,
retraining/reload and actual Streamlit scoring widgets. A model checksum detects inconsistency, not malicious
replacement of both model and manifest. Only load trusted model artifacts.

## GitHub and Docker

Suggested repository name: **fraudlens**. Upload the extracted contents with `README.md` at the root.
Include `artifacts/`, `.github/workflows/ci.yml` and `.gitignore`; do not upload `.venv` or caches.
The small native model is deliberately committed and does not need Git LFS. See `docs/github_publish.md`.

```bash
docker build -t fraudlens .
docker run --rm -p 127.0.0.1:8501:8501 fraudlens
```

Docker runs as a non-root user. Docker, Windows execution and hosted GitHub CI were not run in the delivery
environment; Python training, inference, API and Streamlit tests ran on Linux.

## Files

| Path | Purpose |
| --- | --- |
| `fraudlens/data.py` | Simulator and time-based split |
| `fraudlens/train.py` | Training, calibration, baseline and report |
| `fraudlens/evaluation.py` | Ranking metrics, cost and threshold policy |
| `fraudlens/service.py` | Model loading, scoring, SHAP and CSV contract |
| `fraudlens/drift.py` | Numeric PSI reference and comparison |
| `fraudlens/api.py` | Validated REST endpoints |
| `app.py` | Browser dashboard |
| `artifacts/` | Pretrained model, manifest, report and sample CSV |
| `tests/` | Automated checks |

MIT license. See `docs/model_card.md` for intended use and limitations, and `docs/job_alignment.md` for researched skill alignment.
