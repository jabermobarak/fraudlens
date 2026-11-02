# Publish FraudLens

Repository name: `fraudlens`

Suggested description: Transaction fraud ML service with CatBoost, calibration, time-based evaluation,
SHAP explanations, PSI drift monitoring, FastAPI and Streamlit. Synthetic data.

Suggested topics: `machine-learning`, `catboost`, `fraud-detection`, `fastapi`, `streamlit`, `shap`, `model-monitoring`.

Extract the ZIP and upload the contents of its inner folder to an empty repository. `README.md` should
appear at the repository root. Include the trained model under `artifacts/` and the hidden `.github/` folder.
Upload the delivered files before running, or manually exclude `.venv`, caches and secrets. Website uploads
do not apply `.gitignore` automatically.

Command-line route, replacing YOUR-USERNAME:

```bash
git init
git add .
git commit -m "Build fraud classifier with calibration, SHAP and drift monitoring"
git branch -M main
git remote add origin https://github.com/YOUR-USERNAME/fraudlens.git
git push -u origin main
```

Portfolio wording: “Built a synthetic-data fraud-detection ML service with CatBoost, separate temporal
calibration and policy selection, SHAP explanations, drift monitoring and a tested API/dashboard.”

Before interviewing, run both examples, explain what AP measures, show why the baseline and primary model
trade off recall and review volume, and explain why synthetic results do not establish bank readiness.
