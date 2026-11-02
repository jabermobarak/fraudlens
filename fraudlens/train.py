import hashlib
import json
import platform
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from fraudlens.data import simulate, temporal_split
from fraudlens.drift import compare, fit_reference
from fraudlens.evaluation import choose_threshold, metrics
from fraudlens.schema import CATEGORICAL, FEATURES, NUMERIC

ROOT = Path(__file__).resolve().parents[1]


def calibrated(margins, slope, intercept):
    return 1 / (1 + np.exp(-np.clip(np.asarray(margins) * slope + intercept, -40, 40)))


def train(output=ROOT / "artifacts", n=40000, seed=42, iterations=250):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = simulate(n, seed)
    splits = temporal_split(frame)
    tr, ca, po, te = (splits[k] for k in ["train", "calibration", "policy", "test"])
    model = CatBoostClassifier(
        iterations=iterations,
        depth=5,
        learning_rate=0.07,
        loss_function="Logloss",
        auto_class_weights="Balanced",
        random_seed=seed,
        thread_count=2,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(tr[FEATURES], tr.is_fraud, cat_features=CATEGORICAL)
    # Dedicated later window for calibration; neither fitting nor thresholding sees test labels.
    margin_ca = model.predict(ca[FEATURES], prediction_type="RawFormulaVal")
    calibrator = LogisticRegression(C=1e6, max_iter=1000)
    calibrator.fit(margin_ca.reshape(-1, 1), ca.is_fraud)
    slope, intercept = float(calibrator.coef_[0, 0]), float(calibrator.intercept_[0])
    scores = {
        k: calibrated(model.predict(v[FEATURES], prediction_type="RawFormulaVal"), slope, intercept)
        for k, v in splits.items()
        if k in {"policy", "test"}
    }
    policy = choose_threshold(po.is_fraud, scores["policy"], po.amount_rub)
    preprocessing = ColumnTransformer(
        [
            ("numeric", StandardScaler(), NUMERIC),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL,
            ),
        ]
    )
    baseline = make_pipeline(preprocessing, LogisticRegression(max_iter=2000))
    baseline.fit(tr[FEATURES], tr.is_fraud)
    baseline_policy_scores = baseline.predict_proba(po[FEATURES])[:, 1]
    baseline_policy = choose_threshold(po.is_fraud, baseline_policy_scores, po.amount_rub)
    baseline_test = baseline.predict_proba(te[FEATURES])[:, 1]
    reference = fit_reference(tr)
    model.save_model(str(output / "model.cbm"))
    model_sha = hashlib.sha256((output / "model.cbm").read_bytes()).hexdigest()
    manifest = {
        "version": "fraudlens-v1",
        "features": FEATURES,
        "categorical": CATEGORICAL,
        "model_sha256": model_sha,
        "calibration": {"slope": slope, "intercept": intercept},
        "policy": policy,
        "reference": reference,
        "seed": seed,
        "synthetic": True,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    report = {
        "scope": "Synthetic stochastic simulator. Not real-bank performance or actual savings.",
        "python": platform.python_version(),
        "n": n,
        "seed": seed,
        "splits": {
            k: {
                "rows": len(v),
                "fraud_rate": float(v.is_fraud.mean()),
                "start": str(v.event_time.min()),
                "end": str(v.event_time.max()),
            }
            for k, v in splits.items()
        },
        "selection": "CatBoost predeclared primary. No hyperparameter search or test-based model selection.",
        "catboost": metrics(te.is_fraud, scores["test"], te.amount_rub, policy["threshold"]),
        "logistic_baseline_uncalibrated": metrics(
            te.is_fraud, baseline_test, te.amount_rub, baseline_policy["threshold"]
        ),
        "no_review_cost_rub": float((te.amount_rub * te.is_fraud).sum() * 0.6),
        "all_review_cost_rub": len(te) * 300,
        "policy_validation": policy,
        "held_out_drift": compare(reference, te),
        "shifted_simulation_drift": compare(reference, simulate(3000, seed + 99, shifted=True)),
    }
    (output / "report.json").write_text(json.dumps(report, indent=2))
    te[FEATURES].head(150).to_csv(output / "example_batch.csv", index=False)
    return report


if __name__ == "__main__":
    report = train()
    print(
        json.dumps(
            {k: report[k] for k in ["n", "catboost", "logistic_baseline_uncalibrated"]}, indent=2
        )
    )
