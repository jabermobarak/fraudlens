import hashlib
import json
from pathlib import Path

import pandas as pd
from catboost import CatBoostClassifier, Pool

from fraudlens.drift import compare
from fraudlens.schema import CATEGORICAL, FEATURES, Transaction
from fraudlens.train import ROOT, calibrated


class FraudService:
    def __init__(self, artifacts=ROOT / "artifacts"):
        artifacts = Path(artifacts)
        self.manifest = json.loads((artifacts / "manifest.json").read_text())
        path = artifacts / "model.cbm"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.manifest["model_sha256"]:
            raise ValueError("Model checksum mismatch. Restore or retrain the bundle.")
        if self.manifest["features"] != FEATURES:
            raise ValueError("Feature schema mismatch")
        self.model = CatBoostClassifier()
        self.model.load_model(str(path))

    def score(self, transactions: list[Transaction], explain=True):
        if not 1 <= len(transactions) <= 500:
            raise ValueError("Provide 1 to 500 transactions")
        frame = pd.DataFrame([t.model_dump() for t in transactions])[FEATURES]
        margins = self.model.predict(frame, prediction_type="RawFormulaVal", thread_count=2)
        c = self.manifest["calibration"]
        probabilities = calibrated(margins, c["slope"], c["intercept"])
        shap = (
            self.model.get_feature_importance(
                Pool(frame, cat_features=CATEGORICAL), type="ShapValues", thread_count=2
            )
            if explain
            else None
        )
        output = []
        for i, probability in enumerate(probabilities):
            result = {
                "fraud_probability": float(probability),
                "recommendation": "review"
                if probability >= self.manifest["policy"]["threshold"]
                else "no_review",
                "threshold": self.manifest["policy"]["threshold"],
                "model_version": self.manifest["version"],
                "synthetic_demo": True,
            }
            if explain:
                result["raw_margin"] = float(margins[i])
                result["shap_base_margin"] = float(shap[i, -1])
                result["contributions"] = sorted(
                    [
                        {"feature": name, "raw_margin_contribution": float(shap[i, j])}
                        for j, name in enumerate(FEATURES)
                    ],
                    key=lambda item: -abs(item["raw_margin_contribution"]),
                )
                result["calibration"] = c
            output.append(result)
        return output

    def drift(self, transactions: list[Transaction]):
        return compare(
            self.manifest["reference"], pd.DataFrame([t.model_dump() for t in transactions])
        )


def parse_csv(file):
    frame = pd.read_csv(file, nrows=501)
    if set(frame.columns) != set(FEATURES):
        raise ValueError("CSV must contain exactly the documented feature columns")
    if not 1 <= len(frame) <= 500:
        raise ValueError("CSV must contain 1 to 500 rows")
    return [Transaction.model_validate(row) for row in frame.to_dict(orient="records")]
