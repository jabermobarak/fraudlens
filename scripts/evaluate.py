"""Recompute the delivered primary model's untouched synthetic test metrics."""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fraudlens.data import simulate, temporal_split  # noqa: E402
from fraudlens.evaluation import metrics  # noqa: E402
from fraudlens.schema import FEATURES, Transaction  # noqa: E402
from fraudlens.service import FraudService  # noqa: E402


def evaluate():
    saved = json.loads((ROOT / "artifacts/report.json").read_text())
    service = FraudService()
    test = temporal_split(simulate(saved["n"], saved["seed"]))["test"]
    scores = []
    for start in range(0, len(test), 500):
        items = [
            Transaction(**row)
            for row in test.iloc[start : start + 500][FEATURES].to_dict(orient="records")
        ]
        scores.extend(r["fraud_probability"] for r in service.score(items, explain=False))
    result = metrics(
        test.is_fraud, np.array(scores), test.amount_rub, service.manifest["policy"]["threshold"]
    )
    for key, value in result.items():
        if not np.allclose(value, saved["catboost"][key], atol=1e-10):
            raise AssertionError(f"Saved metric mismatch: {key}")
    print(json.dumps(result, indent=2))
    print(
        "Delivered model matches saved test metrics. No fitting or threshold selection performed."
    )


if __name__ == "__main__":
    evaluate()
