import io
import json

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from fraudlens.api import create_app
from fraudlens.data import simulate, temporal_split
from fraudlens.drift import compare, fit_reference
from fraudlens.evaluation import choose_threshold, policy_cost
from fraudlens.schema import EXAMPLES, FEATURES, Transaction
from fraudlens.service import FraudService, parse_csv
from fraudlens.train import ROOT, calibrated, train


@pytest.fixture(scope="module")
def service():
    return FraudService()


def test_deterministic_simulator_and_temporal_isolation():
    frame = simulate(1000)
    pd.testing.assert_frame_equal(frame, simulate(1000))
    splits = list(temporal_split(frame.sample(frac=1, random_state=1)).values())
    assert sum(map(len, splits)) == len(frame)
    for left, right in zip(splits[:-1], splits[1:], strict=True):
        assert left.event_time.max() < right.event_time.min()
        assert not set(left.transaction_id) & set(right.transaction_id)
    assert "is_fraud" not in FEATURES and "transaction_id" not in FEATURES


@pytest.mark.parametrize(
    "field,value",
    [
        ("amount_rub", -1),
        ("amount_rub", float("nan")),
        ("distance_km", float("inf")),
        ("hour", 24),
        ("new_payee", 3),
        ("channel", "unknown"),
        ("is_fraud", 1),
    ],
)
def test_schema_rejects_invalid_input(field, value):
    item = {**EXAMPLES["Routine purchase"], field: value}
    with pytest.raises(ValidationError):
        Transaction(**item)


def test_cost_hand_calculation():
    # Review first row: 300 review + missed last-row fraud 1000 * .6.
    assert policy_cost([1, 0, 1], [0.9, 0.1, 0.2], [100, 200, 1000], 0.5) == 900


def test_policy_obeys_budget_and_beats_no_review_on_validation():
    y = np.array([0] * 90 + [1] * 10)
    scores = np.arange(100) / 100
    amounts = np.repeat(5000, 100)
    policy = choose_threshold(y, scores, amounts)
    assert policy["validation_alert_rate"] <= 0.1
    assert policy["validation_cost_rub"] < policy_cost(y, scores, amounts, 1)


def test_calibration_is_finite_and_monotone():
    p = calibrated(np.array([-1e9, -10, 0, 10, 1e9]), 1, 0)
    assert np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all()
    assert (np.diff(p) >= 0).all()


@pytest.mark.parametrize("example", list(EXAMPLES))
def test_shap_additivity_and_calibrated_prediction(service, example):
    r = service.score([Transaction(**EXAMPLES[example])])[0]
    total = r["shap_base_margin"] + sum(x["raw_margin_contribution"] for x in r["contributions"])
    assert total == pytest.approx(r["raw_margin"], abs=1e-6)
    assert r["fraud_probability"] == pytest.approx(
        calibrated(
            total,
            **{"slope": r["calibration"]["slope"], "intercept": r["calibration"]["intercept"]},
        )
    )


def test_batch_and_single_agree(service):
    item = Transaction(**EXAMPLES["Routine purchase"])
    a = service.score([item])[0]["fraud_probability"]
    b = service.score([item, item], explain=False)
    assert a == pytest.approx(b[0]["fraud_probability"])
    assert b[0] == b[1]


def test_checksum_guard(tmp_path):
    manifest = json.loads((ROOT / "artifacts/manifest.json").read_text())
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "model.cbm").write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="checksum"):
        FraudService(tmp_path)


def test_drift_identity_shift_and_minimum_size():
    frame = simulate(3000)
    reference = fit_reference(frame)
    assert all(r["psi"] == pytest.approx(0) for r in compare(reference, frame))
    shifted = frame.copy()
    shifted["amount_rub"] *= 20
    amount = next(r for r in compare(reference, shifted) if r["feature"] == "amount_rub")
    assert amount["psi"] > 0.2
    with pytest.raises(ValueError):
        compare(reference, frame.head(10))


def test_csv_contract():
    rows = parse_csv(ROOT / "artifacts/example_batch.csv")
    assert len(rows) == 150
    invalid = pd.DataFrame([EXAMPLES["Routine purchase"]] * 501)
    with pytest.raises(ValueError):
        parse_csv(io.StringIO(invalid.to_csv(index=False)))
    with pytest.raises(ValueError):
        parse_csv(io.StringIO("wrong,column\n1,2\n"))


def test_api_auth_validation_and_batch(service):
    item = EXAMPLES["Routine purchase"]
    with TestClient(create_app(api_key="test")) as client:
        assert client.get("/health").status_code == 200
        assert client.post("/v1/score", json=item).status_code == 401
        headers = {"X-API-Key": "test"}
        r = client.post("/v1/score", headers=headers, json=item)
        assert r.status_code == 200
        assert r.json()["fraud_probability"] == pytest.approx(
            service.score([Transaction(**item)])[0]["fraud_probability"]
        )
        assert (
            client.post("/v1/score", headers=headers, json={**item, "is_fraud": 1}).status_code
            == 422
        )
        assert (
            client.post(
                "/v1/batch", headers=headers, json={"transactions": [item, item]}
            ).status_code
            == 200
        )
        assert (
            client.post("/v1/drift", headers=headers, json={"transactions": [item]}).status_code
            == 422
        )


def test_retraining_and_reload(tmp_path):
    report = train(output=tmp_path, n=6000, iterations=35)
    assert report["n"] == 6000
    assert report["policy_validation"]["validation_alert_rate"] <= 0.1
    result = FraudService(tmp_path).score([Transaction(**EXAMPLES["Routine purchase"])])[0]
    assert 0 <= result["fraud_probability"] <= 1
