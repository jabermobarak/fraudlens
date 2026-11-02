"""A documented stochastic simulator, not a bank dataset or real-world benchmark."""

import numpy as np
import pandas as pd


def simulate(n=40000, seed=42, shifted=False):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame(
        {
            "amount_rub": np.clip(
                rng.lognormal(8.1 + (0.8 if shifted else 0), 1.15, n), 1, 5e6
            ).round(2),
            "account_age_days": rng.integers(0, 3000, n),
            "device_age_days": rng.integers(0, 900, n),
            "distance_km": np.clip(rng.exponential(70, n), 0, 20000).round(2),
            "transactions_last_hour": np.clip(rng.poisson(2 + (3 if shifted else 0), n), 0, 200),
            "hour": rng.integers(0, 24, n),
            "new_payee": rng.binomial(1, 0.18, n),
            "channel": rng.choice(["mobile", "web", "pos"], n, p=[0.55, 0.2, 0.25]),
            "merchant_category": rng.choice(["retail", "travel", "digital", "services"], n),
        }
    )
    # Shared latent suspicious context creates overlapping, imperfectly separable classes.
    suspicious = rng.random(n) < 0.055
    frame.loc[suspicious, "device_age_days"] = rng.integers(0, 12, suspicious.sum())
    frame.loc[suspicious, "transactions_last_hour"] += rng.integers(2, 10, suspicious.sum())
    frame.loc[suspicious, "distance_km"] += rng.uniform(100, 2500, suspicious.sum())
    frame.loc[suspicious, "new_payee"] = rng.binomial(1, 0.75, suspicious.sum())
    f = frame
    logit = (
        -5.6
        + 1.0 * (f.device_age_days < 7)
        + 0.23 * f.transactions_last_hour
        + 0.9 * (f.distance_km > 300)
        + 0.85 * f.new_payee
        + 0.6 * (f.hour < 5)
        + 0.7 * (f.amount_rub > 15000)
        + 0.9 * ((f.new_payee == 1) & (f.device_age_days < 7))
        + rng.normal(0, 0.8, n)
    )
    frame["is_fraud"] = rng.binomial(1, 1 / (1 + np.exp(-logit)))
    frame.insert(0, "event_time", pd.date_range("2026-07-01", periods=n, freq="3min"))
    frame.insert(0, "transaction_id", [f"SYN-{seed}-{i:06}" for i in range(n)])
    return frame


def temporal_split(frame):
    ordered = frame.sort_values("event_time").reset_index(drop=True)
    n = len(ordered)
    cuts = [0, int(n * 0.55), int(n * 0.7), int(n * 0.8), n]
    return {
        name: ordered.iloc[cuts[i] : cuts[i + 1]].copy()
        for i, name in enumerate(["train", "calibration", "policy", "test"])
    }
