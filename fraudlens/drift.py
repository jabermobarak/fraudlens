"""Population Stability Index: distribution shift signal, not fraud-quality measurement."""

import numpy as np

from fraudlens.schema import NUMERIC


def proportions(values, cuts):
    bins = np.r_[-np.inf, cuts, np.inf]
    counts, _ = np.histogram(np.asarray(values, dtype=float), bins=bins)
    return (counts / counts.sum()).tolist()


def fit_reference(frame):
    result = {}
    for name in NUMERIC:
        cuts = np.unique(np.quantile(frame[name], np.linspace(0.1, 0.9, 9))).tolist()
        result[name] = {"cuts": cuts, "proportions": proportions(frame[name], cuts)}
    return result


def compare(reference, frame):
    if len(frame) < 100:
        raise ValueError("At least 100 observations are required for this demo drift report")
    results = []
    for name, ref in reference.items():
        expected = np.clip(np.asarray(ref["proportions"]), 1e-6, 1)
        observed = np.clip(proportions(frame[name], ref["cuts"]), 1e-6, 1)
        value = float(np.sum((observed - expected) * np.log(observed / expected)))
        results.append({"feature": name, "psi": value, "review_distribution": value >= 0.2})
    return sorted(results, key=lambda x: -x["psi"])
