import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
)

REVIEW_COST = 300.0
LOSS_FRACTION = 0.6
ALERT_BUDGET = 0.10


def policy_cost(y, score, amounts, threshold):
    alert = np.asarray(score) >= threshold
    missed = (np.asarray(y) == 1) & ~alert
    # Illustrative RUB costs: every review costs 300; undetected fraud loses 60% of amount.
    # Assumes review prevents all fraud loss. No claim of actual business savings.
    return float(alert.sum() * REVIEW_COST + np.asarray(amounts)[missed].sum() * LOSS_FRACTION)


def choose_threshold(y, score, amounts, budget=ALERT_BUDGET):
    candidates = np.unique(np.r_[np.quantile(score, np.linspace(0, 1, 301)), 1.0])
    valid = [
        (policy_cost(y, score, amounts, float(t)), float(t))
        for t in candidates
        if np.mean(score >= t) <= budget
    ]
    cost, threshold = min(valid, key=lambda pair: (pair[0], -pair[1]))
    return {
        "threshold": threshold,
        "validation_cost_rub": cost,
        "validation_alert_rate": float(np.mean(score >= threshold)),
        "budget": budget,
    }


def metrics(y, scores, amounts, threshold):
    labels = scores >= threshold
    return {
        "average_precision": float(average_precision_score(y, scores)),
        "roc_auc": float(roc_auc_score(y, scores)),
        "brier": float(brier_score_loss(y, scores)),
        "precision": float(precision_score(y, labels, zero_division=0)),
        "recall": float(recall_score(y, labels, zero_division=0)),
        "alert_rate": float(labels.mean()),
        "threshold": threshold,
        "illustrative_cost_rub": policy_cost(y, scores, amounts, threshold),
        "confusion_matrix_tn_fp_fn_tp": confusion_matrix(y, labels, labels=[0, 1]).ravel().tolist(),
    }
