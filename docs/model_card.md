# Model card: FraudLens v1

Owner: Jaber Mobarak. Independent educational and portfolio demonstration, developed with AI assistance.

## Intended use

Demonstrate reproducible tabular machine-learning engineering, calibration, decision-policy selection,
explainability, monitoring and service integration. No real customer or payment decision is authorized.
The service returns a review recommendation only. It cannot approve, decline or block a transaction.

## Data and training

40,000 simulated transactions. No scraped transactions, personal identifiers or proprietary datasets.
The features encode an invented risk process with stochastic labels. Customer entities and delayed outcomes
are absent. Chronological windows separate fitting, calibration, policy selection and final evaluation.
The primary model is CatBoost with balanced class weights, fixed hyperparameters and CPU execution.
A sigmoid is fitted on the calibration window. A standard logistic-regression baseline is also evaluated.

Amounts are numeric analytical features, not a ledger. Prior-transaction counts and device ages must be
available before the transaction in a real deployment; this demo generates them rather than computing histories.

## Evaluation

Exact test performance is in `artifacts/report.json`; the final test contains 8,000 transactions.
No external dataset, confidence intervals, seed sweep or real-world validation is included. Inspect both
ranking metrics and decision costs: CatBoost's higher AP does not imply the lowest policy cost.
The cost assumptions and review-capacity budget are invented and must not be presented as bank savings.

## Limitations and controls

- Calibration is measured on one synthetic distribution; it is not a universal probability guarantee.
- Review recommendation depends on a validation-selected global threshold and assumed costs.
- SHAP is an additive explanation of the model margin, not causal attribution or an appeal mechanism.
- PSI flags numeric distribution changes only; no automatic retraining is performed.
- No sensitive demographic fields are used, but their absence does not establish fairness.
- No identity, tenant isolation, production feature store, high availability or regulatory certification.
- Native model plus manifest checksum reduces accidental mismatch; it does not authenticate the publisher.
- Do not enter real transaction/customer information into this portfolio app.

Next steps for a genuine deployment would start with business/data agreements, point-in-time joins,
label maturity, independent representative validation, subgroup/error analysis and human-review processes.
