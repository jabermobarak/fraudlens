# Why FraudLens complements BankOps

Research checked 2 November 2026. Public vacancy descriptions guide the skills demonstrated here;
they do not reveal Sber's internal systems or guarantee an opening remains active.

Sber's public classic-ML recruitment page names Python, SQL, scikit-learn, boosting frameworks and
experimentation. An accessible anti-fraud DS/ML vacancy description highlights classification/ranking,
feature engineering, Python/SQL, metrics and model deployment; its publication is older and current
availability should be confirmed. A retail-risk listing includes CatBoost and scikit-learn.

Sources:

- [Official Sber classic-ML vacancy page](https://developers.sber.ru/kak-v-sbere/vacancies/data-ml-ai)
- [Anti-fraud DS/ML vacancy mirror](https://h.careers/job/cfa2f9fa-e3f9-42b0-872c-3335ca9f4b30)
- [Retail risk-modeling listing mirror](https://h.careers/job/7198e2dc-9d43-4cae-ae87-76124b0a2e9c)

| Skill | Concrete implementation |
| --- | --- |
| Tabular ML and imbalance | Actual CPU-trained CatBoost with class weights |
| Classical baseline | Logistic regression with train-fitted preprocessing |
| Experiment discipline | Disjoint chronological calibration/policy/test windows |
| Decision metrics | AP, ROC-AUC, Brier, precision/recall, illustrative costs |
| Model explanation | Native Tree SHAP with additivity checks |
| Model monitoring | Numeric PSI references and shifted-window example |
| Serving and validation | FastAPI schemas, batch route, CSV validation |
| Reproducibility | Fixed seed, pinned packages, native model, report and checksum |
| Quality engineering | Tests, retraining check, CI configuration and Dockerfile |

Not represented: commercial experience, real fraud outcomes, Spark/Hadoop-scale data, SQL feature-store
implementation, MLflow, real-time streaming and production operational experience. Do not list these as
implemented project features. This project demonstrates ML engineering; BankOps demonstrates AI workflows.

Technical references:

- [CatBoost SHAP API](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_get_feature_importance)
- [CatBoost SHAP output interpretation](https://catboost.ai/docs/en/concepts/output-data_feature-analysis_shap-values)
- [scikit-learn average precision](https://scikit-learn.org/1.7/modules/generated/sklearn.metrics.average_precision_score.html)
