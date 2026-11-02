# Delivery checks

Verified 2 November 2026, Python 3.12, Linux.

- 20 pytest checks passed, including small-pipeline retraining and reload.
- Streamlit AppTest exercised the dashboard, scoring button and example switch using the real model.
- FastAPI endpoints were exercised through the test client.
- Ruff lint and format checks passed.
- pip check found no dependency conflicts.
- The included model was re-scored on the final 8,000-row synthetic test window; saved metrics matched.
- Three deprecation warnings originate inside Streamlit/Pandas chart handling; UI scoring passed.

Not executed: Windows launch script, macOS, Docker build/run or hosted GitHub Actions. No real-bank dataset, live payment integration or production evaluation is claimed.
