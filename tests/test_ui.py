from streamlit.testing.v1 import AppTest

from fraudlens.train import ROOT


def test_dashboard_score_and_example_switch():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=30).run()
    assert not app.exception
    assert len(app.metric) == 4
    app.button[0].click().run()
    assert not app.exception
    low = app.session_state["result"]["fraud_probability"]
    app.selectbox[0].select("Unusual transfer").run()
    app.button[0].click().run()
    assert not app.exception
    assert app.session_state["result"]["fraud_probability"] > low
