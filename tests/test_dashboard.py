"""Execute the Streamlit app headlessly and assert it raises nothing.

``AppTest`` actually runs the script top to bottom, so this catches the errors
an HTTP health check cannot: a bad channel index, a missing artefact, a plotly
call with the wrong argument.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DATA = REPO / "data" / "processed"
CKPTS = sorted((REPO / "outputs" / "checkpoints").glob("*_best.pt"))


@pytest.mark.skipif(not DATA.exists() or not CKPTS,
                    reason="needs a built dataset and a trained checkpoint")
def test_dashboard_runs_without_exception():
    from streamlit.testing.v1 import AppTest

    argv = sys.argv
    sys.argv = ["streamlit", "--", "--data", str(DATA), "--ckpt", str(CKPTS[-1])]
    try:
        at = AppTest.from_file(str(REPO / "app" / "dashboard.py"), default_timeout=600)
        at.run()
        assert not at.exception, [str(e) for e in at.exception]
        # The app must have rendered its headline metrics and the input maps.
        assert len(at.metric) >= 5, f"expected the metric row, got {len(at.metric)}"
        assert at.title, "no title rendered"

        # Exercise the sub-basin path too: it restricts both the map view and
        # the scoring mask, so a mistake there silently changes the numbers.
        region = next(b for b in at.selectbox if b.label == "Zoom to")
        region.select("bay_of_bengal").run()
        assert not at.exception, [str(e) for e in at.exception]
        assert len(at.metric) >= 5
    finally:
        sys.argv = argv
