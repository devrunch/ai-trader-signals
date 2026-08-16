"""
The daily refresh task: log in, push the fresh token to NestJS. Mirrors
square_off_positions's shape in worker/tasks.py — a thin trigger, loud on
failure, no special-case handling needed beyond that (a failed refresh just
means the router's Kite-then-yfinance fallback quietly takes over for the
rest of that day — see the design spec's "Error handling" section).
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from app.market.providers.kite_auth import KiteAuthError, KiteSession
from app.worker.tasks import refresh_zerodha_session


def test_a_successful_login_is_pushed_to_nestjs():
    session = KiteSession(access_token="tok_abc", nse_instruments=[], bse_instruments=[])

    with patch("app.worker.tasks.kite_auth.refresh_session", return_value=session) as mock_refresh, \
         patch("app.worker.tasks.httpx.put") as mock_put:
        mock_put.return_value = MagicMock(status_code=200)
        mock_put.return_value.raise_for_status = lambda: None

        result = refresh_zerodha_session()

    mock_refresh.assert_called_once()
    put_call = mock_put.call_args
    assert put_call.kwargs["json"] == {"accessToken": "tok_abc"}
    assert result == {"ok": True}


def test_a_login_failure_is_logged_and_returned_not_raised():
    """The whole point: a failed refresh must not crash the beat worker or
    take down anything else it schedules."""
    with patch("app.worker.tasks.kite_auth.refresh_session",
               side_effect=KiteAuthError("bad TOTP")):
        result = refresh_zerodha_session()

    assert result["ok"] is False
    assert "bad TOTP" in result["error"]
