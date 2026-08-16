"""
kite_auth — the daily scripted login.

The login itself (password + TOTP against Kite's unofficial endpoints, then
the official generate_session() exchange) was proven live end-to-end before
this was written — see the design spec. What's unit-testable without a real
network call is the one fragile parsing step: pulling request_token out of
whatever redirect chain Kite's connect/finish response actually is.
"""
from __future__ import annotations

from types import SimpleNamespace

from app.market.providers.kite_auth import _extract_request_token


def _response(url: str, location: str | None = None) -> SimpleNamespace:
    headers = {"Location": location} if location else {}
    return SimpleNamespace(url=url, headers=headers)


def test_finds_the_token_in_a_redirect_location_header():
    history = [_response("https://kite.zerodha.com/connect/login")]
    final = _response(
        "https://kite.zerodha.com/connect/finish",
        location="http://localhost:8000/api/broker/zerodha/callback?status=success&request_token=ABC123&action=login",
    )
    assert _extract_request_token(history, final) == "ABC123"


def test_finds_the_token_in_the_final_url_itself():
    """If httpx already followed the redirect, the token is in `.url`, not a
    `Location` header on the last response."""
    history = []
    final = _response(
        "http://localhost:8000/api/broker/zerodha/callback?request_token=XYZ789&status=success"
    )
    assert _extract_request_token(history, final) == "XYZ789"


def test_stops_at_the_first_ampersand():
    final = _response(
        "http://localhost:8000/callback?other=1&request_token=TOK&status=success"
    )
    assert _extract_request_token([], final) == "TOK"


def test_returns_none_when_no_token_is_anywhere_in_the_chain():
    """The real failure mode this guards: connect/finish returned an error
    (e.g. "the user is not enabled for the app") instead of a redirect."""
    final = _response("https://kite.zerodha.com/connect/finish?api_key=x&sess_id=y")
    assert _extract_request_token([], final) is None
