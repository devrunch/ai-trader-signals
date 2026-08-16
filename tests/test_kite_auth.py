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
from unittest.mock import MagicMock, patch

import pytest

from app.config import Settings
from app.market.providers.kite_auth import KiteAuthError, _extract_request_token, refresh_session


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


def _settings() -> Settings:
    return Settings(
        zerodha_user_id="FHU286", zerodha_password="pw", zerodha_totp_secret="JBSWY3DPEHPK3PXP",
        zerodha_api_key="key", zerodha_api_secret="secret",
    )


def test_the_redirect_chain_is_walked_without_reaching_the_final_url():
    """Live bug this closes: the registered redirect URL is never actually
    reachable from wherever this runs (a container's own "localhost" is not
    the api service's — this is exactly what broke on the deployed box). The
    real observed chain is exactly 2 hops (connect/login -> connect/finish ->
    the registered URL), with the token in the SECOND hop's Location header —
    nothing past that hop is ever fetched or needs to be reachable."""
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.side_effect = [
        MagicMock(json=lambda: {"data": {"request_id": "rid"}}, raise_for_status=lambda: None),
        MagicMock(raise_for_status=lambda: None),
    ]
    hop1 = MagicMock(headers={"Location": "https://kite.zerodha.com/connect/finish?sess_id=x"})
    hop2 = MagicMock(headers={"Location": "http://localhost:8000/callback?status=success&request_token=TOK123"})
    mock_client.get.side_effect = [hop1, hop2]

    with patch("app.market.providers.kite_auth.httpx.Client", return_value=mock_client), \
         patch("app.market.providers.kite_auth.pyotp.TOTP") as mock_totp, \
         patch("app.market.providers.kite_auth.KiteConnect") as mock_kite_cls:
        mock_totp.return_value.now.return_value = "123456"
        mock_kite = mock_kite_cls.return_value
        mock_kite.generate_session.return_value = {"access_token": "atok", "user_name": "Test User"}
        mock_kite.instruments.side_effect = [[{"tradingsymbol": "RELIANCE"}], []]

        session = refresh_session(_settings())

    assert session.access_token == "atok"
    # Exactly 2 GETs — never a third call attempting to actually load the
    # (unreachable) callback URL.
    assert mock_client.get.call_count == 2
    mock_kite.generate_session.assert_called_once_with("TOK123", api_secret="secret")


def test_a_dead_end_redirect_chain_raises_kiteautherror():
    """No Location header at all on the response — the real shape of the
    pre-authorization "user is not enabled for the app" failure."""
    mock_client = MagicMock()
    mock_client.__enter__.return_value = mock_client
    mock_client.post.side_effect = [
        MagicMock(json=lambda: {"data": {"request_id": "rid"}}, raise_for_status=lambda: None),
        MagicMock(raise_for_status=lambda: None),
    ]
    dead_end = MagicMock(headers={}, url="https://kite.zerodha.com/connect/finish?api_key=x&sess_id=y")
    mock_client.get.return_value = dead_end

    with patch("app.market.providers.kite_auth.httpx.Client", return_value=mock_client), \
         patch("app.market.providers.kite_auth.pyotp.TOTP") as mock_totp:
        mock_totp.return_value.now.return_value = "123456"
        with pytest.raises(KiteAuthError):
            refresh_session(_settings())
