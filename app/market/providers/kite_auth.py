"""
The daily Kite Connect login.

Kite Connect's access_token expires roughly every 24 hours (~6am IST), and
the standard flow expects a human to log in through a browser and click
"Authorize" once for a new app. That one-time authorize click is a real,
one-time manual step — already done for this app + account — but everything
after it can be scripted: a password + TOTP login against Kite's own
(unofficial, undocumented) /api/login and /api/twofa endpoints produces a
logged-in session, which the *official* generate_session() call then
exchanges for a real access_token.

Using the unofficial endpoints to script what's meant to be an interactive
browser login is against Kite's terms of service in spirit. That's a known,
accepted trade-off — it's what lets the token refresh happen at 6am unattended
instead of requiring someone to click through a browser before every trading
day.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx
import pyotp
from kiteconnect import KiteConnect

from app.config import Settings

logger = logging.getLogger(__name__)

_LOGIN_URL = "https://kite.zerodha.com/api/login"
_TWOFA_URL = "https://kite.zerodha.com/api/twofa"
_CONNECT_LOGIN_URL = "https://kite.zerodha.com/connect/login"


class KiteAuthError(Exception):
    """The daily login failed. Callers should NOT crash the process over
    this — a stale token in NestJS just means every NSE/BSE call falls
    through to yfinance until the next successful refresh."""


@dataclass
class KiteSession:
    access_token: str
    nse_instruments: list[dict[str, Any]]
    bse_instruments: list[dict[str, Any]]


def _extract_request_token(history: list, final_response: Any) -> str | None:
    """Walk a redirect chain (oldest first) plus the final response, looking
    for request_token= in a Location header or in the final URL itself —
    httpx may or may not have already followed the redirect depending on how
    it was called, so both are checked."""
    for resp in [*history, final_response]:
        loc = resp.headers.get("Location") or str(resp.url)
        if "request_token=" in loc:
            return loc.split("request_token=")[1].split("&")[0]
    return None


def refresh_session(settings: Settings) -> KiteSession:
    """Password + TOTP login, official token exchange, fresh NSE+BSE
    instrument dumps. Raises KiteAuthError on any failure — see the module
    docstring for why that's the right thing to raise rather than degrade."""
    try:
        with httpx.Client(timeout=15) as client:
            login = client.post(_LOGIN_URL, data={
                "user_id": settings.zerodha_user_id,
                "password": settings.zerodha_password,
            })
            login.raise_for_status()
            request_id = login.json()["data"]["request_id"]

            totp_code = pyotp.TOTP(settings.zerodha_totp_secret).now()
            twofa = client.post(_TWOFA_URL, data={
                "user_id": settings.zerodha_user_id,
                "request_id": request_id,
                "twofa_value": totp_code,
                "twofa_type": "totp",
            })
            twofa.raise_for_status()

            final = client.get(
                _CONNECT_LOGIN_URL,
                params={"api_key": settings.zerodha_api_key, "v": 3},
                follow_redirects=True,
            )
            request_token = _extract_request_token(final.history, final)
            if not request_token:
                raise KiteAuthError(
                    f"No request_token in Kite's response. Final URL: {final.url}"
                )
    except httpx.HTTPError as e:
        raise KiteAuthError(f"Login request failed: {e}") from e

    kite = KiteConnect(api_key=settings.zerodha_api_key)
    try:
        result = kite.generate_session(request_token, api_secret=settings.zerodha_api_secret)
        access_token = result["access_token"]
        kite.set_access_token(access_token)
        nse = kite.instruments("NSE")
        bse = kite.instruments("BSE")
    except Exception as e:
        raise KiteAuthError(f"Token exchange or instrument fetch failed: {e}") from e

    logger.info("Zerodha session refreshed — user %s", result.get("user_name"))
    return KiteSession(access_token=access_token, nse_instruments=nse, bse_instruments=bse)
