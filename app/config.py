"""
Application settings.

Two deliberate properties:

  * `Settings()` performs NO I/O. It reads environment variables and nothing
    else. An earlier version minted a Bedrock API key from a `model_validator`,
    which meant importing *any* module in the package walked boto3's full
    credential chain — including EC2 instance metadata, with a multi-second
    timeout off-EC2 — and made the pure-maths modules (`analysis`, `indicators`)
    unimportable in a test without AWS. Key minting now lives in
    `app.llm.client` and happens lazily on first use.

  * Settings are obtained through `get_settings()`, not a module-level
    singleton, so tests can `get_settings.cache_clear()` after patching the
    environment.
"""
from __future__ import annotations

import base64
import logging
from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


def generate_bedrock_key(access_key: str, secret_key: str, region: str, expires: int = 43200) -> str:
    """
    Build a Bedrock Mantle presigned key from IAM credentials.
    Works with IAM user keys (AKIA…) or instance-role credentials.
    Falls back to boto3's default credential chain if keys are empty.

    NOTE: this performs network-capable credential resolution. Never call it at
    import time — see the module docstring.
    """
    import boto3
    from botocore.auth import SigV4QueryAuth
    from botocore.awsrequest import AWSRequest

    session = boto3.Session(
        aws_access_key_id=access_key or None,
        aws_secret_access_key=secret_key or None,
        region_name=region,
    )
    frozen = session.get_credentials().get_frozen_credentials()

    auth = SigV4QueryAuth(frozen, "bedrock", region, expires=expires)
    req = AWSRequest(
        method="POST",
        url="https://bedrock.amazonaws.com/?Action=CallWithBearerToken&Version=1",
        headers={"host": "bedrock.amazonaws.com"},
    )
    auth.add_auth(req)

    # Bedrock Mantle expects the URL without the https:// scheme
    url_body = req.url.replace("https://", "")
    encoded = base64.b64encode(url_body.encode()).decode()
    return f"bedrock-api-key-{encoded}"


class Settings(BaseSettings):
    environment: str = "development"

    # NestJS API — for pulling the dynamic watchlist before each screener run,
    # and the chat agent's per-user trading context
    api_service_url: str = "http://localhost:8000"
    internal_api_key: str = ""       # shared secret for service-to-service calls
    redis_url: str = "redis://redis:6379/0"

    # AWS
    aws_region: str = "ap-south-1"
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # SQS
    sqs_signals_queue_url: str = ""
    sqs_tasks_queue_url: str = ""

    # Bedrock Mantle — OpenAI-compatible endpoint
    # Available: deepseek.v3.2 | mistral.mistral-large-3-675b-instruct | qwen.qwen3-235b-a22b-2507
    # Set BEDROCK_API_KEY from a long-term key generated in the Bedrock console
    # (API keys > Generate long-term API key). The auto-mint fallback in
    # app/llm/client.py only kicks in if this is left blank — its SigV4
    # bearer-token scheme is unverified against a real AWS mechanism, prefer
    # setting a real key.
    bedrock_api_key: str = ""
    bedrock_model_id: str = "deepseek.v3.2"
    bedrock_base_url: str = "https://bedrock-mantle.ap-south-1.api.aws/v1"

    # Market data
    news_api_key: str = ""
    # Zerodha Kite Connect — real NSE/BSE market data. ZERODHA_USER_ID/PASSWORD/
    # TOTP_SECRET are only used by the daily scripted login (see
    # app/market/providers/kite_auth.py); ZERODHA_API_KEY/SECRET are also
    # needed for the official generate_session() token exchange.
    zerodha_api_key: str = ""
    zerodha_api_secret: str = ""
    zerodha_user_id: str = ""
    zerodha_password: str = ""
    zerodha_totp_secret: str = ""
    # Twelve Data — forex/commodity spot pairs (XAU/USD, ...) that neither
    # Kite (India-only, no forex) nor yfinance (XAUUSD=X is flaky/unreliable
    # in practice) actually cover. Free-tier key.
    twelve_data_api_key: str = ""

    # ---- Signal thresholds ----
    confidence_threshold: float = 0.65
    min_reward_risk: float = 1.5
    min_adx_trend: float = 20.0          # skip choppy/range-bound regimes entirely
    min_stop_atr_multiple: float = 1.0   # stop must be >= 1x ATR away (no noise-tight stops)
    max_stop_atr_multiple: float = 3.5   # stop must be <= 3.5x ATR away (no sloppy sizing)
    min_quant_confluence: float = 0.75   # >=75% of rule-based indicators must agree with LLM's direction
    # Hard veto on entering an already-stretched move. The confluence gate
    # counts RSI as a BUY vote whenever rsi > 50, so RSI 85 votes for buying —
    # it selects for extension rather than filtering it. These two cap that.
    max_buy_rsi: float = 75.0
    min_sell_rsi: float = 25.0
    # Phase 1 is long-only. That is deliberately NOT a setting: SELL signals are
    # short sales, the paper account has no short path at all, and an env var
    # that lets them back in would silently restore the exact dishonesty the
    # `long_only` gate in app/signals/validation.py exists to remove.

    # ---- Trading frictions ----
    # Realistic all-in round-trip cost for NSE intraday equity with a discount
    # broker: brokerage + STT + exchange txn charge + GST + stamp duty + SEBI fee
    # (~0.082%), plus conservative spread/slippage (~0.04%). Backtests that omit
    # this are not slightly optimistic — at this trade frequency they flip the
    # sign of the expectancy. Broker-dependent, so configurable.
    cost_pct_round_trip: float = 0.12
    # An LLM-proposed entry far from the last traded price would never have
    # filled. Cap it at a fraction of ATR so the backtest cannot book imaginary
    # fills.
    max_entry_drift_atr: float = 0.25

    # ---- Agent / LLM budgets ----
    agent_max_tool_rounds: int = 6       # chat agent tool-calling iterations
    chat_turn_budget_seconds: float = 55.0  # wall clock ceiling for one chat turn
    signal_max_tool_rounds: int = 4      # signal-generation tool-calling iterations
    chat_history_turns: int = 8          # how much conversation history to resend
    sentiment_headline_limit: int = 10   # cost control on the FinBERT call
    # Third budget for a chat turn, alongside rounds and wall clock. The tool
    # schemas alone are ~3,400 tokens and are resent every round, so a turn that
    # runs its full round cap costs ~20k tokens before a single candle. Without
    # a ceiling here, one question could cost more than a day of normal use.
    chat_token_budget: int = 60_000
    # Tool results are appended to the transcript and resent on every later
    # round, so an oversized one is paid for repeatedly. Beyond this it is
    # replaced by its summary.
    max_tool_result_chars: int = 6_000
    # Per-tool call limit within one turn — a fourth request for the same series
    # is a loop, not research.
    max_calls_per_tool: int = 3
    # Route greetings and product questions to a small agent that holds no tool
    # schemas, so they cannot trigger a full analysis. Asked "hi", the analyst
    # ran four rounds and spent 14,299 tokens. Off switches the front desk off
    # and sends everything straight to the analyst.
    triage_enabled: bool = True

    # ---- Risk limits (chat agent sizing tools) ----
    # No single position may exceed this share of the account, regardless of how
    # tight the stop is. Stop distance controls loss *if the stop holds*;
    # position size controls loss when it gaps.
    max_position_pct: float = 20.0
    min_risk_pct: float = 0.25           # clamp floor on requested per-trade risk
    max_risk_pct: float = 2.0            # clamp ceiling — nothing may pass 25%
    watchlist_scan_limit: int = 15       # symbols the scan_watchlist tool will fetch

    # ---- Morning brief ----
    brief_driver_move_threshold: float = 1.0   # % overnight move to count as a driver
    brief_min_expected_move: float = 0.15      # % implied move below which a reason is noise
    brief_universe: str = ""                   # comma-separated override of DEFAULT_UNIVERSE
    # One candidate per sector by default. Five picks that are all banks are
    # one bet, not five, and they lose together.
    brief_max_per_sector: int = 1

    # ---- Concurrency ----
    # Bounded, not unbounded — yfinance rate-limits aggressively.
    screener_concurrency: int = 5
    brief_concurrency: int = 4

    # FinBERT via Hugging Face Inference API (free tier — no local torch needed)
    # Get a free token at https://huggingface.co/settings/tokens
    hf_api_token: str = ""
    finbert_model: str = "ProsusAI/finbert"

    # pydantic-settings defaults to extra='forbid', so any key present in .env
    # but not declared here raises at import time. Docker hides this (compose
    # injects via environment, where extras are ignored), so the failure only
    # appears on developer machines — "works in Docker, broken locally".
    # Ignore unknown keys instead.
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Process-wide settings, resolved once.

    Cached rather than a module-level singleton so tests can reset it:
        get_settings.cache_clear()
    """
    return Settings()


def __getattr__(name: str):
    """Lazy module attribute so `from app.config import settings` keeps working.

    Kept because a module-level `settings = Settings()` would re-introduce the
    import-time resolution this module exists to avoid. Prefer `get_settings()`
    in new code.
    """
    if name == "settings":
        return get_settings()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
