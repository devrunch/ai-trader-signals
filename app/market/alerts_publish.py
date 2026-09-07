"""Shared publish call for both alert-producing pipelines (drift_check,
reddit_sentiment) -- one POST to the same internal ai-trader-api endpoint,
same shape either way. Mirrors app/signals/brief.py's own `publish`."""
from __future__ import annotations

import logging

import httpx

from app.config import get_settings

logger = logging.getLogger(__name__)


async def publish(alert: dict) -> bool:
    settings = get_settings()
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            r = await client.post(
                f"{settings.api_service_url}/api/internal/alerts",
                headers={"x-internal-key": settings.internal_api_key},
                json=alert,
            )
            r.raise_for_status()
        logger.info("Alert published: %s", alert.get("title"))
        return True
    except httpx.HTTPError as e:
        logger.warning("Alert publish failed: %s", e)
        return False
