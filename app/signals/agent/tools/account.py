"""
Account tools: the user's own book.

Thin adapters. Every calculation lives in `agent/portfolio_tools.py` as a pure
function over a context dict, so the maths that decides position size can be
tested against a fixture with no NestJS instance and no network. That module is
the code with the worst consequences when it is wrong — an earlier version
invented a ₹1,00,000 account and handed a ₹20,000 account 5x oversize.

Every tool here is scoped to the authenticated user: the book comes from
`ToolContext.book()`, which resolves through the user_id taken from the JWT.
"""
from __future__ import annotations

from typing import Any

from app.signals.agent import portfolio_tools
from app.signals.agent.tools.base import Handler, ToolContext


async def get_portfolio(ctx: ToolContext, args: dict) -> Any:
    return portfolio_tools.get_portfolio(await ctx.book())


async def get_positions(ctx: ToolContext, args: dict) -> Any:
    return portfolio_tools.get_positions(await ctx.book())


async def analyse_exposure(ctx: ToolContext, args: dict) -> Any:
    return portfolio_tools.analyse_exposure(await ctx.book())


async def risk_limits(ctx: ToolContext, args: dict) -> Any:
    return portfolio_tools.risk_limits(await ctx.book())


async def position_size(ctx: ToolContext, args: dict) -> Any:
    return portfolio_tools.position_size(
        await ctx.book(),
        float(args["entry"]),
        float(args["stop"]),
        float(args.get("risk_pct") or 1.0),
        ctx.settings,
    )


async def portfolio_risk(ctx: ToolContext, args: dict) -> Any:
    return portfolio_tools.portfolio_risk(
        await ctx.book(),
        float(args.get("stop_pct") or 2.0),
        float(args.get("market_drop_pct") or 2.0),
    )


TOOLS: dict[str, Handler] = {
    "get_portfolio": get_portfolio,
    "get_positions": get_positions,
    "analyse_exposure": analyse_exposure,
    "risk_limits": risk_limits,
    "position_size": position_size,
    "portfolio_risk": portfolio_risk,
}
