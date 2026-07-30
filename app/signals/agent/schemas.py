"""
Tool schemas advertised to the chat agent's LLM.

Declarative only — no logic. Split out because the schema list is ~210 lines
and shares nothing with the code that executes the calls.

Design rules (see docs/agent-roadmap/):
  * The agent supplies INTENT; deterministic maths supplies numbers. No price
    level, size, or statistic in a tool result is authored by the model.
  * Tools are read-only in Phase 1. Nothing here places an order or mutates
    state — execution tools are a later phase and must be confirmation-gated.
  * Every tool result is plain JSON-serialisable data.
"""
from __future__ import annotations

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "read_chart",
            "description": (
                "Read the chart: indicators AND support/resistance/trend/Fibonacci levels "
                "in one call. Prefer this over calling get_indicators and get_levels "
                "separately — it is the same data in one step."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {
                        "type": "array", "items": {"type": "string"},
                        "description": "Indicators to compute. Omit for the default set.",
                    },
                    "interval": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "1d"]},
                    "symbol": {"type": "string", "description": "Defaults to the chart symbol"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_candles",
            "description": "Fetch recent OHLCV candles for a symbol at a chosen timeframe. Use to inspect real price action, check a lower timeframe for a recent reversal, or a higher timeframe for trend context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "interval": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "1d"]},
                    "count": {"type": "integer", "description": "How many recent candles. Up to 30 are returned in full; ask for more and you also get a summary of the wider window."},
                    "symbol": {"type": "string", "description": "Defaults to the chart symbol"},
                },
                "required": ["interval"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_indicators",
            "description": (
                "Compute technical indicators for a symbol. Request exactly what you need by name. "
                "Available: rsi, stochastic, stochrsi, macd, williams_r, cci, mfi, roc, tsi, "
                "ultimate_oscillator, adx, aroon, ema, sma, hma, bollinger, keltner, donchian, "
                "supertrend, psar, ichimoku, atr, obv, cmf, volume, vwap. "
                "Omit 'names' for a sensible default set."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "names": {"type": "array", "items": {"type": "string"}, "description": "Indicator names to compute"},
                    "interval": {"type": "string", "enum": ["5m", "15m", "1h", "1d"]},
                    "symbol": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "detect_patterns",
            "description": "Detect candlestick patterns (doji, hammer, engulfing, marubozu, inside bar, morning/evening star) over recent bars. Patterns are weak alone — combine with trend and level context.",
            "parameters": {
                "type": "object",
                "properties": {
                    "lookback": {"type": "integer", "description": "How many recent bars to scan (default 10, max 40)"},
                    "interval": {"type": "string", "enum": ["5m", "15m", "1h", "1d"]},
                    "symbol": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_symbols",
            "description": "Head-to-head comparison of two or more symbols on trend, momentum, volatility and distance to key levels. Use when the user asks which setup is better.",
            "parameters": {
                "type": "object",
                "properties": {
                    "symbols": {"type": "array", "items": {"type": "string"}, "description": "2-4 symbols"},
                },
                "required": ["symbols"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "scan_watchlist",
            "description": "Screen the user's watchlist and rank stocks by setup quality — trend strength, momentum, volume and proximity to support/resistance. Use for 'what's setting up?' style questions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "criteria": {
                        "type": "string",
                        "enum": ["near_support", "near_resistance", "trending", "oversold", "overbought", "volume_spike", "all"],
                        "description": "What to look for; 'all' returns a general ranking",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "add_chart_indicator",
            "description": "Add or remove indicator overlays on the user's chart (e.g. RSI, MACD, Bollinger, EMA). Use when the user asks to see something on the chart.",
            "parameters": {
                "type": "object",
                "properties": {
                    "add": {"type": "array", "items": {"type": "string"}, "description": "Chart indicators to show: EMA, MA, BOLL, SAR, VOL, MACD, RSI, KDJ"},
                    "remove": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_levels",
            "description": "Compute support/resistance levels, the current trend line, and the recent Fibonacci swing for a symbol. Returns exact prices computed from swing structure.",
            "parameters": {
                "type": "object",
                "properties": {"symbol": {"type": "string"}},
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_portfolio",
            "description": "The user's paper trading account: cash, total value, realised and unrealised P&L. Use before giving any advice that depends on account size or affordability.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_positions",
            "description": "The user's currently open positions with quantity, average cost, current price and unrealised P&L.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "analyse_exposure",
            "description": "Concentration analysis of the user's open positions — how much of the book sits in each symbol, and how much capital is deployed vs free.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "position_size",
            "description": "Calculate how many shares to buy so that a stop-out costs no more than a chosen percentage of the account. Always use this instead of estimating size yourself.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entry": {"type": "number"},
                    "stop": {"type": "number"},
                    "risk_pct": {"type": "number", "description": "Percent of account to risk, e.g. 1 for 1%. Defaults to 1."},
                },
                "required": ["entry", "stop"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "portfolio_risk",
            "description": "The user's real open risk. There are no stop-loss orders in this product, so the honest answer is that open risk is unbounded — this returns that plus the deployed value and a clearly-labelled hypothetical. Never quote the hypothetical as if stops existed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "stop_pct": {"type": "number", "description": "Assumed stop distance percent per position, default 2"},
                    "market_drop_pct": {"type": "number", "description": "What-if adverse move, default 2"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "risk_limits",
            "description": "The hard limits the paper account enforces at order placement — max concurrent positions, max aggregate open risk, daily loss limit — and where the user currently stands against each. Call this before suggesting a new position, so you do not propose a trade that will be refused.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "backtest_strategy",
            "description": "Backtest a standard rule strategy on a symbol's history. Returns trade count, win rate and total return. Report sample size honestly — small samples are unreliable.",
            "parameters": {
                "type": "object",
                "properties": {
                    "strategy": {"type": "string", "enum": ["ma_cross", "rsi", "macd", "bollinger"]},
                    "symbol": {"type": "string"},
                    "stop_pct": {"type": "number", "description": "Stop-loss distance percent. Defaults to 2. Costs are always applied."},
                },
                "required": ["strategy"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "build_strategy",
            "description": (
                "Design and backtest a CUSTOM rule-based strategy from a specification. "
                "Use this when the user describes trading rules in their own words "
                "('buy when RSI drops under 30 while price is above the 50-EMA, exit at RSI 65 "
                "or a 1.5x ATR stop') rather than naming one of the four preset strategies. "
                "Entries fill at the next bar's open and round-trip costs are deducted, so the "
                "result is what the rules would actually have produced. "
                "Report the trade count with every result — a strategy with 6 trades has told "
                "you almost nothing, however good its win rate looks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Short name for the strategy"},
                    "symbol": {"type": "string", "description": "Defaults to the chart symbol"},
                    "interval": {"type": "string", "enum": ["5m", "15m", "1h", "1d"]},
                    "entry": {
                        "type": "object",
                        "description": (
                            "Condition tree. A leaf is "
                            "{indicator, params?, op, value} or {indicator, params?, op, compare_to, compare_params?}. "
                            "Combine with {\"all\": [...]}, {\"any\": [...]} or {\"not\": {...}}. "
                            "Operators: <, <=, >, >=, ==, !=, above, below, crosses_above, crosses_below. "
                            "Indicators: close, open, high, low, volume, rsi, ema, sma, atr, adx, cci, "
                            "williams_r, mfi, roc, macd, macd_hist, macd_signal, bb_upper, bb_mid, "
                            "bb_lower, supertrend_dir, vwap, volume_ratio. "
                            "Params: length, fast, slow, signal, std, multiplier."
                        ),
                    },
                    "exit": {
                        "type": "object",
                        "description": (
                            "Same shape as entry, and may additionally contain "
                            "{\"type\": \"stop_loss\", \"atr_multiple\": 1.5} or "
                            "{\"type\": \"take_profit\", \"percent\": 4}. "
                            "Always include a stop — prefer atr_multiple over percent, because a "
                            "fixed percentage is too tight on a volatile stock and too loose on a "
                            "quiet one."
                        ),
                    },
                },
                "required": ["entry", "exit"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_trade",
            "description": "Scenario maths for a proposed trade: profit at target, loss at stop, reward:risk, capital required.",
            "parameters": {
                "type": "object",
                "properties": {
                    "side": {"type": "string", "enum": ["BUY", "SELL"]},
                    "entry": {"type": "number"},
                    "target": {"type": "number"},
                    "stop": {"type": "number"},
                    "quantity": {"type": "integer"},
                },
                "required": ["side", "entry", "target", "stop"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "draw_on_chart",
            "description": "Draw computed structure on the user's chart. Geometry is calculated server-side from real price data — you only choose what to draw.",
            "parameters": {
                "type": "object",
                "properties": {
                    "what": {"type": "string", "enum": ["trendline", "support_resistance", "fibonacci"]},
                },
                "required": ["what"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "plot_series",
            "description": (
                "Plot ANY of these as a line on the chart, over a chosen lookback window: "
                "close, open, high, low, volume, rsi, ema, sma, atr, adx, cci, williams_r, mfi, roc, "
                "macd, macd_hist, macd_signal, bb_upper, bb_mid, bb_lower, supertrend_dir, vwap, "
                "volume_ratio, highest, lowest. "
                "Use this for anything not covered by draw_on_chart or add_chart_indicator — e.g. "
                "'the 5-bar highest high and lowest low' is two calls: series=highest length=5, "
                "then series=lowest length=5. Values are computed server-side from real bars; you "
                "only choose which series and what parameters."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "series": {"type": "string", "description": "One of the names listed above."},
                    "params": {
                        "type": "object",
                        "description": "e.g. {\"length\": 5}. Omit for each series's normal default.",
                        "properties": {
                            "length": {"type": "number"}, "fast": {"type": "number"},
                            "slow": {"type": "number"}, "signal": {"type": "number"},
                            "std": {"type": "number"}, "multiplier": {"type": "number"},
                        },
                    },
                    "label": {"type": "string", "description": "Short label shown next to the line, e.g. '5-bar high'."},
                    "interval": {"type": "string", "enum": ["1m", "5m", "15m", "1h", "1d"]},
                    "symbol": {"type": "string", "description": "Defaults to the chart symbol"},
                },
                "required": ["series"],
            },
        },
    },
]
