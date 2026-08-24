"""
The tool registry.

One name -> one handler, assembled from four modules that each own a coherent
group. Groups are named here rather than inferred, because the next step needs
them: offering all seventeen schemas on every LLM round costs ~3,400 tokens per
round, and account tools are useless when there is no authenticated user.

Adding a tool means adding a function to the right module and a line to that
module's `TOOLS` — the registry, the labels and the schemas are then the only
three places that mention it.
"""
from __future__ import annotations

from app.signals.agent.tools import account, chart, chart_indicators, graph_agent, market, strategy, web
from app.signals.agent.tools.base import Handler, ToolContext

GROUPS: dict[str, dict[str, Handler]] = {
    "market": market.TOOLS,
    "account": account.TOOLS,
    "strategy": strategy.TOOLS,
    "chart": chart.TOOLS,
    "chart_indicators": chart_indicators.TOOLS,
    "graph_agent": graph_agent.TOOLS,
    "web": web.TOOLS,
}

REGISTRY: dict[str, Handler] = {
    name: handler for group in GROUPS.values() for name, handler in group.items()
}

# Which group a tool belongs to — the reverse index, built once.
GROUP_OF: dict[str, str] = {
    name: group for group, tools in GROUPS.items() for name in tools
}


def get(name: str) -> Handler | None:
    return REGISTRY.get(name)


__all__ = ["GROUPS", "GROUP_OF", "REGISTRY", "Handler", "ToolContext", "get"]
