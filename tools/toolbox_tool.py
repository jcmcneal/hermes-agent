"""Toolbox gateway — single root tool for all tool interactions.

The schema, command dispatch, and helper logic all live in ``toolbox_gateway``.
This file just wires the registry into Hermes and registers the gateway
as a single discoverable tool.

All tool calls go through toolbox. The LLM discovers tools at runtime:
``list`` → ``explain`` → ``run`` → ``hints``.

Requires: ``toolbox-gateway`` package (pip install toolbox-gateway)
"""

import json
import logging

from toolbox_gateway import GATEWAY_TOOL_NAME, Toolbox, is_available
from toolbox_gateway.backends.sqlite_store import SQLiteHintStore

from tools.registry import registry

logger = logging.getLogger(__name__)


def _tool_definitions():
    """Return the registry's tool list in toolbox-compatible format."""
    return [
        {
            "name": entry.name,
            "description": entry.description or entry.schema.get("description", ""),
            "schema": entry.schema.get("parameters", {}),
        }
        for entry in registry._snapshot_entries()
        if entry.name != GATEWAY_TOOL_NAME
        and (not entry.check_fn or entry.check_fn())
    ]


def _dispatch_tool(name: str, args: dict) -> dict:
    """Dispatch a tool call through the Hermes registry."""
    result = registry.dispatch(name, args)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return {"result": result}
    return result


def _get_toolbox():
    """Lazy-init the Toolbox singleton, with tools fetched lazily from the registry."""
    global _toolbox_instance
    if _toolbox_instance is None:
        from hermes_constants import get_hermes_home
        hint_db = get_hermes_home() / "toolbox_hints.db"
        hint_store = SQLiteHintStore(path=str(hint_db))
        _toolbox_instance = Toolbox.from_provider(
            provider=_tool_definitions,
            dispatcher=_dispatch_tool,
            hint_store=hint_store,
        )
    return _toolbox_instance


def toolbox_handler(args: dict, **kwargs) -> str:
    """Handle toolbox tool calls by dispatching to the toolbox-gateway library."""
    try:
        tb = _get_toolbox()
        tb.refresh()  # pick up any newly registered tools
        result = tb.handle_args(args)
        return json.dumps(result.to_dict() if hasattr(result, "to_dict") else result)
    except Exception as e:
        logger.exception("Toolbox handler error: %s", e)
        return json.dumps({"success": False, "error": str(e)})


def _get_schema() -> dict:
    """Get the canonical toolbox schema from toolbox-gateway."""
    if not is_available():
        return {}
    from toolbox_gateway import Toolbox
    return Toolbox.get_tool_definition()


_toolbox_instance = None

registry.register(
    name=GATEWAY_TOOL_NAME,
    toolset=GATEWAY_TOOL_NAME,
    schema=_get_schema(),
    handler=toolbox_handler,
    check_fn=is_available,
    emoji="🧰",
)
