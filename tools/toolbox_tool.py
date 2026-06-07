"""Toolbox gateway — single root tool for all tool interactions.

The schema, command dispatch, and helper logic all live in ``toolbox_gateway``.
This file just wires the registry into Hermes: builds a ``Toolbox`` instance
from the registry's tool list, registers it as a single discoverable tool,
and proxies calls to ``toolbox.handle()``.

All tool calls go through toolbox. The LLM discovers tools at runtime:
``list`` → ``explain`` → ``run`` → ``hints``.

Requires: ``toolbox-gateway`` package (pip install toolbox-gateway)
"""

import json
import logging

from tools.registry import registry

logger = logging.getLogger(__name__)


def _get_toolbox():
    """Lazy-init the Toolbox singleton with all registry tools and SQLite hints."""
    global _toolbox_instance
    if _toolbox_instance is None:
        from toolbox_gateway import Toolbox, Tool
        from toolbox_gateway.backends.sqlite_store import SQLiteHintStore

        from hermes_constants import get_hermes_home

        hint_db = get_hermes_home() / "toolbox_hints.db"
        hint_store = SQLiteHintStore(path=str(hint_db))

        tools = [
            Tool(
                name=entry.name,
                description=entry.description or entry.schema.get("description", ""),
                schema=entry.schema.get("parameters", {}),
                execute=lambda args, _name=entry.name, **kw: _dispatch_tool(_name, args),
            )
            for entry in registry._snapshot_entries()
            if entry.name != "toolbox"
            and (not entry.check_fn or entry.check_fn())
        ]

        _toolbox_instance = Toolbox(
            tools=tools,
            hint_store=hint_store,
            schema_format="markdown",  # compact markdown is the whole point
        )
    return _toolbox_instance


def _dispatch_tool(name: str, args: dict) -> dict:
    """Dispatch a tool call through the Hermes registry."""
    result = registry.dispatch(name, args)
    if isinstance(result, str):
        try:
            return json.loads(result)
        except (json.JSONDecodeError, TypeError):
            return {"result": result}
    return result


def _reset_toolbox():
    """Force re-initialization on next call (tools may have changed)."""
    global _toolbox_instance
    _toolbox_instance = None


def toolbox_handler(args: dict, **kwargs) -> str:
    """Handle toolbox tool calls by dispatching to the toolbox-gateway library."""
    try:
        _reset_toolbox()
        tb = _get_toolbox()
        result = tb.handle(
            command=args.get("command", ""),
            toolNames=args.get("toolNames"),
            toolName=args.get("toolName"),
            subject=args.get("subject"),
            args=args.get("args", {}),
            format=args.get("format"),
            method=args.get("method"),
            category=args.get("category"),
            key=args.get("key"),
            hint=args.get("hint"),
            mcp=args.get("mcp"),
        )
        return json.dumps(result.to_dict() if hasattr(result, "to_dict") else result)
    except Exception as e:
        logger.exception("Toolbox handler error: %s", e)
        return json.dumps({"success": False, "error": str(e)})


def check_toolbox_requirements() -> bool:
    """Toolbox is available if the toolbox-gateway package is installed."""
    try:
        import toolbox_gateway  # noqa: F401
        return True
    except ImportError:
        return False


def _get_schema() -> dict:
    """Get the canonical toolbox schema from toolbox-gateway."""
    if not check_toolbox_requirements():
        return {}
    from toolbox_gateway import Toolbox
    return Toolbox(tools=[]).get_tool_definition()


_toolbox_instance = None

registry.register(
    name="toolbox",
    toolset="toolbox",
    schema=_get_schema(),
    handler=toolbox_handler,
    check_fn=check_toolbox_requirements,
    emoji="🧰",
)
