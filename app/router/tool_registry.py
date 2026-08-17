"""
tool_registry.py
─────────────────
A lightweight tool registry for the Trase query router.

## Design
Tools are plain Python callables with a standardized signature:
    fn(query: str, **kwargs) -> str

To add a new tool in the future:
    1. Implement the function anywhere (e.g., app/tools/fare_calculator.py).
    2. Import it here and call register_tool(...).
    3. That's it — the router and intent classifier will automatically know
       about it via get_tool_descriptions().

## Example (future usage)
    from app.tools.fare_calculator import calculate_fare
    register_tool(
        name="fare_calculator",
        fn=calculate_fare,
        description="Calculates exact bus fares between two stops given the route name and number of stops.",
    )
"""

from typing import Callable

# ──────────────────────────────────────────────────────────────────────────────
# Internal registry store
# ──────────────────────────────────────────────────────────────────────────────

_TOOL_REGISTRY: dict[str, dict] = {}
# Shape: { "tool_name": { "fn": Callable, "description": str } }


# ──────────────────────────────────────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────────────────────────────────────

def register_tool(name: str, fn: Callable, description: str) -> None:
    """
    Register a new callable tool.

    Args:
        name:        Unique snake_case identifier (e.g. "fare_calculator").
        fn:          Python callable — must accept (query: str, **kwargs) -> str.
        description: One-sentence plain-English description used by the intent
                     classifier so the LLM understands what the tool does.
    """
    if name in _TOOL_REGISTRY:
        raise ValueError(
            f"Tool '{name}' is already registered. Use a unique name or "
            "call unregister_tool() first."
        )
    _TOOL_REGISTRY[name] = {"fn": fn, "description": description}
    print(f"  [ToolRegistry] Registered tool: '{name}'")


def unregister_tool(name: str) -> None:
    """Remove a tool from the registry (useful for testing)."""
    _TOOL_REGISTRY.pop(name, None)


def get_tool(name: str) -> Callable:
    """
    Returns the callable for a registered tool.
    Raises KeyError if the tool does not exist.
    """
    entry = _TOOL_REGISTRY.get(name)
    if not entry:
        raise KeyError(f"No tool named '{name}' is registered.")
    return entry["fn"]


def list_tools() -> list[str]:
    """Returns a list of all registered tool names."""
    return list(_TOOL_REGISTRY.keys())


def get_tool_descriptions() -> dict[str, str]:
    """
    Returns {tool_name: description} for every registered tool.
    Passed to the intent classifier so the LLM knows what tools exist.
    """
    return {name: entry["description"] for name, entry in _TOOL_REGISTRY.items()}


def has_tools() -> bool:
    """Returns True if at least one tool is currently registered."""
    return bool(_TOOL_REGISTRY)
