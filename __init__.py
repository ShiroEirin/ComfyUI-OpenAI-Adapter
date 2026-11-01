"""ComfyUI-OpenAI-Adapter — expose workflows through the OpenAI Images API.

Drop an API-format workflow into ./workflows as ``<name>.json`` (plus an optional
``<name>.meta.json`` describing node ids and the parameter schema), and it becomes a
model that any OpenAI-compatible client — Snow App's imagegen channel, for instance —
can call at ``POST /v1/images/generations``.

No extra process, no extra port, no bridge config: the routes live inside ComfyUI.
"""
from .adapter import register_routes

# No node classes are contributed; the plugin only adds HTTP routes. ComfyUI still
# needs the mapping to exist so the module loads as a plugin.
NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

register_routes()
