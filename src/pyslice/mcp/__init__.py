"""MCP tools for driving PySlice simulations from LLM clients.

Run the server with ``python -m pyslice.mcp`` (requires the ``mcp`` extra:
``pip install 'pyslice[mcp]'``). See :mod:`pyslice.mcp.server` for the tool
catalog and :mod:`pyslice.mcp.service` for the underlying service layer.
"""
from pyslice.mcp.server import build_server, main
from pyslice.mcp.service import PySliceService

__all__ = ["PySliceService", "build_server", "main"]
