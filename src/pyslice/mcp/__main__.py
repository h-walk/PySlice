"""Command-line entry point for the PySlice MCP server.

Allows ``python -m pyslice.mcp [--workspace DIR]`` to start the stdio
server.
"""
import sys

from pyslice.mcp.server import main

if __name__ == "__main__":
    sys.exit(main())
