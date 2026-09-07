"""
Entrypoint: `python -m frisks.mcp`.

Supports two transports, selected via FRISKS_MCP_TRANSPORT
(default "stdio"):

  stdio (default, for local testing with Claude Code):
      claude mcp add frisks -- python -m frisks.mcp

      Run that from the project root, using whichever Python interpreter
      actually has requirements.txt and frisks/mcp/requirements.txt
      installed. Claude Code does not inherit an already-activated shell
      virtualenv for a stdio server it spawns, so if `python` on PATH
      isn't the right interpreter, point at it explicitly:
          claude mcp add frisks -- /path/to/venv/bin/python -m frisks.mcp
      To remove it again: `claude mcp remove frisks`.

  streamable-http (for the Render deployment -- see render.yaml):
      FRISKS_MCP_TRANSPORT=streamable-http python -m frisks.mcp
      Binds 0.0.0.0:$PORT (Render injects PORT; defaults to 8000
      locally), mounted at /mcp -- so the deployed MCP config URL is
      https://<host>/mcp, and a lightweight /health route (added in
      server.py via @mcp.custom_route) is available for Render's health
      check and the self-ping loop.

FastMCP-CLI note: `fastmcp run frisks/mcp/server.py` (FastMCP's own CLI)
is also a valid way to start this server for the stdio case -- it
auto-detects the `mcp` instance in server.py -- but per FastMCP's own
documented behavior, `fastmcp run` completely ignores this file's
`if __name__ == "__main__"` block, so it would skip the self-ping task
started here for the HTTP case. Use `python -m frisks.mcp` for the
streamable-http/Render path; either entrypoint is fine for local stdio
testing since there's nothing else to lose there.
"""
from __future__ import annotations

import asyncio
import logging
import os
import threading

from frisks.mcp.server import mcp
from frisks.selfping import run_self_ping_loop


def _run_self_ping_in_background_thread() -> None:
    """
    The streamable-http path below (mcp.run(...)) blocks the main thread
    running its own event loop internally -- there's no straightforward
    lifespan hook to hang a background asyncio task off of the way
    FastAPI's is (see frisks/api/app.py). Simplest reliable approach: run
    the self-ping loop's own tiny event loop in a daemon thread. It only
    ever does HTTP GETs on a timer (see frisks/selfping.py) -- no shared
    state with the MCP server -- so plain thread isolation is sufficient
    here without needing to hook into FastMCP's own loop.
    """

    def _worker() -> None:
        asyncio.run(run_self_ping_loop(own_health_path="/health"))

    thread = threading.Thread(target=_worker, name="frisks-mcp-self-ping", daemon=True)
    thread.start()


def main() -> None:
    from frisks.config import load_config
    from frisks.logging_setup import configure_logging

    cfg = load_config()
    configure_logging(cfg.log_level)

    transport = os.environ.get("FRISKS_MCP_TRANSPORT", "stdio").lower()

    if transport == "streamable-http":
        port = int(os.environ.get("PORT", "8000"))
        _run_self_ping_in_background_thread()
        logging.getLogger(__name__).info(
            "Starting Frisks MCP server on streamable-http, 0.0.0.0:%s, path=/mcp", port
        )
        mcp.run(transport="streamable-http", host="0.0.0.0", port=port, path="/mcp")
    elif transport == "stdio":
        mcp.run()  # defaults to stdio
    else:
        raise ValueError(f"Unknown FRISKS_MCP_TRANSPORT '{transport}' -- expected 'stdio' or 'streamable-http'")


if __name__ == "__main__":
    main()
