"""
FastMCP adapter exposing Frisks' StrategyHunterService as an MCP tool.

This package is additive only -- it does not modify frisks/api/,
frisks/service.py, or any engine/data-layer module. Both the FastAPI
route (frisks/api/routes.py) and the tool defined in frisks/mcp/server.py
call the exact same StrategyHunterService.handle_structured_request
method, so the two adapters cannot silently diverge in behavior.

See frisks/mcp/server.py for the tool definition and payment-gate seam,
and frisks/mcp/__main__.py for the standalone stdio entrypoint
(`python -m frisks.mcp`).
"""
