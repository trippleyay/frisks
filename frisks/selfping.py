"""
Cold-start mitigation for Render's free Hobby tier, shared by both
services (frisks/api/app.py for FastAPI, frisks/mcp/__main__.py for the
MCP streamable-http server).

VERIFIED BEHAVIOR (checked against Render's own docs and community
answers, not assumed -- this exact point matters enough to get right):
Render's free web services spin down after ~15 minutes WITHOUT INBOUND
HTTP OR WEBSOCKET TRAFFIC. Internal process activity -- CPU usage, a
background asyncio task doing work entirely in-process -- does NOT reset
that timer by itself. The only thing that resets it is a genuine request
arriving at the service's public URL from outside. Render's own
community answers are explicit that keep-alive pinging is a workaround,
not an officially supported guarantee ("I would not treat that as a
reliable or supported fix").

Given that, this module's background task makes REAL outbound HTTP GETs
-- to the service's own public URL (via Render's auto-injected
RENDER_EXTERNAL_URL) and to the sibling service's public URL (via
FRISKS_PEER_HEALTH_URL, set explicitly in render.yaml) -- not in-process
function calls. A GET to your own public hostname is a genuine round
trip through Render's edge, which is what actually counts as "inbound
traffic" from Render's perspective. This is still a best-effort layer,
not a guarantee: if Render enforces something stricter in the future, or
if both services happen to go idle at exactly the same moment before
either one's ping fires, a cold start can still occur. The fully reliable
mechanism -- and the one to actually rely on for a judging window -- is
an external uptime pinger (UptimeRobot or similar) hitting both public
URLs on a schedule; that is a manual, one-time signup only Trip can do
(see the README runbook). This module is a free, zero-signup complement
to that, not a replacement for it.

No-ops safely in any environment where RENDER_EXTERNAL_URL isn't set
(i.e. anywhere other than an actual Render deployment -- local dev,
tests, CI) so it's always safe to wire in.
"""
from __future__ import annotations

import asyncio
import logging
import os

import httpx

logger = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 600.0  # 10 minutes, per the build prompt


def _self_health_url(own_health_path: str) -> str | None:
    """
    RENDER_EXTERNAL_URL is a Render-injected env var (runtime-only, web
    services only) giving this service's own public https URL -- exactly
    what's needed to ping "ourselves" as genuine external traffic rather
    than an in-process call. Returns None outside Render (safe no-op).
    """
    base = os.environ.get("RENDER_EXTERNAL_URL")
    if not base:
        return None
    return base.rstrip("/") + own_health_path


def _peer_health_url() -> str | None:
    """
    Explicitly configured in render.yaml (service-to-service reference)
    -- the sibling service's own health URL. Not auto-derivable the way
    RENDER_EXTERNAL_URL is, since Render doesn't expose "the other
    service's URL" automatically; render.yaml wires this via `fromService`.
    """
    return os.environ.get("FRISKS_PEER_HEALTH_URL") or None


async def run_self_ping_loop(
    own_health_path: str,
    interval_s: float = DEFAULT_INTERVAL_S,
    client: httpx.AsyncClient | None = None,
) -> None:
    """
    Runs forever (intended to be launched as a background asyncio task
    from a service's startup/lifespan hook). Pings its own public health
    URL and, if configured, its peer's, every `interval_s` seconds.
    Every failure is caught and logged -- a ping failure must never crash
    the service it's trying to keep warm.
    """
    self_url = _self_health_url(own_health_path)
    peer_url = _peer_health_url()

    if not self_url and not peer_url:
        logger.info(
            "Self-ping loop not started: neither RENDER_EXTERNAL_URL nor "
            "FRISKS_PEER_HEALTH_URL is set (expected outside an actual Render "
            "deployment -- this is a normal no-op locally/in tests)."
        )
        return

    owns_client = client is None
    http_client = client or httpx.AsyncClient(timeout=10.0)
    logger.info(
        "Self-ping loop started: self_url=%s peer_url=%s interval_s=%s",
        self_url, peer_url, interval_s,
    )
    try:
        while True:
            for label, url in (("self", self_url), ("peer", peer_url)):
                if not url:
                    continue
                try:
                    resp = await http_client.get(url)
                    logger.debug("Self-ping (%s) %s -> %s", label, url, resp.status_code)
                except Exception as exc:  # noqa: BLE001 -- a ping failure must never crash the loop
                    logger.warning("Self-ping (%s) to %s failed: %s", label, url, exc)
            await asyncio.sleep(interval_s)
    finally:
        if owns_client:
            await http_client.aclose()
