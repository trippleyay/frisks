"""Run with: python -m frisks.main  (or: uvicorn frisks.api.app:app)"""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    # DEPLOYMENT FIX: Render injects the port to bind to via the `PORT`
    # env var (not a Frisks-specific name), and explicitly states it
    # fails the deploy if it can't detect the bound port. This
    # previously only read FRISKS_PORT, which Render never sets --
    # meaning this would have silently bound to the wrong port (the
    # hardcoded 8000 fallback) on an actual Render deployment. `PORT` is
    # now checked first (Render's real convention); FRISKS_PORT remains
    # as a manual-override fallback for anyone who was already setting
    # it locally, then 8000 for plain local dev with nothing set.
    port = int(os.environ.get("PORT") or os.environ.get("FRISKS_PORT", "8000"))
    uvicorn.run(
        "frisks.api.app:app",
        host=os.environ.get("FRISKS_HOST", "0.0.0.0"),
        port=port,
        reload=bool(os.environ.get("FRISKS_RELOAD")),
    )


if __name__ == "__main__":
    main()
