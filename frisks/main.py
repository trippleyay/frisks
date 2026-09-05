"""Run with: python -m frisks.main  (or: uvicorn frisks.api.app:app)"""
from __future__ import annotations

import os

import uvicorn


def main() -> None:
    uvicorn.run(
        "frisks.api.app:app",
        host=os.environ.get("FRISKS_HOST", "0.0.0.0"),
        port=int(os.environ.get("FRISKS_PORT", "8000")),
        reload=bool(os.environ.get("FRISKS_RELOAD")),
    )


if __name__ == "__main__":
    main()
