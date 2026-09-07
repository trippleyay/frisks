from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI

from frisks.api.routes import router
from frisks.b402.middleware import PaymentGate
from frisks.config import AppConfig, load_config
from frisks.logging_setup import configure_logging
from frisks.selfping import run_self_ping_loop
from frisks.service import StrategyHunterService


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    configure_logging(cfg.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = StrategyHunterService(cfg)
        app.state.payment_gate = PaymentGate(enabled=cfg.b402_enabled)
        # Cold-start mitigation (deployment build) -- see frisks/selfping.py
        # for the verified reasoning and honest limitations. No-ops safely
        # anywhere RENDER_EXTERNAL_URL isn't set (i.e. everywhere except an
        # actual Render deployment), so this is always safe to start.
        ping_task = asyncio.create_task(run_self_ping_loop(own_health_path="/healthz"))
        try:
            yield
        finally:
            ping_task.cancel()
            try:
                await ping_task
            except asyncio.CancelledError:
                pass
            app.state.service.close()

    app = FastAPI(
        title="Frisks",
        description="Agent-native crypto options Strategy Hunter",
        version="1.0",
        lifespan=lifespan,
    )
    app.include_router(router)
    return app


app = create_app()
