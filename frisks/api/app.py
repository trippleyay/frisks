from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI

from frisks.api.routes import router
from frisks.b402.middleware import PaymentGate
from frisks.config import AppConfig, load_config
from frisks.logging_setup import configure_logging
from frisks.service import StrategyHunterService


def create_app(config: AppConfig | None = None) -> FastAPI:
    cfg = config or load_config()
    configure_logging(cfg.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = StrategyHunterService(cfg)
        app.state.payment_gate = PaymentGate(enabled=cfg.b402_enabled)
        try:
            yield
        finally:
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
