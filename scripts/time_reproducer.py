"""
Local latency reproducer for the multi-expiry candidate latency fix.

Times a full StrategyHunterService.handle_structured_request against live
Binance data (BTC, bullish, max_loss=5000, risk_adjusted_return), exactly
the reproducer from the handoff note. The expensive multi-expiry check cap is
read from the FRISKS_MAX_EXPENSIVE_CHECKS env var at import/process start
(via frisks.config), so run this script once with the env var unset/large to
get the "before" number and again with =300 to get the "after".

Run from repo root:
  FRISKS_MAX_EXPENSIVE_CHECKS=100000 python scripts/time_reproducer.py   # ~"before"
  FRISKS_MAX_EXPENSIVE_CHECKS=300     python scripts/time_reproducer.py   # "after"
"""
from __future__ import annotations

import os
import time

from frisks.config import EngineConfig, load_config


def main() -> None:
    from frisks.service import StrategyHunterService

    cfg = load_config()
    print(f"FRISKS_MAX_EXPENSIVE_CHECKS={cfg.engine.max_expensive_checks}")

    raw = {
        "schema_version": "1.0",
        "asset": "BTC",
        "horizon": {"expiry_date": "2026-09-25"},
        "direction": "bullish",
        "max_loss": 5000,
        "objective": "risk_adjusted_return",
        "target_cost": None,
        "constraints": {"max_legs": 4, "max_expiries": 2},
    }

    service = StrategyHunterService(cfg)
    try:
        t0 = time.perf_counter()
        result = service.handle_structured_request(raw)
        elapsed = time.perf_counter() - t0
        print(f"elapsed_seconds={elapsed:.2f}")
        name = type(result).__name__
        n_strategies = len(getattr(result, "strategies", []))
        print(f"result_type={name} num_strategies={n_strategies} request_id={getattr(result, 'request_id', '')[:12]}")
        meta = getattr(result, "meta", None)
        if meta:
            print(f"meta={meta}")
    finally:
        service.close()


if __name__ == "__main__":
    main()