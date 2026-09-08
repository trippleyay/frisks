"""
Isolated timing of the multi-expiry expensive-check cap, on identical
pre-fetched market data (no network noise in the measured section).

Fetches the BTC/2026-09-25 snapshot once via the real Binance client, then
times generate_candidates for (a) the capped default (300) and (b) an
uncapped run (1M), on the SAME data, reporting the expensive-check call
count in each case. This isolates the fix's actual effect from fetch and
network variance.
"""
from __future__ import annotations

import time
import unittest.mock as mock

from frisks.engine.generator import generate_candidates
from frisks.service import StrategyHunterService
from frisks.config import load_config


def main() -> None:
    cfg = load_config()
    service = StrategyHunterService(cfg)
    try:
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
        request = service._normalize_request(raw)
        snapshot = service._resolve_and_fetch(request)
        quotes_by_expiry = service._gather_quotes_by_expiry(request, snapshot)
        underlying_price = service._estimate_underlying_price(snapshot)
        risk_free_rate = service._representative_risk_free_rate(snapshot)
        total_liquid = sum(len(qs) for qs in quotes_by_expiry.values())
        print(f"liquid quotes: {total_liquid} across {len(quotes_by_expiry)} expiries; spot~{underlying_price:.0f}")

        from frisks.engine import generator as gen

        def timed(cap: int) -> tuple[float, int, int]:
            calls = {"n": 0}
            real = gen.build_payoff_model

            def spy(candidate, underlying, rfr, gp, gs):
                calls["n"] += 1
                return real(candidate, underlying, rfr, gp, gs)

            with mock.patch.object(gen, "build_payoff_model", spy):
                t0 = time.perf_counter()
                candidates, _gen, _excl = generate_candidates(
                    quotes_by_expiry=quotes_by_expiry,
                    underlying_price=underlying_price,
                    direction=request.direction,
                    max_loss_budget=request.max_loss,
                    constraints=request.constraints,
                    primary_expiry_ms=snapshot.expiry_ms,
                    risk_free_rate=risk_free_rate,
                    grid_points=cfg.engine.distribution_grid_points,
                    grid_sigmas=cfg.engine.distribution_grid_sigmas,
                    max_expensive_checks=cap,
                )
                elapsed = time.perf_counter() - t0
            return elapsed, calls["n"], len(candidates)

        e_uncapped, c_uncapped, n_uncapped = timed(1_000_000)
        print(f"UNCAPPED : generate_candidates={e_uncapped:7.2f}s  expensive_checks={c_uncapped:6d}  candidates={n_uncapped}")
        e_capped, c_capped, n_capped = timed(300)
        print(f"CAPPED300: generate_candidates={e_capped:7.2f}s  expensive_checks={c_capped:6d}  candidates={n_capped}")
        print(f"speedup  : {e_uncapped / e_capped:.1f}x   expensive_checks {c_uncapped} -> {c_capped}")
    finally:
        service.close()


if __name__ == "__main__":
    main()