"""
Synthetic stress test to demonstrate the cap's behavior when there are far
more multi-expiry survivors than the cap -- the scenario the handoff's
~9,400 number describes, which today's live market (only ~482) does not
reproduce. Uses the synthetic quote fixtures from tests.conftest so the
candidate pool is independent of live Binance liquidity.

Two 16-strike expiries (wide chain => large combinatorial multi-expiry
candidate count), then times generate_candidates at cap=300 vs an
effectively-uncapped cap, counting expensive build_payoff_model calls.
The scale is kept modest so the uncapped (slow) run still completes in a
bounded time.
"""
from __future__ import annotations

import time
import unittest.mock as mock

from frisks.engine.generator import generate_candidates
from frisks.engine.models import Constraints, Direction
from frisks.engine import generator as gen
from tests.conftest import build_chain, near_expiry_ms


def main() -> None:
    strikes = list(range(85, 116))  # 16 strikes x 2 sides per expiry, 2 expiries
    primary = near_expiry_ms(19)
    far = near_expiry_ms(54)
    quotes_by_expiry = {
        primary: build_chain("BTCUSDT", 100.0, strikes, primary),
        far: build_chain("BTCUSDT", 100.0, strikes, far),
    }

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
                underlying_price=100.0,
                direction=Direction.NEUTRAL,
                max_loss_budget=5000.0,
                constraints=Constraints(max_legs=4, max_expiries=2),
                primary_expiry_ms=primary,
                risk_free_rate=0.0,
                grid_points=300,
                grid_sigmas=6.0,
                max_nodes=14_000,
                max_expensive_checks=cap,
            )
            elapsed = time.perf_counter() - t0
        print(f"  ...{cap=} done in {elapsed:.1f}s checks={calls['n']} candidates={len(candidates)}", flush=True)
        return elapsed, calls["n"], len(candidates)

    print("synthetic: 2 expiries, 16 strikes, max_nodes=14000", flush=True)
    e_uncapped, c_uncapped, n_uncapped = timed(1_000_000)
    e_capped, c_capped, n_capped = timed(300)
    print(f"UNCAPPED : checks={c_uncapped:6d}  candidates={n_uncapped:5d}  gen={e_uncapped:6.2f}s", flush=True)
    print(f"CAPPED300: checks={c_capped:6d}  candidates={n_capped:5d}  gen={e_capped:6.2f}s", flush=True)
    print(f"speedup={e_uncapped / e_capped:.1f}x   expensive_checks {c_uncapped} -> {c_capped}", flush=True)


if __name__ == "__main__":
    main()