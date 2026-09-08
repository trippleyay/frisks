from frisks.engine.generator import _bounded_worst_case_loss, _net_call_slope, generate_candidates
from frisks.engine.models import Constraints, Direction, Leg, OrderSide
from tests.conftest import build_chain, near_expiry_ms


def _chain():
    expiry = near_expiry_ms(21)
    strikes = [85, 90, 95, 100, 105, 110, 115]
    return build_chain("BTCUSDT", 100.0, strikes, expiry), expiry


def test_naked_short_call_is_unbounded():
    quotes, _ = _chain()
    call = next(q for q in quotes if q.contract.side.value == "CALL" and q.contract.strike == 100)
    leg = Leg(side=OrderSide.SELL, quote=call, quantity=1.0)
    assert _net_call_slope([leg]) < 0
    assert _bounded_worst_case_loss([leg], running_cost=0.0) is None


def test_covered_call_spread_is_bounded():
    quotes, _ = _chain()
    low = next(q for q in quotes if q.contract.side.value == "CALL" and q.contract.strike == 100)
    high = next(q for q in quotes if q.contract.side.value == "CALL" and q.contract.strike == 110)
    legs = [Leg(side=OrderSide.BUY, quote=low), Leg(side=OrderSide.SELL, quote=high)]
    assert _net_call_slope(legs) >= 0
    worst = _bounded_worst_case_loss(legs, running_cost=0.0)
    assert worst is not None
    assert worst >= 0


def test_generate_candidates_respects_max_loss_budget():
    quotes, expiry = _chain()
    candidates, _stats, _excl_stats = generate_candidates(
        quotes_by_expiry={expiry: quotes},
        underlying_price=100.0,
        direction=Direction.NEUTRAL,
        max_loss_budget=50.0,
        constraints=Constraints(max_legs=4, max_expiries=2),
        primary_expiry_ms=expiry,
        risk_free_rate=0.0,
        grid_points=200,
        grid_sigmas=6.0,
        max_nodes=5000,
    )
    for c in candidates:
        assert c.entry_cost <= 50.0 + 1e-6


def test_generate_candidates_respects_max_legs():
    quotes, expiry = _chain()
    candidates, _stats, _excl_stats = generate_candidates(
        quotes_by_expiry={expiry: quotes},
        underlying_price=100.0,
        direction=Direction.NEUTRAL,
        max_loss_budget=10_000.0,
        constraints=Constraints(max_legs=2, max_expiries=1),
        primary_expiry_ms=expiry,
        risk_free_rate=0.0,
        grid_points=200,
        grid_sigmas=6.0,
        max_nodes=5000,
    )
    assert candidates
    for c in candidates:
        assert len(c.legs) <= 2
        assert len(c.expiries) <= 1


def test_no_candidate_has_redundant_inverse_legs():
    quotes, expiry = _chain()
    candidates, _stats, _excl_stats = generate_candidates(
        quotes_by_expiry={expiry: quotes},
        underlying_price=100.0,
        direction=Direction.NEUTRAL,
        max_loss_budget=10_000.0,
        constraints=Constraints(max_legs=4, max_expiries=2),
        primary_expiry_ms=expiry,
        risk_free_rate=0.0,
        grid_points=200,
        grid_sigmas=6.0,
        max_nodes=5000,
    )
    for c in candidates:
        symbols_sides = [(leg.symbol, leg.side, leg.quantity) for leg in c.legs]
        for symbol, side, qty in symbols_sides:
            opposite = (symbol, OrderSide.SELL if side is OrderSide.BUY else OrderSide.BUY, qty)
            assert symbols_sides.count(opposite) == 0 or symbols_sides.count((symbol, side, qty)) == 0


def test_every_candidate_includes_a_leg_at_the_requested_expiry():
    """
    Regression test for the exact reported bug: with a second (later)
    expiry fetched only to support calendar structures, every returned
    candidate must still include at least one leg at the requested
    (primary) expiry -- the primary expiry must never be entirely
    replaced by the additional expiries.
    """
    strikes = [90, 95, 100, 105, 110, 115, 120]
    primary_expiry = near_expiry_ms(19)
    other_expiry = near_expiry_ms(54)
    primary_quotes = build_chain("BTCUSDT", 100.0, strikes, primary_expiry)
    other_quotes = build_chain("BTCUSDT", 100.0, strikes, other_expiry)

    candidates, _stats, _excl_stats = generate_candidates(
        quotes_by_expiry={primary_expiry: primary_quotes, other_expiry: other_quotes},
        underlying_price=100.0,
        direction=Direction.NEUTRAL,
        max_loss_budget=5000.0,
        constraints=Constraints(max_legs=4, max_expiries=2),
        primary_expiry_ms=primary_expiry,
        risk_free_rate=0.0,
        grid_points=200,
        grid_sigmas=6.0,
        max_nodes=8000,
    )
    assert candidates, "expected at least one candidate"
    for c in candidates:
        assert primary_expiry in c.expiries, (
            f"candidate {c.structure_name} with expiries {c.expiries} has no leg at the "
            f"requested expiry {primary_expiry}"
        )


def test_multi_expiry_expensive_check_is_capped(monkeypatch):
    """
    Regression test for the latency fix (bug fix #5): given more
    multi-expiry survivors than `max_expensive_checks`, only the top N by
    cheap intrinsic worst-case loss may reach the expensive payoff-model
    check. We spy on build_payoff_model to count real calls; with a cap of
    2, at most 2 calls may happen, and with a huge cap the same input
    produces many more calls -- proving the cap is what bounds it, not an
    artifact of the fixture producing few candidates.
    """
    import frisks.engine.generator as generator_module
    from frisks.engine.payoff import build_payoff_model as real_build_payoff_model

    strikes = [90, 95, 100, 105, 110, 115, 120]
    primary_expiry = near_expiry_ms(19)
    other_expiry = near_expiry_ms(54)
    quotes_by_expiry = {
        primary_expiry: build_chain("BTCUSDT", 100.0, strikes, primary_expiry),
        other_expiry: build_chain("BTCUSDT", 100.0, strikes, other_expiry),
    }
    kwargs = dict(
        quotes_by_expiry=quotes_by_expiry,
        underlying_price=100.0,
        direction=Direction.NEUTRAL,
        max_loss_budget=5000.0,
        constraints=Constraints(max_legs=4, max_expiries=2),
        primary_expiry_ms=primary_expiry,
        risk_free_rate=0.0,
        grid_points=200,
        grid_sigmas=6.0,
        max_nodes=8000,
    )
    call_count = {"n": 0}

    def spy(candidate, underlying_price, risk_free_rate, grid_points, grid_sigmas):
        call_count["n"] += 1
        return real_build_payoff_model(
            candidate, underlying_price, risk_free_rate, grid_points, grid_sigmas
        )

    monkeypatch.setattr(generator_module, "build_payoff_model", spy)

    # Capped: at most `max_expensive_checks` expensive checks.
    call_count["n"] = 0
    generate_candidates(**kwargs, max_expensive_checks=2)
    assert call_count["n"] <= 2, f"expensive check ran {call_count['n']} times, expected <= 2"

    # Sanity: the same input with a huge cap produces many more calls, so the
    # cap above is what actually excluded them (test is not vacuous).
    call_count["n"] = 0
    generate_candidates(**kwargs, max_expensive_checks=100_000)
    assert call_count["n"] > 300, f"expected far more than 2 expensive checks uncapped, got {call_count['n']}"
