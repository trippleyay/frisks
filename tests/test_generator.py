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
    candidates, _stats = generate_candidates(
        quotes_by_expiry={expiry: quotes},
        underlying_price=100.0,
        direction=Direction.NEUTRAL,
        max_loss_budget=50.0,
        constraints=Constraints(max_legs=4, max_expiries=2),
        max_nodes=5000,
    )
    for c in candidates:
        assert c.entry_cost <= 50.0 + 1e-6


def test_generate_candidates_respects_max_legs():
    quotes, expiry = _chain()
    candidates, _stats = generate_candidates(
        quotes_by_expiry={expiry: quotes},
        underlying_price=100.0,
        direction=Direction.NEUTRAL,
        max_loss_budget=10_000.0,
        constraints=Constraints(max_legs=2, max_expiries=1),
        max_nodes=5000,
    )
    assert candidates
    for c in candidates:
        assert len(c.legs) <= 2
        assert len(c.expiries) <= 1


def test_no_candidate_has_redundant_inverse_legs():
    quotes, expiry = _chain()
    candidates, _stats = generate_candidates(
        quotes_by_expiry={expiry: quotes},
        underlying_price=100.0,
        direction=Direction.NEUTRAL,
        max_loss_budget=10_000.0,
        constraints=Constraints(max_legs=4, max_expiries=2),
        max_nodes=5000,
    )
    for c in candidates:
        symbols_sides = [(leg.symbol, leg.side, leg.quantity) for leg in c.legs]
        for symbol, side, qty in symbols_sides:
            opposite = (symbol, OrderSide.SELL if side is OrderSide.BUY else OrderSide.BUY, qty)
            assert symbols_sides.count(opposite) == 0 or symbols_sides.count((symbol, side, qty)) == 0
