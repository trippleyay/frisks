from frisks.config import LiquidityConfig
from frisks.engine.generator import generate_candidates
from frisks.engine.models import Constraints, Direction, Objective
from frisks.engine.payoff import build_payoff_model
from frisks.engine.scoring import rank_candidates
from tests.conftest import build_chain, near_expiry_ms


def _setup(objective: Objective, target_cost=None):
    expiry = near_expiry_ms(30)
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    quotes = build_chain("BTCUSDT", 100.0, strikes, expiry)
    candidates, _, _excl_stats = generate_candidates(
        quotes_by_expiry={expiry: quotes},
        underlying_price=100.0,
        direction=Direction.NEUTRAL,
        max_loss_budget=2000.0,
        constraints=Constraints(max_legs=4, max_expiries=1),
        primary_expiry_ms=expiry,
        max_nodes=6000,
    )
    payoff_models = [(c, build_payoff_model(c, 100.0, 0.0, 200, 6.0)) for c in candidates]
    ranked = rank_candidates(
        scored_payoffs=payoff_models,
        objective=objective,
        liquidity_cfg=LiquidityConfig(),
        target_cost=target_cost,
        max_loss_constraint=2000.0,
        top_n=3,
    )
    return ranked


def test_risk_adjusted_return_ranking_is_descending():
    ranked = _setup(Objective.RISK_ADJUSTED_RETURN)
    assert len(ranked) <= 3
    scores = [r.final_score for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_probability_of_profit_ranking_is_descending():
    ranked = _setup(Objective.PROBABILITY_OF_PROFIT)
    scores = [r.payoff_model.probability_of_profit() for r in ranked]
    assert scores == sorted(scores, reverse=True)
    for r in ranked:
        assert 0.0 <= r.payoff_model.probability_of_profit() <= 1.0


def test_cost_for_target_minimizes_distance():
    ranked = _setup(Objective.COST_FOR_TARGET, target_cost=100.0)
    assert ranked
    distances = [abs(r.candidate.entry_cost - 100.0) for r in ranked]
    assert distances == sorted(distances)
