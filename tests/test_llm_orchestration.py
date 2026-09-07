"""
Unit tests for the LLM-as-orchestrator features added to
StrategyHunterService: feasibility-aware re-querying (Feature 1) and
self-critique (Feature 2).

Scope, stated plainly (see also the accompanying delivery notes): these
tests confirm the TRIGGER LOGIC and DATA FLOW are correct -- that the
right signal fires under the right synthetic conditions, that the LLM
gets called with the right data when and only when it should, and that
its response is threaded into the right place in the response (budget_note
/ rationale) without ever touching a numeric field. They do NOT and
cannot confirm that a real LLM's actual sentences are insightful rather
than generic -- FakeLLMClient returns a fixed, injected response
regardless of input, by design, so it can prove the wiring works without
depending on network access or a real provider. Whether the real
synthesis reads as genuinely useful is a human judgment call against
real Groq/DeepSeek output, explicitly out of scope for an automated test.
"""
from __future__ import annotations

import datetime as dt

from frisks.config import AppConfig
from frisks.engine.models import Constraints, Direction, Objective, StrategyRequest
from frisks.service import NoValidStrategiesResult, StrategyHunterResult, StrategyHunterService, _EngineRun
from tests.conftest import build_chain, near_expiry_ms
from tests.test_service_integration import FakeLLMClient, FakeMarketDataService


def _request(asset="BTC", direction=Direction.BULLISH, max_loss=500.0, objective=Objective.RISK_ADJUSTED_RETURN):
    return StrategyRequest(
        asset=asset,
        expiry_date="2026-09-25",
        days_out=None,
        direction=direction,
        max_loss=max_loss,
        objective=objective,
        target_cost=None,
        constraints=Constraints(max_legs=4, max_expiries=2),
    )


def _engine_run(strategies_payload, meta, request=None):
    return _EngineRun(
        strategies_payload=strategies_payload,
        meta=meta,
        ranked=[],
        candidates=[],
        request=request or _request(),
    )


def _strategy(rank=1, structure_name="bull call spread", pop=0.5, rationale="base rationale"):
    return {
        "rank": rank,
        "structure_name": structure_name,
        "legs": [],
        "entry_cost": 100.0,
        "max_profit": 500.0,
        "max_loss": 100.0,
        "breakeven": [101000.0],
        "probability_of_profit": pop,
        "greeks": {"delta": 0.3, "gamma": 0.0001, "theta": -1.0, "vega": 10.0},
        "liquidity": {"min_volume_24h": 5.0, "max_spread_pct": 3.0, "open_interest": 20.0},
        "objective_score": 5.0,
        "rationale": rationale,
    }


def _base_meta(evaluated=10, excluded_budget=0):
    return {
        "candidates_evaluated": evaluated,
        "candidates_excluded_illiquid": 3,
        "candidates_excluded_budget": excluded_budget,
        "constraints_applied": {"max_legs": 4, "max_expiries": 2},
    }


def _service() -> StrategyHunterService:
    return StrategyHunterService(config=AppConfig(), market_data=FakeMarketDataService({}), llm_client=FakeLLMClient())


# -- Feature 1, Signal A: budget-excluded fraction --------------------------


def test_signal_a_fires_when_budget_exclusion_dominates():
    svc = _service()
    request = _request(direction=Direction.NEUTRAL)  # isolate signal A from signal B
    run = _engine_run([_strategy()], _base_meta(evaluated=2, excluded_budget=8), request=request)

    signals = svc._check_feasibility_signals(request, run)
    assert any("budget_excluded_fraction" in s for s in signals)


def test_signal_a_does_not_fire_when_budget_exclusion_is_minor():
    svc = _service()
    request = _request(direction=Direction.NEUTRAL)
    run = _engine_run([_strategy()], _base_meta(evaluated=9, excluded_budget=1), request=request)

    signals = svc._check_feasibility_signals(request, run)
    assert not any("budget_excluded_fraction" in s for s in signals)


# -- Feature 1, Signal B: weak-fit structure for a directional request ------


def test_signal_b_fires_for_calendar_spread_on_directional_request():
    svc = _service()
    request = _request(direction=Direction.BULLISH)
    run = _engine_run([_strategy(structure_name="call calendar spread")], _base_meta(), request=request)

    signals = svc._check_feasibility_signals(request, run)
    assert any("weak_fit_top_structure" in s for s in signals)


def test_signal_b_does_not_fire_for_vertical_spread_on_directional_request():
    svc = _service()
    request = _request(direction=Direction.BULLISH)
    run = _engine_run([_strategy(structure_name="bull call spread")], _base_meta(), request=request)

    signals = svc._check_feasibility_signals(request, run)
    assert not any("weak_fit_top_structure" in s for s in signals)


def test_signal_b_never_fires_for_neutral_direction():
    """A calendar spread is a perfectly reasonable top pick for a neutral
    request -- the weak-fit check only applies to directional requests."""
    svc = _service()
    request = _request(direction=Direction.NEUTRAL)
    run = _engine_run([_strategy(structure_name="call calendar spread")], _base_meta(), request=request)

    signals = svc._check_feasibility_signals(request, run)
    assert not any("weak_fit_top_structure" in s for s in signals)


def test_no_signals_fire_for_a_clean_well_budgeted_request():
    """The 'should trigger neither' control case -- confirms the system
    stays quiet when nothing is actually wrong."""
    svc = _service()
    request = _request(direction=Direction.BULLISH)
    run = _engine_run([_strategy(structure_name="bull call spread", pop=0.5)], _base_meta(evaluated=20, excluded_budget=0), request=request)

    signals = svc._check_feasibility_signals(request, run)
    assert signals == []


# -- Feature 1, Signal C: soft, session-relative PoP drop -------------------


def test_signal_c_fires_after_enough_history_and_a_real_drop():
    svc = _service()
    request = _request(direction=Direction.NEUTRAL, asset="BTC")

    # Build up session history with 3 "normal" PoP samples -- no signal
    # expected yet (fewer prior samples than the minimum).
    for _ in range(3):
        run = _engine_run([_strategy(structure_name="bull call spread", pop=0.6)], _base_meta(), request=request)
        signals = svc._check_feasibility_signals(request, run)
        assert not any("session_pop_drop" in s for s in signals)

    # 4th call, same (asset, direction) key, with a much lower PoP: now
    # there are 3 prior samples averaging 0.6, and 0.15 is well below
    # 0.6 * default drop ratio (0.5) = 0.30.
    run = _engine_run([_strategy(structure_name="bull call spread", pop=0.15)], _base_meta(), request=request)
    signals = svc._check_feasibility_signals(request, run)
    assert any("session_pop_drop" in s for s in signals)


def test_signal_c_does_not_fire_across_different_asset_direction_keys():
    """Session history is keyed per (asset, direction) -- a drop for ETH
    bearish shouldn't be influenced by BTC bullish history."""
    svc = _service()
    btc_bullish = _request(asset="BTC", direction=Direction.BULLISH)
    eth_bearish = _request(asset="ETH", direction=Direction.BEARISH)

    for _ in range(3):
        run = _engine_run([_strategy(pop=0.6)], _base_meta(), request=btc_bullish)
        svc._check_feasibility_signals(btc_bullish, run)

    # First-ever sample for the ETH/bearish key -- not enough history to
    # compare against, regardless of how low this PoP is.
    run = _engine_run([_strategy(pop=0.05)], _base_meta(), request=eth_bearish)
    signals = svc._check_feasibility_signals(eth_bearish, run)
    assert not any("session_pop_drop" in s for s in signals)


# -- Feature 1: full re-query wiring (LLM gets called only when triggered) --


def test_maybe_run_feasibility_requery_calls_llm_and_returns_note_when_triggered():
    expiry = near_expiry_ms(30)
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    quotes = build_chain("BTCUSDT", 100.0, strikes, expiry)
    fake_llm = FakeLLMClient(budget_note_response="Raising the budget unlocks a cleaner directional trade.")
    svc = StrategyHunterService(
        config=AppConfig(), market_data=FakeMarketDataService({expiry: quotes}), llm_client=fake_llm
    )
    request = StrategyRequest(
        asset="BTC", expiry_date=_expiry_date_str(expiry), days_out=None, direction=Direction.NEUTRAL,
        max_loss=500.0, objective=Objective.RISK_ADJUSTED_RETURN, target_cost=None,
        constraints=Constraints(max_legs=4, max_expiries=2),
    )
    # Force-trigger via a synthetic primary result with heavy budget exclusion,
    # but supply REAL prefetched market data so the relaxed re-query is a
    # genuine second engine pass, not a mock.
    primary = _engine_run([_strategy()], _base_meta(evaluated=1, excluded_budget=9), request=request)
    prefetched = (
        {expiry: quotes},
        100.0,
        svc._resolve_and_fetch(request),
    )

    note = svc._maybe_run_feasibility_requery(request, primary, prefetched)

    assert note == "Raising the budget unlocks a cleaner directional trade."
    assert len(fake_llm.synthesize_budget_note_calls) == 1
    request_summary, results_json = fake_llm.synthesize_budget_note_calls[0]
    assert "original_max_loss=500.0" in request_summary
    assert "relaxed_max_loss=1500.0" in request_summary  # default 3x multiplier
    assert "original_budget_top_result" in results_json
    assert "relaxed_budget_top_result" in results_json


def test_maybe_run_feasibility_requery_skips_llm_call_when_no_signal_fires():
    expiry = near_expiry_ms(30)
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    quotes = build_chain("BTCUSDT", 100.0, strikes, expiry)
    fake_llm = FakeLLMClient()
    svc = StrategyHunterService(
        config=AppConfig(), market_data=FakeMarketDataService({expiry: quotes}), llm_client=fake_llm
    )
    request = StrategyRequest(
        asset="BTC", expiry_date=_expiry_date_str(expiry), days_out=None, direction=Direction.NEUTRAL,
        max_loss=500.0, objective=Objective.RISK_ADJUSTED_RETURN, target_cost=None,
        constraints=Constraints(max_legs=4, max_expiries=2),
    )
    primary = _engine_run([_strategy(structure_name="bull call spread")], _base_meta(evaluated=20, excluded_budget=0), request=request)
    prefetched = ({expiry: quotes}, 100.0, svc._resolve_and_fetch(request))

    note = svc._maybe_run_feasibility_requery(request, primary, prefetched)

    assert note is None
    assert fake_llm.synthesize_budget_note_calls == []


# -- Feature 2: self-critique data flow --------------------------------------


def test_self_critique_appends_caveat_only_to_flagged_rank():
    fake_llm = FakeLLMClient(self_critique_response={"1": "This delta is only marginally bullish."})
    svc = StrategyHunterService(config=AppConfig(), market_data=FakeMarketDataService({}), llm_client=fake_llm)
    request = _request()
    strategies = [
        _strategy(rank=1, rationale="rank 1 base rationale"),
        _strategy(rank=2, rationale="rank 2 base rationale"),
    ]
    primary = _engine_run(strategies, _base_meta(), request=request)

    svc._apply_self_critique(request, primary)

    assert "marginally bullish" in primary.strategies_payload[0]["rationale"]
    assert primary.strategies_payload[0]["rationale"].startswith("rank 1 base rationale")
    # rank 2 got no key in the fake response -> untouched, per spec ("omit
    # a key entirely if no caveat needed" -- absence must not add anything).
    assert primary.strategies_payload[1]["rationale"] == "rank 2 base rationale"


def test_self_critique_leaves_rationale_untouched_when_no_caveat_returned():
    fake_llm = FakeLLMClient(self_critique_response={})
    svc = StrategyHunterService(config=AppConfig(), market_data=FakeMarketDataService({}), llm_client=fake_llm)
    request = _request()
    strategies = [_strategy(rank=1, rationale="original rationale")]
    primary = _engine_run(strategies, _base_meta(), request=request)

    svc._apply_self_critique(request, primary)

    assert primary.strategies_payload[0]["rationale"] == "original rationale"


def test_self_critique_sends_real_computed_fields_not_a_summary():
    """Confirms the LLM is handed the actual computed numbers (delta,
    probability_of_profit, objective_score, meta) rather than something
    reconstructed/approximated -- the whole point of 'reason over real
    engine output, never invent'."""
    fake_llm = FakeLLMClient()
    svc = StrategyHunterService(config=AppConfig(), market_data=FakeMarketDataService({}), llm_client=fake_llm)
    request = _request()
    strategies = [_strategy(rank=1, pop=0.12)]  # low-PoP-high-score-style case
    meta = _base_meta(evaluated=3)  # thin candidate pool
    primary = _engine_run(strategies, meta, request=request)

    svc._apply_self_critique(request, primary)

    assert len(fake_llm.self_critique_calls) == 1
    _request_summary, strategies_json = fake_llm.self_critique_calls[0]
    assert "0.12" in strategies_json
    assert '"candidates_evaluated": 3' in strategies_json


# -- graceful degradation & the enable/disable switch ------------------------


def test_handle_structured_request_survives_llm_failure_without_crashing():
    class RaisingLLMClient(FakeLLMClient):
        def self_critique(self, *args, **kwargs):
            raise RuntimeError("provider is down")

        def synthesize_budget_note(self, *args, **kwargs):
            raise RuntimeError("provider is down")

    expiry = near_expiry_ms(30)
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    quotes = build_chain("BTCUSDT", 100.0, strikes, expiry)
    svc = StrategyHunterService(
        config=AppConfig(), market_data=FakeMarketDataService({expiry: quotes}), llm_client=RaisingLLMClient()
    )

    result = svc.handle_structured_request(
        {
            "asset": "BTC",
            "horizon": {"expiry_date": _expiry_date_str(expiry)},
            "direction": "bullish",
            "max_loss": 500.0,
        }
    )

    assert isinstance(result, StrategyHunterResult)
    assert result.budget_note is None  # feasibility requery failed soft -> no note, not a crash
    assert result.strategies  # deterministic engine result still returned in full


def test_orchestration_disabled_makes_zero_llm_calls():
    from frisks.config import AppConfig as _AppConfig

    expiry = near_expiry_ms(30)
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    quotes = build_chain("BTCUSDT", 100.0, strikes, expiry)
    fake_llm = FakeLLMClient()

    import os

    os.environ["FRISKS_LLM_ORCHESTRATION_ENABLED"] = "false"
    try:
        cfg = _AppConfig()
        assert cfg.orchestration.enabled is False
        svc = StrategyHunterService(config=cfg, market_data=FakeMarketDataService({expiry: quotes}), llm_client=fake_llm)
        result = svc.handle_structured_request(
            {
                "asset": "BTC",
                "horizon": {"expiry_date": _expiry_date_str(expiry)},
                "direction": "bullish",
                "max_loss": 500.0,
            }
        )
        assert isinstance(result, (StrategyHunterResult, NoValidStrategiesResult))
        assert fake_llm.self_critique_calls == []
        assert fake_llm.synthesize_budget_note_calls == []
    finally:
        del os.environ["FRISKS_LLM_ORCHESTRATION_ENABLED"]


def _expiry_date_str(expiry_ms: int) -> str:
    return dt.datetime.fromtimestamp(expiry_ms / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d")
