"""
Exercises StrategyHunterService end to end (normalize -> fetch -> generate
-> score -> assemble response) against a fake MarketDataService built from
synthetic quotes, so it runs without httpx/network and without a real LLM
key. The LLM client is not invoked by handle_structured_request at all
(by design — see service.py), so this validates the engine path the
locked /v1/strategies contract actually depends on.
"""
from __future__ import annotations

from dataclasses import dataclass

from frisks.config import AppConfig
from frisks.data.market_data import MarketSnapshot
from frisks.service import NoValidStrategiesResult, StrategyHunterResult, StrategyHunterService
from tests.conftest import build_chain, near_expiry_ms


class FakeMarketDataService:
    def __init__(self, quotes_by_expiry: dict[int, list]):
        self._quotes_by_expiry = quotes_by_expiry

    def close(self) -> None:
        pass

    def list_expiries(self, underlying_asset: str) -> list[int]:
        return sorted(self._quotes_by_expiry)

    def nearest_expiry(self, underlying_asset: str, expiry_date_str, days_out) -> int:
        expiries = self.list_expiries(underlying_asset)
        if expiry_date_str is not None:
            import datetime as dt

            target = dt.datetime.strptime(expiry_date_str, "%Y-%m-%d").date()
            for e in expiries:
                if dt.datetime.fromtimestamp(e / 1000, tz=dt.timezone.utc).date() == target:
                    return e
            raise ValueError("no matching expiry")
        return expiries[0]

    def get_snapshot(self, underlying_asset: str, expiry_ms: int) -> MarketSnapshot:
        quotes = self._quotes_by_expiry.get(expiry_ms, [])
        return MarketSnapshot(underlying=f"{underlying_asset}USDT", expiry_ms=expiry_ms, quotes=quotes, excluded_illiquid_count=7)


class FakeLLMClient:
    def close(self):
        pass


def _service_with_chain():
    expiry = near_expiry_ms(30)
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    quotes = build_chain("BTCUSDT", 100.0, strikes, expiry)
    fake_md = FakeMarketDataService({expiry: quotes})
    service = StrategyHunterService(config=AppConfig(), market_data=fake_md, llm_client=FakeLLMClient())
    return service, expiry


def test_end_to_end_returns_ranked_strategies_matching_contract_shape():
    service, expiry = _service_with_chain()
    import datetime as dt

    expiry_date = dt.datetime.fromtimestamp(expiry / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d")

    raw_request = {
        "schema_version": "1.0",
        "asset": "BTC",
        "horizon": {"expiry_date": expiry_date},
        "direction": "bullish",
        "max_loss": 500,
        "objective": "risk_adjusted_return",
        "target_cost": None,
        "constraints": {"max_legs": 4, "max_expiries": 1},
    }
    result = service.handle_structured_request(raw_request)
    assert isinstance(result, StrategyHunterResult)
    assert result.asset == "BTC"
    assert result.objective_used == "risk_adjusted_return"
    assert 1 <= len(result.strategies) <= 3

    for i, strat in enumerate(result.strategies, start=1):
        assert strat["rank"] == i
        assert strat["legs"]
        for leg in strat["legs"]:
            assert leg["side"] in ("BUY", "SELL")
            assert leg["symbol"]
        assert "delta" in strat["greeks"]
        assert "min_volume_24h" in strat["liquidity"]
        assert isinstance(strat["objective_score"], float)

    assert result.meta["candidates_excluded_illiquid"] == 7
    assert result.meta["constraints_applied"] == {"max_legs": 4, "max_expiries": 1}


def test_no_valid_strategies_when_budget_too_small():
    service, expiry = _service_with_chain()
    import datetime as dt

    expiry_date = dt.datetime.fromtimestamp(expiry / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d")
    raw_request = {
        "schema_version": "1.0",
        "asset": "BTC",
        "horizon": {"expiry_date": expiry_date},
        "direction": "bullish",
        "max_loss": 0.01,
        "objective": "risk_adjusted_return",
        "target_cost": None,
        "constraints": {"max_legs": 4, "max_expiries": 1},
    }
    result = service.handle_structured_request(raw_request)
    assert isinstance(result, NoValidStrategiesResult)
    assert result.asset == "BTC"
    assert "no liquid candidates" in result.message.lower() or "No liquid candidates" in result.message


def test_invalid_asset_rejected():
    service, expiry = _service_with_chain()
    import datetime as dt
    from frisks.service import InvalidRequestError

    expiry_date = dt.datetime.fromtimestamp(expiry / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d")
    raw_request = {
        "asset": "DOGE",
        "horizon": {"expiry_date": expiry_date},
        "direction": "neutral",
        "max_loss": 500,
    }
    try:
        service.handle_structured_request(raw_request)
        assert False, "expected InvalidRequestError"
    except InvalidRequestError:
        pass


def test_cost_for_target_objective():
    service, expiry = _service_with_chain()
    import datetime as dt

    expiry_date = dt.datetime.fromtimestamp(expiry / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d")
    raw_request = {
        "asset": "BTC",
        "horizon": {"expiry_date": expiry_date},
        "direction": "neutral",
        "max_loss": 500,
        "objective": "cost_for_target",
        "target_cost": 50.0,
        "constraints": {"max_legs": 4, "max_expiries": 1},
    }
    result = service.handle_structured_request(raw_request)
    assert isinstance(result, StrategyHunterResult)
    assert result.objective_used == "cost_for_target"
