"""
Unit tests for frisks/mcp/server.py -- argument mapping and error
handling for the `find_option_strategies` MCP tool.

Reuses FakeMarketDataService/FakeLLMClient from test_service_integration.py
(the same fixtures the FastAPI-route-equivalent tests use) rather than
inventing separate ones, per the build prompt: both adapters call the
identical StrategyHunterService.handle_structured_request, so they should
be checked against identical expected behavior and identical fixtures.

These tests exercise frisks.mcp.server's tool function directly (real
FastMCP leaves an @mcp.tool()-decorated function directly callable as a
plain Python function -- that's what makes this kind of unit test
possible against the real library too, not just this sandbox's stub).
They do NOT exercise the MCP wire protocol itself (tool discovery, JSON-
RPC framing, transport) -- see the accompanying manual-verification
report for that; a passing unit test here confirms the tool's logic is
correct, not that a real MCP client can discover and invoke it.
"""
from __future__ import annotations

import frisks.mcp.server as mcp_server
from frisks.b402.middleware import PaymentGate
from frisks.config import AppConfig
from frisks.service import StrategyHunterService
from tests.conftest import build_chain, near_expiry_ms
from tests.test_service_integration import FakeLLMClient, FakeMarketDataService

try:
    from fastmcp.exceptions import ToolError
except ImportError:  # pragma: no cover
    ToolError = Exception  # type: ignore[assignment, misc]


def _install_fake_service(quotes_by_expiry: dict[int, list]) -> None:
    """
    Monkeypatches the module-level singleton service in frisks.mcp.server.
    Safe to do repeatedly: `find_option_strategies` and
    `_find_option_strategies_impl` both reference `_service` as a
    module-level name, resolved at call time (not baked in at import
    time), so replacing the module attribute takes effect immediately for
    subsequent calls -- no need to reload the module between tests.
    """
    fake_md = FakeMarketDataService(quotes_by_expiry)
    mcp_server._service = StrategyHunterService(
        config=AppConfig(), market_data=fake_md, llm_client=FakeLLMClient()
    )


def _valid_chain():
    expiry = near_expiry_ms(30)
    strikes = [80, 85, 90, 95, 100, 105, 110, 115, 120]
    quotes = build_chain("BTCUSDT", 100.0, strikes, expiry)
    return expiry, quotes


def _expiry_date_str(expiry_ms: int) -> str:
    import datetime as dt

    return dt.datetime.fromtimestamp(expiry_ms / 1000, tz=dt.timezone.utc).strftime("%Y-%m-%d")


# -- argument mapping: valid requests ----------------------------------------


def test_valid_request_returns_ranked_strategies_matching_locked_contract_shape():
    expiry, quotes = _valid_chain()
    _install_fake_service({expiry: quotes})

    result = mcp_server.find_option_strategies(
        asset="BTC",
        direction="bullish",
        max_loss=500.0,
        expiry_date=_expiry_date_str(expiry),
    )

    assert isinstance(result, dict)
    assert result["schema_version"] == "1.0"
    assert result["asset"] == "BTC"
    assert result["objective_used"] == "risk_adjusted_return"
    assert 1 <= len(result["strategies"]) <= 3
    for i, strat in enumerate(result["strategies"], start=1):
        assert strat["rank"] == i
        assert strat["legs"]
        for leg in strat["legs"]:
            assert leg["side"] in ("BUY", "SELL")
    assert result["meta"]["candidates_excluded_illiquid"] == 7


def test_days_out_horizon_argument_mapping():
    """Confirms the days_out branch of the flattened horizon arguments works,
    not just expiry_date."""
    expiry, quotes = _valid_chain()
    _install_fake_service({expiry: quotes})

    result = mcp_server.find_option_strategies(
        asset="BTC", direction="neutral", max_loss=500.0, days_out=30,
    )
    assert isinstance(result, dict)
    assert "strategies" in result or result.get("error") == "no_valid_strategies"


def test_no_valid_strategies_returns_locked_error_shape_not_an_exception():
    expiry, quotes = _valid_chain()
    _install_fake_service({expiry: quotes})

    result = mcp_server.find_option_strategies(
        asset="BTC", direction="bullish", max_loss=0.01, expiry_date=_expiry_date_str(expiry),
    )
    assert result["error"] == "no_valid_strategies"
    assert "message" in result
    assert result["asset"] == "BTC"


def test_cost_for_target_objective_argument_mapping():
    expiry, quotes = _valid_chain()
    _install_fake_service({expiry: quotes})

    result = mcp_server.find_option_strategies(
        asset="BTC", direction="neutral", max_loss=500.0, expiry_date=_expiry_date_str(expiry),
        objective="cost_for_target", target_cost=50.0,
    )
    assert result["objective_used"] == "cost_for_target"


def test_max_legs_and_max_expiries_argument_mapping():
    expiry, quotes = _valid_chain()
    _install_fake_service({expiry: quotes})

    result = mcp_server.find_option_strategies(
        asset="BTC", direction="neutral", max_loss=500.0, expiry_date=_expiry_date_str(expiry),
        max_legs=2, max_expiries=1,
    )
    assert result["meta"]["constraints_applied"] == {"max_legs": 2, "max_expiries": 1}


# -- error handling: invalid arguments raise ToolError, not a raw crash ------


def test_both_expiry_date_and_days_out_raises_tool_error():
    expiry, quotes = _valid_chain()
    _install_fake_service({expiry: quotes})

    try:
        mcp_server.find_option_strategies(
            asset="BTC", direction="neutral", max_loss=500.0,
            expiry_date=_expiry_date_str(expiry), days_out=30,
        )
        assert False, "expected ToolError for both expiry_date and days_out given"
    except ToolError as e:
        assert "expiry_date" in str(e) or "days_out" in str(e)


def test_neither_expiry_date_nor_days_out_raises_tool_error():
    expiry, quotes = _valid_chain()
    _install_fake_service({expiry: quotes})

    try:
        mcp_server.find_option_strategies(asset="BTC", direction="neutral", max_loss=500.0)
        assert False, "expected ToolError when neither expiry_date nor days_out given"
    except ToolError:
        pass


def test_target_cost_without_cost_for_target_objective_raises_tool_error():
    expiry, quotes = _valid_chain()
    _install_fake_service({expiry: quotes})

    try:
        mcp_server.find_option_strategies(
            asset="BTC", direction="neutral", max_loss=500.0, expiry_date=_expiry_date_str(expiry),
            objective="risk_adjusted_return", target_cost=50.0,
        )
        assert False, "expected ToolError for target_cost given without cost_for_target objective"
    except ToolError:
        pass


def test_invalid_asset_raises_tool_error_not_a_raw_exception():
    expiry, quotes = _valid_chain()
    _install_fake_service({expiry: quotes})

    try:
        mcp_server.find_option_strategies(
            asset="DOGE", direction="neutral", max_loss=500.0, expiry_date=_expiry_date_str(expiry),
        )
        assert False, "expected ToolError for unsupported asset"
    except ToolError:
        pass


def test_non_positive_max_loss_raises_tool_error():
    expiry, quotes = _valid_chain()
    _install_fake_service({expiry: quotes})

    try:
        mcp_server.find_option_strategies(
            asset="BTC", direction="neutral", max_loss=0.0, expiry_date=_expiry_date_str(expiry),
        )
        assert False, "expected ToolError for non-positive max_loss"
    except ToolError:
        pass


def test_unresolvable_expiry_raises_tool_error_not_a_raw_valueerror():
    """
    FakeMarketDataService.nearest_expiry raises a plain ValueError for an
    expiry with no listed match -- confirms _find_option_strategies_impl
    maps that to a clean ToolError rather than letting the raw ValueError
    (or its message shape) leak to the MCP client.
    """
    expiry, quotes = _valid_chain()
    _install_fake_service({expiry: quotes})

    try:
        mcp_server.find_option_strategies(
            asset="BTC", direction="neutral", max_loss=500.0, expiry_date="2099-01-01",
        )
        assert False, "expected ToolError for an unresolvable expiry"
    except ToolError:
        pass


def test_unexpected_internal_error_is_masked_not_leaked():
    """
    An unexpected exception from deep inside the service (not one of the
    known InvalidRequestError/ValueError branches) must surface as a
    generic, non-leaking ToolError -- never the raw exception message or
    a traceback.
    """
    expiry, quotes = _valid_chain()
    _install_fake_service({expiry: quotes})

    class ExplodingMarketData(FakeMarketDataService):
        def get_snapshot(self, *args, **kwargs):
            raise RuntimeError("SECRET INTERNAL DETAIL: database password is hunter2")

    mcp_server._service = StrategyHunterService(
        config=AppConfig(), market_data=ExplodingMarketData({expiry: quotes}), llm_client=FakeLLMClient()
    )

    try:
        mcp_server.find_option_strategies(
            asset="BTC", direction="neutral", max_loss=500.0, expiry_date=_expiry_date_str(expiry),
        )
        assert False, "expected ToolError for an unexpected internal exception"
    except ToolError as e:
        assert "hunter2" not in str(e), "internal exception detail leaked to the MCP client!"
        assert "password" not in str(e)


# -- payment-gate seam --------------------------------------------------------


def test_payment_gate_noop_by_default():
    """Confirms the payment-gate wrapper is genuinely wired (not just a
    comment) but is a true no-op while disabled -- a valid request must
    succeed exactly as if the wrapper weren't there at all."""
    expiry, quotes = _valid_chain()
    _install_fake_service({expiry: quotes})
    assert mcp_server._payment_gate.enabled is False

    result = mcp_server.find_option_strategies(
        asset="BTC", direction="neutral", max_loss=500.0, expiry_date=_expiry_date_str(expiry),
    )
    assert "strategies" in result or result.get("error") == "no_valid_strategies"


def test_payment_gate_blocks_when_enabled():
    """
    Confirms the seam actually enforces something the moment it's turned
    on, using the exact same PaymentGate class the FastAPI route uses
    (frisks/b402/middleware.py) -- not a second, MCP-specific mechanism.
    PaymentGate.authorize is itself a stub that always rejects while
    enabled (B402 wire format isn't implemented yet -- see
    frisks/b402/middleware.py), so this also confirms _payment_gated
    correctly maps PaymentRequiredError to ToolError.
    """
    expiry, quotes = _valid_chain()
    _install_fake_service({expiry: quotes})

    original_gate = mcp_server._payment_gate
    mcp_server._payment_gate = PaymentGate(enabled=True)
    try:
        try:
            mcp_server.find_option_strategies(
                asset="BTC", direction="neutral", max_loss=500.0, expiry_date=_expiry_date_str(expiry),
            )
            assert False, "expected ToolError when the payment gate is enabled"
        except ToolError as e:
            assert "payment" in str(e).lower()
    finally:
        mcp_server._payment_gate = original_gate
