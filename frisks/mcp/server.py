"""
FastMCP server exposing Frisks' strategy-search engine as an MCP tool.

Architecture: one shared service, two adapters. Both the FastAPI route
(frisks/api/routes.py, POST /v1/strategies) and the tool in this module
call the exact same StrategyHunterService.handle_structured_request
method -- this module contains zero engine logic of its own, only
argument mapping (MCP call -> StrategyRequestSchema) and result shaping
(service result -> the same locked response schemas the HTTP route uses).
Each adapter constructs its own StrategyHunterService instance (they run
as separate OS processes in practice -- stdio MCP servers are spawned as
a subprocess by the calling client, entirely separate from the FastAPI/
uvicorn process), but both call the identical shared method, so the two
paths cannot silently diverge in behavior.

--------------------------------------------------------------------------
FastMCP-specific decisions (flagged per the build prompt -- nothing here
is explicit in it, so checked against real FastMCP conventions/docs
rather than assumed):

1. PACKAGE CHOICE. Two different things are called "FastMCP" in the
   wild: the standalone `fastmcp` PyPI package (PrefectHQ/fastmcp,
   "FastMCP 2.x/3.x", docs at gofastmcp.com) and the
   `mcp.server.fastmcp.FastMCP` class bundled inside the official `mcp`
   python-sdk (an earlier version absorbed into the SDK). We use the
   standalone `fastmcp` package: it's the actively developed superset,
   is what "the standard framework for Python MCP servers" points to
   today, and both expose the same @mcp.tool()/mcp.run() surface used
   below, so this choice does not lock us into anything unusual.

2. PARAMETER SHAPE. The build prompt says tool parameters should map
   directly to StrategyRequestSchema's fields. FastMCP *can* take a
   single Pydantic-model parameter directly (`def tool(request: SomeModel)`)
   and will generate a single nested-object property in the tool's input
   schema for it (confirmed: FastMCP's own schema generation "wraps,
   does not unwrap" a Pydantic-typed parameter). We deliberately did NOT
   do this. Two reasons: (a) real-world reports of MCP clients (e.g.
   Cursor) failing to correctly send a single nested-object tool
   argument ("Invalid type for parameter 'agent'... expected undefined,
   got object"), which is exactly the failure mode a "tested against a
   real MCP client" requirement is meant to catch; (b) a flat parameter
   list is simply easier for a calling agent to construct correctly
   without first inspecting a nested schema. So `find_option_strategies`
   takes flat, individually-typed/described parameters (asset,
   direction, max_loss, expiry_date, days_out, objective, target_cost,
   max_legs, max_expiries) -- a 1:1 flattening of StrategyRequestSchema's
   fields (Horizon's two mutually-exclusive fields and Constraints' two
   fields become individual top-level parameters). Validation is NOT
   reimplemented for this flattened shape: the tool function immediately
   reassembles a real StrategyRequestSchema (importing Horizon and
   Constraints from frisks.api.schemas, unchanged) and lets pydantic run
   its existing validators (the horizon exactly-one-of check, the
   target_cost/objective consistency check, max_loss>0, the asset/
   direction/objective enums) -- there is exactly one validation path,
   shared with the HTTP route, just entered through flat arguments.

3. RETURN TYPE. FastMCP auto-generates a tool's `outputSchema` from the
   function's return type annotation when that annotation is a Pydantic
   model (or dict/TypedDict/dataclass with type hints); plain dicts
   without a fixed shape are still supported for structured output
   (returned under a generic object schema) but don't get a single fixed
   `outputSchema`. Frisks' /v1/strategies endpoint legitimately returns
   one of two different shapes (StrategyResponseSchema on success,
   NoStrategyResponseSchema when nothing valid is found) -- there is no
   single Pydantic model that covers both without a wrapper type the
   locked API contract doesn't define. Rather than invent a wrapper
   schema (which would diverge from the locked HTTP contract) or force
   one of the two models to double as the annotation (which would make
   FastMCP validate the *other* branch's output against the wrong
   schema), we annotate the return type as `dict[str, Any]` and return
   `model_dump(mode="json")` of whichever locked schema actually applies.
   MCP clients still get a fully valid, JSON-serializable structured
   result; they just don't get a single machine-checked outputSchema
   distinguishing the two branches ahead of time. This is a real
   trade-off, flagged rather than silently made.

4. ERROR SURFACING. FastMCP wraps any exception escaping a tool function
   and returns it to the client as an MCP tool error (isError=True)
   rather than a raw traceback -- but by default it also includes the
   *original* exception's message in that wrapped text
   (`mask_error_details` is opt-in, not opt-out, in FastMCP's own
   constructor). Relying on that default for a production service is
   fragile: it depends on a library default not changing under us, and
   it doesn't distinguish "your request was invalid" (safe to show in
   full) from "our server broke" (should never leak internals). So every
   known failure mode is caught explicitly here and re-raised as
   `fastmcp.exceptions.ToolError` with an intentionally-written message
   -- ToolError's entire purpose is "this message is meant to be seen by
   the client," independent of masking configuration. Anything
   unexpected is logged in full server-side and surfaced to the client
   only as a generic, non-leaking ToolError.

5. SYNC TOOL FUNCTION. StrategyHunterService's methods are synchronous
   (blocking) -- they call Binance/LLM providers via httpx.Client, not an
   async client. FastMCP supports both sync and async tool functions; a
   sync function is run in FastMCP's own thread pool automatically, so
   this is deliberately a plain `def`, not `async def`, rather than
   wrapping fundamentally synchronous work in an unnecessary async shell.

6. HEALTH ENDPOINT. The Python `fastmcp` package (unlike the unrelated
   TypeScript `fastmcp` package, which does ship one) has no built-in
   health/readiness route -- verified against gofastmcp.com's docs
   rather than assumed. `@mcp.custom_route(path, methods=[...])` is
   FastMCP's documented way to mount a plain HTTP handler alongside the
   MCP endpoint when running over an HTTP transport; used below to add
   `/health` for Render's health-check config and the self-ping loop.
   Only reachable when running with transport="streamable-http" (see
   __main__.py) -- there is no HTTP surface to check under stdio.
--------------------------------------------------------------------------

PAYMENT-GATE EXTENSION POINT (read this before touching B402 later):
See `_payment_gated` and `_payment_gate` below. `_find_option_strategies_impl`
is wrapped by `_payment_gated`, which currently no-ops (B402 partner-account
approval is blocked -- see build prompt) but is wired to the *same*
`PaymentGate` stub class already used by the FastAPI route
(frisks/b402/middleware.py) rather than inventing a second payment
mechanism for MCP specifically. Once B402 access is available: implement
`PaymentGate.authorize` for real (it already raises `PaymentRequiredError`
on rejection, already mapped to `ToolError` below), and set
`FRISKS_B402_ENABLED=true`. Nothing in `_find_option_strategies_impl` or
`find_option_strategies` needs to change.
"""
from __future__ import annotations

import functools
import logging
from typing import Any, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from pydantic import ValidationError

from frisks.api.schemas import Constraints as ConstraintsSchema
from frisks.api.schemas import Horizon, NoStrategyResponseSchema, StrategyRequestSchema, StrategyResponseSchema
from frisks.b402.middleware import PaymentGate, PaymentRequiredError
from frisks.config import load_config
from frisks.logging_setup import configure_logging
from frisks.service import InvalidRequestError, NoValidStrategiesResult, StrategyHunterResult, StrategyHunterService

logger = logging.getLogger(__name__)

# -- module-level singletons ------------------------------------------------
# Constructed once per process (this stdio server is its own process,
# separate from the FastAPI process -- see module docstring). Tests
# monkeypatch `_service` directly rather than requiring a factory, since
# Python resolves this module-level name at call time, not at import time.

_config = load_config()
configure_logging(_config.log_level)
_service = StrategyHunterService(_config)
_payment_gate = PaymentGate(enabled=_config.b402_enabled)

mcp = FastMCP(name="Frisks")


@mcp.custom_route("/health", methods=["GET"])
async def health_check(request):
    """
    Deployment build addition. The Python `fastmcp` package has no
    built-in health endpoint by default (verified -- unlike some other
    MCP server frameworks, e.g. the unrelated TypeScript `fastmcp`
    package, which does) -- `@mcp.custom_route` is FastMCP's documented
    mechanism for mounting a plain HTTP handler alongside the MCP
    endpoint itself when running over an HTTP transport (streamable-http
    here). Only meaningful when running with transport="streamable-http"
    (see __main__.py) -- stdio has no HTTP surface to check. Wired into
    Render's health-check config for this service (see render.yaml) so
    Render itself monitors and restarts on failure, same as the FastAPI
    service's existing /healthz.
    """
    from starlette.responses import JSONResponse

    return JSONResponse({"status": "ok"})


def _payment_gated(fn):
    """
    Payment-gate seam (see module docstring). No-op today
    (_payment_gate.enabled is False by default / until B402 is wired up
    for real), but genuinely wired, not just commented -- the moment
    FRISKS_B402_ENABLED=true and PaymentGate.authorize has a real
    implementation, this wrapper starts enforcing it with no change to
    the wrapped function.
    """

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if _payment_gate.enabled:
            try:
                _payment_gate.authorize({})
            except PaymentRequiredError as exc:
                raise ToolError(f"Payment required: {exc}") from exc
        return fn(*args, **kwargs)

    return wrapper


@_payment_gated
def _find_option_strategies_impl(request: StrategyRequestSchema) -> dict[str, Any]:
    """
    The actual logic -- calls StrategyHunterService.handle_structured_request,
    the exact same method the FastAPI route (POST /v1/strategies) calls.
    This is what gets wrapped by the payment gate; it must never be
    changed to accommodate payment logic itself.
    """
    try:
        result = _service.handle_structured_request(request.model_dump(mode="json"))
    except InvalidRequestError as exc:
        raise ToolError(f"Invalid request: {exc}") from exc
    except ValueError as exc:
        # e.g. unresolvable expiry, no matched strikes for underlying-price
        # estimation -- same failure class the HTTP route maps to 422.
        raise ToolError(f"Could not resolve request against live market data: {exc}") from exc
    except Exception:  # noqa: BLE001 -- deliberate: never leak internals to the client
        logger.exception("Unexpected error handling find_option_strategies request")
        raise ToolError("An unexpected internal error occurred while searching for strategies.")

    if isinstance(result, NoValidStrategiesResult):
        body = NoStrategyResponseSchema(request_id=result.request_id, asset=result.asset, message=result.message)
        return body.model_dump(mode="json")

    assert isinstance(result, StrategyHunterResult)
    body = StrategyResponseSchema(
        request_id=result.request_id,
        asset=result.asset,
        generated_at=result.generated_at,
        objective_used=result.objective_used,
        strategies=result.strategies,
        meta=result.meta,
        budget_note=result.budget_note,
    )
    return body.model_dump(mode="json")


@mcp.tool()
def find_option_strategies(
    asset: Literal["BTC", "ETH"],
    direction: Literal["bullish", "bearish", "neutral"],
    max_loss: float,
    expiry_date: str | None = None,
    days_out: int | None = None,
    objective: Literal["risk_adjusted_return", "probability_of_profit", "cost_for_target"] = (
        "risk_adjusted_return"
    ),
    target_cost: float | None = None,
    max_legs: int = 4,
    max_expiries: int = 2,
) -> dict[str, Any]:
    """
    Search the live Binance European Options market and return the
    strongest option strategies for a stated trading objective and
    constraints.

    Frisks is a specialist options-strategy search engine, not a general
    Binance account assistant: it works purely from public market data
    (no account, API key, balance, or position access is used or
    required), searches up to `max_legs` legs across up to `max_expiries`
    distinct expiries, and every returned strategy carries the real,
    tradeable Binance contract symbol for each leg -- no further lookup
    is needed before executing it.

    Args:
        asset: Underlying asset. Only "BTC" and "ETH" are supported in
            this version.
        direction: Market view -- "bullish", "bearish", or "neutral".
            Structures with no real net delta lean in the requested
            direction (e.g. a delta-neutral combination) are excluded
            whenever direction is not "neutral"; structures that are
            merely *imperfectly* aligned (collars, ratio spreads) are
            not penalized for that alone.
        max_loss: Maximum acceptable loss in USDT. Also serves as the
            entry-cost budget for net-debit structures. Must be positive.
        expiry_date: Exact contract expiry as "YYYY-MM-DD", e.g.
            "2026-09-25". Provide exactly one of expiry_date or days_out
            -- never both, never neither.
        days_out: Alternative to expiry_date: match the nearest listed
            expiry to this many days from now. Provide exactly one of
            expiry_date or days_out -- never both, never neither.
        objective: Ranking objective. One of:
            - "risk_adjusted_return" (default): rank by expected value
              (integrated across the market-implied risk-neutral price
              distribution) divided by max loss.
            - "probability_of_profit": rank by the literal probability
              that payoff at expiry is positive, tie-broken by higher
              expected value.
            - "cost_for_target": rank by closeness of entry cost to
              `target_cost`, subject to still satisfying max_loss.
        target_cost: The entry cost (USDT) to get as close to as
            possible. Required when objective is "cost_for_target";
            must be omitted (left null) for any other objective.
        max_legs: Maximum number of legs in a returned structure.
            Defaults to 4.
        max_expiries: Maximum number of distinct expiries a returned
            structure may span. Defaults to 2. The requested expiry
            (from expiry_date/days_out) is always included in every
            returned structure -- when max_expiries is 2, the additional
            expiry only ever supplies the *other* leg of a calendar or
            diagonal structure, it never replaces the requested one.

    Returns:
        A dict. On success: schema_version, request_id, asset,
        generated_at, objective_used, a `strategies` list (up to 3,
        ranked; each with legs, entry_cost, max_profit, max_loss,
        breakeven, probability_of_profit, greeks, liquidity,
        objective_score, and a rationale that may include a short,
        genuinely-warranted caveat -- e.g. a marginal directional lean or
        a low-probability/high-score mismatch -- reasoned over the real
        computed numbers, never inventing one), and `meta` (candidate
        counts, including how many were excluded specifically for
        exceeding max_loss vs. for illiquidity, and applied constraints).
        An optional `budget_note` field is present only when the stated
        max_loss looked like the binding constraint on the result: it
        contains a short explanation, grounded in a second real engine
        run at a relaxed budget, of what a larger budget would actually
        unlock. Most well-budgeted requests omit it entirely -- that's
        expected, not a missed feature. If no valid strategy satisfies
        the given constraints, returns `error: "no_valid_strategies"`
        with an explanatory `message` instead of a degraded or partial
        result.

    Raises:
        A tool error if the arguments are invalid (e.g. both expiry_date
        and days_out given, or neither; target_cost given without
        objective="cost_for_target"; max_loss not positive) or if the
        request cannot be resolved against live Binance market data.
    """
    try:
        horizon = Horizon(expiry_date=expiry_date, days_out=days_out)
        constraints = ConstraintsSchema(max_legs=max_legs, max_expiries=max_expiries)
        request = StrategyRequestSchema(
            asset=asset,
            horizon=horizon,
            direction=direction,
            max_loss=max_loss,
            objective=objective,
            target_cost=target_cost,
            constraints=constraints,
        )
    except ValidationError as exc:
        raise ToolError(f"Invalid arguments: {exc}") from exc

    return _find_option_strategies_impl(request)
