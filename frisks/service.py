"""
StrategyHunterService: the orchestrator that wires the data layer, the
engine, and the LLM layer together into the locked request/response
contract. This is what both the FastAPI route (frisks/api/routes.py) and
the MCP tool (frisks/mcp/server.py) call -- one shared service, two
adapters, so LLM-orchestration logic added here runs for either caller
automatically.

Per the spec's "LLM vs. engine split": the LLM never computes a price, a
Greek, a probability, or any number. Every number in every response comes
from the deterministic engine. The LLM's job here is threefold:
  1. (pre-existing) turn free text into structured constraints when the
     caller sends text instead of the structured schema.
  2. (pre-existing) narrate already-computed results in natural language,
     via the separate /explain endpoint.
  3. (new) reason ABOUT already-computed engine output, on every request,
     regardless of how it arrived -- see "LLM-as-orchestrator" below.

-------------------------------------------------------------------------
LLM-AS-ORCHESTRATOR (build note): previously the LLM only touched the
natural-language-only endpoints, which structured callers (MCP, direct
API) never use -- making its real value near-zero for the primary
calling pattern. This build moves two LLM-driven behaviors inside
handle_structured_request itself:

Feature 1 -- feasibility-aware re-querying. After the primary engine run,
check whether the result looks constraint-starved (see
_check_feasibility_signals). If so, run a SECOND, REAL engine query with
a relaxed max_loss (frisks.config.OrchestrationConfig.budget_relax_multiplier)
reusing the already-fetched market data (no extra Binance calls -- only
max_loss differs between the two runs, and market data doesn't depend on
it), then have the LLM synthesize a short, real-numbers-only trade-off
explanation into the response's optional `budget_note` field. Most
requests won't trigger this -- that's correct, not a missed feature.

Feature 2 -- self-critique. Before finalizing the response, the LLM
reviews the actual returned strategies' actual computed fields (delta,
max_loss, probability_of_profit, objective_score, meta) and may append a
short, honest caveat to a strategy's existing `rationale` field where one
is genuinely warranted (marginal directional lean, low-probability/
high-score mismatch, thin candidate pool) -- never inventing a number,
never touching anything but the rationale text.

Both features are gated by OrchestrationConfig.enabled (default True,
env FRISKS_LLM_ORCHESTRATION_ENABLED) and fail SOFT: if the LLM is
unavailable or errors, the request still completes with the deterministic
engine result and default rationale -- these are enrichments, not
dependencies the core response requires.

FRISKS_DEBUG_LLM=true (also gated on the app's log level being INFO or
DEBUG) logs, for every request: which feasibility signal(s) fired and
the real numbers behind them, the full prompt text sent for both new LLM
calls, and the full raw LLM response before any post-processing -- this
is what actually lets a human judge output quality locally, before any
deployment work continues, per the build prompt.
-------------------------------------------------------------------------
"""
from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, replace
from datetime import datetime, timezone

from frisks.config import AppConfig
from frisks.data.market_data import MarketDataService, MarketSnapshot
from frisks.engine.generator import generate_candidates
from frisks.engine.models import Constraints as EngineConstraints
from frisks.engine.models import Direction, Objective, StrategyRequest
from frisks.engine.payoff import build_payoff_model
from frisks.engine.scoring import ScoredCandidate, rank_candidates
from frisks.llm.client import LLMClient

logger = logging.getLogger(__name__)


class InvalidRequestError(ValueError):
    pass


@dataclass
class NoValidStrategiesResult:
    request_id: str
    asset: str
    message: str


@dataclass
class StrategyHunterResult:
    request_id: str
    asset: str
    generated_at: str
    objective_used: str
    strategies: list[dict]
    meta: dict
    # Additive (LLM-orchestration build): populated only when the
    # feasibility-aware re-query triggers. See module docstring.
    budget_note: str | None = None


@dataclass
class _EngineRun:
    """Internal, non-schema result of one full deterministic engine pass
    (generation -> pricing -> scoring -> payload assembly). Used both for
    the primary request and, when feasibility signals fire, for a second
    relaxed-budget run -- kept internal so neither the API layer nor the
    MCP layer needs to know this exists."""

    strategies_payload: list[dict]
    meta: dict
    ranked: list[ScoredCandidate]
    candidates: list
    request: StrategyRequest


class StrategyHunterService:
    def __init__(
        self,
        config: AppConfig | None = None,
        market_data: MarketDataService | None = None,
        llm_client: LLMClient | None = None,
    ) -> None:
        self._config = config or AppConfig()
        self._market_data = market_data or MarketDataService(self._config)
        self._llm = llm_client or LLMClient(self._config.llm)
        # Feature 1, Signal C (soft/session-level, see
        # _check_feasibility_signals): in-memory only, per-process,
        # per-(asset, direction) rolling history of top-result PoP.
        # Deliberately simple per the build prompt ("don't over-engineer
        # this one") -- resets on restart, no persistence.
        self._session_pop_history: dict[tuple[str, str], list[float]] = {}

    def close(self) -> None:
        self._market_data.close()
        self._llm.close()

    # -- entry points -----------------------------------------------------

    def interpret_natural_language(self, text: str) -> dict:
        """LLM responsibility: free text -> structured request dict."""
        parsed = self._llm.interpret_request(text)
        return parsed

    def handle_structured_request(self, raw: dict) -> StrategyHunterResult | NoValidStrategiesResult:
        request = self._normalize_request(raw)
        request_id = f"frisks_{uuid.uuid4().hex[:12]}"

        snapshot = self._resolve_and_fetch(request)
        quotes_by_expiry = self._gather_quotes_by_expiry(request, snapshot)
        total_liquid = sum(len(qs) for qs in quotes_by_expiry.values())

        if total_liquid == 0:
            return NoValidStrategiesResult(
                request_id=request_id,
                asset=request.asset,
                message=(
                    f"No liquid candidates satisfied max_loss={request.max_loss} with "
                    f"direction={request.direction.value} for the given expiries."
                ),
            )

        underlying_price = self._estimate_underlying_price(snapshot)
        # Reused for a relaxed re-query if feasibility signals fire below
        # -- market data doesn't depend on max_loss, so there's no reason
        # to re-hit Binance for the second run.
        prefetched = (quotes_by_expiry, underlying_price, snapshot)

        primary = self._run_engine_once(request, prefetched=prefetched)
        if isinstance(primary, NoValidStrategiesResult):
            return NoValidStrategiesResult(request_id=request_id, asset=request.asset, message=primary.message)

        budget_note: str | None = None
        if self._config.orchestration.enabled:
            try:
                budget_note = self._maybe_run_feasibility_requery(request, primary, prefetched)
            except Exception:  # noqa: BLE001 -- enrichment, must never break the core response
                logger.exception("Feasibility re-query failed; continuing without budget_note")

            try:
                self._apply_self_critique(request, primary)
            except Exception:  # noqa: BLE001 -- same: enrichment only
                logger.exception("Self-critique failed; continuing with deterministic rationale only")

        return StrategyHunterResult(
            request_id=request_id,
            asset=request.asset,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            objective_used=request.objective.value,
            strategies=primary.strategies_payload,
            meta=primary.meta,
            budget_note=budget_note,
        )

    def explain(self, request_summary: str, results_json: str) -> str:
        """LLM responsibility: narrate already-computed results (separate, opt-in endpoint)."""
        return self._llm.explain_results(request_summary, results_json)

    # -- engine execution (single real pass; called for both the primary
    #    and, when triggered, the relaxed re-query) ------------------------

    def _run_engine_once(
        self,
        request: StrategyRequest,
        prefetched: tuple[dict, float, MarketSnapshot] | None = None,
    ) -> _EngineRun | NoValidStrategiesResult:
        if prefetched is not None:
            quotes_by_expiry, underlying_price, snapshot = prefetched
        else:
            snapshot = self._resolve_and_fetch(request)
            quotes_by_expiry = self._gather_quotes_by_expiry(request, snapshot)
            underlying_price = self._estimate_underlying_price(snapshot)

        total_liquid = sum(len(qs) for qs in quotes_by_expiry.values())
        if total_liquid == 0:
            return NoValidStrategiesResult(
                request_id="",
                asset=request.asset,
                message=(
                    f"No liquid candidates satisfied max_loss={request.max_loss} with "
                    f"direction={request.direction.value} for the given expiries."
                ),
            )

        engine_constraints = EngineConstraints(
            max_legs=request.constraints.max_legs, max_expiries=request.constraints.max_expiries
        )

        candidates, _gen_stats, exclusion_stats = generate_candidates(
            quotes_by_expiry=quotes_by_expiry,
            underlying_price=underlying_price,
            direction=request.direction,
            max_loss_budget=request.max_loss,
            constraints=engine_constraints,
            primary_expiry_ms=snapshot.expiry_ms,
        )

        if not candidates:
            return NoValidStrategiesResult(
                request_id="",
                asset=request.asset,
                message=(
                    f"No liquid candidates satisfied max_loss={request.max_loss} with "
                    f"direction={request.direction.value} for the given expiries."
                ),
            )

        risk_free_rate = self._representative_risk_free_rate(snapshot)
        payoff_models = [
            (
                c,
                build_payoff_model(
                    c,
                    underlying_price,
                    risk_free_rate,
                    self._config.engine.distribution_grid_points,
                    self._config.engine.distribution_grid_sigmas,
                ),
            )
            for c in candidates
        ]

        ranked = rank_candidates(
            scored_payoffs=payoff_models,
            objective=request.objective,
            liquidity_cfg=self._config.liquidity,
            target_cost=request.target_cost,
            max_loss_constraint=request.max_loss,
            top_n=self._config.engine.top_n_results,
        )

        strategies_payload = [
            self._strategy_to_dict(rank_idx + 1, scored) for rank_idx, scored in enumerate(ranked)
        ]

        meta = {
            "candidates_evaluated": len(candidates),
            "candidates_excluded_illiquid": snapshot.excluded_illiquid_count,
            "candidates_excluded_budget": exclusion_stats.budget_excluded,
            "constraints_applied": {
                "max_legs": engine_constraints.max_legs,
                "max_expiries": engine_constraints.max_expiries,
            },
        }

        return _EngineRun(
            strategies_payload=strategies_payload,
            meta=meta,
            ranked=ranked,
            candidates=candidates,
            request=request,
        )

    # -- Feature 1: feasibility-aware re-querying ----------------------------

    def _check_feasibility_signals(self, request: StrategyRequest, primary: _EngineRun) -> list[str]:
        """
        Returns a list of human-readable signal descriptions that fired
        (empty if none). Each check is independent and configurable (see
        OrchestrationConfig) rather than one hardcoded rule, per the
        build prompt.
        """
        cfg = self._config.orchestration
        signals: list[str] = []

        # Signal A: a high proportion of otherwise-valid candidates were
        # excluded specifically for exceeding max_loss (never illiquidity
        # -- that's a separate, already-reported count).
        total_considered = primary.meta["candidates_evaluated"] + primary.meta["candidates_excluded_budget"]
        if total_considered > 0:
            budget_fraction = primary.meta["candidates_excluded_budget"] / total_considered
            if budget_fraction >= cfg.budget_excluded_fraction_threshold:
                signals.append(
                    f"budget_excluded_fraction={budget_fraction:.2f} "
                    f"(excluded={primary.meta['candidates_excluded_budget']}, "
                    f"evaluated={primary.meta['candidates_evaluated']})"
                )

        # Signal B: the top-ranked result is a structurally weak fit for a
        # stated (non-neutral) direction -- e.g. a calendar spread returned
        # for a directional bullish/bearish request.
        if request.direction is not Direction.NEUTRAL and primary.strategies_payload:
            top_structure = primary.strategies_payload[0]["structure_name"].lower()
            if any(weak in top_structure for weak in cfg.weak_fit_structures):
                signals.append(f"weak_fit_top_structure={top_structure!r}")

        # Signal C (soft, lower priority -- session-relative PoP drop).
        if primary.strategies_payload:
            top_pop = primary.strategies_payload[0]["probability_of_profit"]
            key = (request.asset, request.direction.value)
            history = self._session_pop_history.setdefault(key, [])
            if len(history) >= cfg.session_pop_history_min_samples:
                avg = sum(history) / len(history)
                if avg > 0 and top_pop < avg * cfg.session_pop_drop_ratio:
                    signals.append(f"session_pop_drop current={top_pop:.3f} session_avg={avg:.3f}")
            history.append(top_pop)
            if len(history) > 50:  # bound memory for a long-lived process
                history.pop(0)

        if cfg.debug_llm:
            logger.info(
                "[llm-debug] feasibility signals for asset=%s direction=%s max_loss=%s: %s",
                request.asset, request.direction.value, request.max_loss, signals or "(none)",
            )

        return signals

    def _maybe_run_feasibility_requery(
        self,
        request: StrategyRequest,
        primary: _EngineRun,
        prefetched: tuple[dict, float, MarketSnapshot],
    ) -> str | None:
        cfg = self._config.orchestration
        signals = self._check_feasibility_signals(request, primary)
        if not signals:
            return None

        relaxed_request = replace(request, max_loss=request.max_loss * cfg.budget_relax_multiplier)
        # Reuses the SAME already-fetched market data as the primary run
        # -- a second, real engine call (candidate generation + scoring),
        # not a second Binance round-trip, since market data doesn't
        # depend on max_loss.
        relaxed = self._run_engine_once(relaxed_request, prefetched=prefetched)
        if isinstance(relaxed, NoValidStrategiesResult) or not relaxed.strategies_payload:
            # The relaxed run found nothing useful either -- nothing
            # meaningful to report, so stay quiet rather than force a note.
            return None

        request_summary = (
            f"asset={request.asset} direction={request.direction.value} objective={request.objective.value} "
            f"original_max_loss={request.max_loss} relaxed_max_loss={relaxed_request.max_loss} "
            f"(relax_multiplier={cfg.budget_relax_multiplier})"
        )
        results_json = json.dumps(
            {
                "original_budget_top_result": primary.strategies_payload[0],
                "relaxed_budget_top_result": relaxed.strategies_payload[0],
                "triggered_by_signals": signals,
            }
        )

        if cfg.debug_llm:
            logger.info(
                "[llm-debug] budget re-query TRIGGERED (signals=%s)\n"
                "[llm-debug] budget_note prompt request_summary: %s\n"
                "[llm-debug] budget_note prompt results_json: %s",
                signals, request_summary, results_json,
            )

        note = self._llm.synthesize_budget_note(request_summary, results_json)

        if cfg.debug_llm:
            logger.info("[llm-debug] budget_note raw LLM response: %s", note)

        return note

    # -- Feature 2: self-critique --------------------------------------------

    def _apply_self_critique(self, request: StrategyRequest, primary: _EngineRun) -> None:
        """
        Mutates primary.strategies_payload in place, appending a short
        caveat to a strategy's existing `rationale` string where the LLM
        judges one genuinely warranted. Never touches any numeric field.
        """
        if not primary.strategies_payload:
            return
        cfg = self._config.orchestration

        request_summary = (
            f"asset={request.asset} direction={request.direction.value} objective={request.objective.value} "
            f"max_loss={request.max_loss}"
        )
        payload_with_meta = {"strategies": primary.strategies_payload, "meta": primary.meta}
        strategies_json = json.dumps(payload_with_meta)

        if cfg.debug_llm:
            logger.info(
                "[llm-debug] self-critique prompt request_summary: %s\n"
                "[llm-debug] self-critique prompt strategies_json: %s",
                request_summary, strategies_json,
            )

        critique_by_rank = self._llm.self_critique(request_summary, strategies_json)

        if cfg.debug_llm:
            logger.info("[llm-debug] self-critique raw LLM response: %s", critique_by_rank)

        for strat in primary.strategies_payload:
            addendum = critique_by_rank.get(str(strat["rank"]))
            if addendum:
                strat["rationale"] = f"{strat['rationale']} {addendum}".strip()

    # -- internals ----------------------------------------------------------

    def _normalize_request(self, raw: dict) -> StrategyRequest:
        asset = raw.get("asset")
        if asset not in ("BTC", "ETH"):
            raise InvalidRequestError(f"Unsupported asset '{asset}' — MVP supports BTC and ETH only")

        horizon = raw.get("horizon") or {}
        expiry_date = horizon.get("expiry_date")
        days_out = horizon.get("days_out")
        if (expiry_date is None) == (days_out is None):
            raise InvalidRequestError("horizon must specify exactly one of expiry_date or days_out")

        try:
            direction = Direction(raw.get("direction"))
        except ValueError as exc:
            raise InvalidRequestError(f"Invalid direction: {raw.get('direction')}") from exc

        max_loss = raw.get("max_loss")
        if max_loss is None or float(max_loss) <= 0:
            raise InvalidRequestError("max_loss must be a positive number")

        objective_raw = raw.get("objective", "risk_adjusted_return")
        try:
            objective = Objective(objective_raw)
        except ValueError as exc:
            raise InvalidRequestError(f"Invalid objective: {objective_raw}") from exc

        target_cost = raw.get("target_cost")
        if objective is Objective.COST_FOR_TARGET and target_cost is None:
            raise InvalidRequestError("target_cost is required when objective is cost_for_target")
        if objective is not Objective.COST_FOR_TARGET and target_cost is not None:
            raise InvalidRequestError("target_cost must be omitted/null unless objective is cost_for_target")

        constraints_raw = raw.get("constraints") or {}
        constraints = EngineConstraints(
            max_legs=int(constraints_raw.get("max_legs", 4)),
            max_expiries=int(constraints_raw.get("max_expiries", 2)),
        )

        return StrategyRequest(
            asset=asset,
            expiry_date=expiry_date,
            days_out=days_out,
            direction=direction,
            max_loss=float(max_loss),
            objective=objective,
            target_cost=float(target_cost) if target_cost is not None else None,
            constraints=constraints,
        )

    def _resolve_and_fetch(self, request: StrategyRequest) -> MarketSnapshot:
        primary_expiry_ms = self._market_data.nearest_expiry(request.asset, request.expiry_date, request.days_out)
        return self._market_data.get_snapshot(request.asset, primary_expiry_ms)

    def _gather_quotes_by_expiry(self, request: StrategyRequest, primary_snapshot: MarketSnapshot) -> dict:
        """
        Builds the quote pool the generator searches over. Always includes
        the requested (primary) expiry. When max_expiries allows a second
        expiry, also pulls in the next couple of listed expiries after the
        primary one so calendar/diagonal templates and free-form search
        have a second leg-source to draw from.

        Decision (not specified in spec): "next couple" = up to 2 further
        listed expiries beyond the primary one, to bound the number of
        Binance calls and the resulting search space. Flagged for tuning
        if calendar-structure coverage needs to reach further-dated expiries.
        """
        quotes_by_expiry = {primary_snapshot.expiry_ms: primary_snapshot.liquid_quotes}

        if request.constraints.max_expiries >= 2:
            all_expiries = self._market_data.list_expiries(request.asset)
            later = [e for e in all_expiries if e > primary_snapshot.expiry_ms]
            for expiry_ms in later[:2]:
                snap = self._market_data.get_snapshot(request.asset, expiry_ms)
                if snap.liquid_quotes:
                    quotes_by_expiry[expiry_ms] = snap.liquid_quotes

        return quotes_by_expiry

    @staticmethod
    def _estimate_underlying_price(snapshot: MarketSnapshot) -> float:
        """
        Binance's ticker payload includes an `exercisePrice` field that is
        the index price outside the pre-settlement window — but we don't
        carry that field through TickerSnapshot (it's specific to expiry
        mechanics, not general market data). Instead we derive a robust
        underlying estimate from put-call parity on the most liquid
        at-the-money-ish pair in the snapshot: for a call/put sharing a
        strike K, C - P = S - K*exp(-rT) approximately (ignoring the small
        discount factor at typical short-dated crypto option maturities
        introduces negligible error relative to bid/ask spread noise).

        Falls back to the single most at-the-money contract available
        (by |delta|) if no strike has both a call and a put with real
        data at all -- see market_data.py for why that's now rare rather
        than the common case.
        """
        quotes = snapshot.quotes  # all quotes with data, not just liquid, for a stable mid-based estimate
        from frisks.data.models import OptionSide

        calls = {q.contract.strike: q for q in quotes if q.contract.side is OptionSide.CALL}
        puts = {q.contract.strike: q for q in quotes if q.contract.side is OptionSide.PUT}
        common_strikes = sorted(set(calls) & set(puts))

        if common_strikes:
            def parity_gap(k: float) -> float:
                c, p = calls[k], puts[k]
                return abs(c.mark.mark_price - p.mark.mark_price)

            best_strike = min(common_strikes, key=parity_gap)
            c, p = calls[best_strike], puts[best_strike]
            return best_strike + (c.mark.mark_price - p.mark.mark_price)

        if not quotes:
            raise InvalidRequestError(
                "Cannot estimate underlying price: snapshot has no contracts with market data at all"
            )

        most_atm = min(quotes, key=lambda q: abs(abs(q.mark.delta) - 0.5))
        logger.warning(
            "No matched call/put strike pair found for %s expiry=%s; falling back to nearest-ATM-by-delta "
            "estimate using %s (delta=%.4f, strike=%s). This is a lower-confidence estimate than parity.",
            snapshot.underlying, snapshot.expiry_ms, most_atm.symbol, most_atm.mark.delta, most_atm.contract.strike,
        )
        return most_atm.contract.strike

    @staticmethod
    def _representative_risk_free_rate(snapshot: MarketSnapshot) -> float:
        rates = [q.mark.risk_free_interest for q in snapshot.quotes if q.mark.risk_free_interest]
        return sum(rates) / len(rates) if rates else 0.0

    @staticmethod
    def _strategy_to_dict(rank: int, scored) -> dict:
        candidate = scored.candidate
        model = scored.payoff_model
        greeks = candidate.net_greeks()
        liquidity = candidate.liquidity_summary()

        legs_payload = [
            {
                "side": leg.side.value,
                "symbol": leg.symbol,
                "strike": leg.strike,
                "expiry": leg.quote.contract.expiry_date_str,
                "quantity": leg.quantity,
            }
            for leg in candidate.legs
        ]

        rationale = (
            f"{candidate.structure_name} — objective_score={scored.final_score:.4f} "
            f"({'liquidity-penalized' if scored.liquidity_penalty_applied else 'no liquidity penalty'}), "
            f"net delta={greeks['delta']:.3f}."
        )

        return {
            "rank": rank,
            "structure_name": candidate.structure_name,
            "legs": legs_payload,
            "entry_cost": round(candidate.entry_cost, 2),
            "max_profit": round(model.max_profit(), 2),
            "max_loss": round(model.max_loss(), 2),
            "breakeven": [round(b, 2) for b in model.breakevens()],
            "probability_of_profit": round(model.probability_of_profit(), 4),
            "greeks": {k: round(v, 5) for k, v in greeks.items()},
            "liquidity": {
                "min_volume_24h": round(liquidity["min_volume_24h"], 2),
                "max_spread_pct": round(liquidity["max_spread_pct"] * 100, 2),
                "open_interest": round(liquidity["open_interest"], 2),
            },
            "objective_score": round(scored.final_score, 4),
            "rationale": rationale,
        }
