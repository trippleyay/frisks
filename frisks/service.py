"""
StrategyHunterService: the orchestrator that wires the data layer, the
engine, and the LLM layer together into the locked request/response
contract. This is what the API layer (frisks/api) calls.

Per the spec's "LLM vs. engine split": the LLM here is used only to
(a) turn free text into structured constraints when the caller sends
text instead of the structured schema, and (b) narrate the engine's
already-computed results. Every number in the response comes from the
deterministic engine.
"""
from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

from frisks.config import AppConfig
from frisks.data.market_data import MarketDataService, MarketSnapshot
from frisks.engine.generator import generate_candidates
from frisks.engine.models import Constraints as EngineConstraints
from frisks.engine.models import Direction, Objective, StrategyRequest
from frisks.engine.payoff import build_payoff_model
from frisks.engine.scoring import rank_candidates
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

    def close(self) -> None:
        self._market_data.close()
        self._llm.close()

    # -- entry points -----------------------------------------------------

    def interpret_natural_language(self, text: str) -> dict:
        """LLM responsibility #1: free text -> structured request dict."""
        parsed = self._llm.interpret_request(text)
        return parsed

    def handle_structured_request(self, raw: dict) -> StrategyHunterResult | NoValidStrategiesResult:
        request = self._normalize_request(raw)
        request_id = f"frisks_{uuid.uuid4().hex[:12]}"

        snapshot = self._resolve_and_fetch(request)
        engine_constraints = EngineConstraints(
            max_legs=request.constraints.max_legs, max_expiries=request.constraints.max_expiries
        )

        quotes_by_expiry = self._gather_quotes_by_expiry(request, snapshot)
        total_liquid = sum(len(qs) for qs in quotes_by_expiry.values())
        total_excluded = snapshot.excluded_illiquid_count

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

        candidates, gen_stats = generate_candidates(
            quotes_by_expiry=quotes_by_expiry,
            underlying_price=underlying_price,
            direction=request.direction,
            max_loss_budget=request.max_loss,
            constraints=engine_constraints,
        )

        if not candidates:
            return NoValidStrategiesResult(
                request_id=request_id,
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
            "candidates_excluded_illiquid": total_excluded,
            "constraints_applied": {
                "max_legs": engine_constraints.max_legs,
                "max_expiries": engine_constraints.max_expiries,
            },
        }

        return StrategyHunterResult(
            request_id=request_id,
            asset=request.asset,
            generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            objective_used=request.objective.value,
            strategies=strategies_payload,
            meta=meta,
        )

    def explain(self, request_summary: str, results_json: str) -> str:
        """LLM responsibility #2: narrate already-computed results."""
        return self._llm.explain_results(request_summary, results_json)

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

        BUG FIX: this previously hard-failed whenever `snapshot.quotes`
        had no strike with both a call and a put present. That was almost
        always a symptom of a separate bug (market_data.py silently
        dropping every illiquid contract from `.quotes`, contradicting
        this method's own "not just liquid" comment) rather than a truly
        unusable snapshot — real option chains routinely have a call and
        put both listed at nearly every strike, just not always both
        clearing the liquidity bar simultaneously. Now that
        MarketDataService retains every contract with real ticker/mark
        data in `.quotes` (see market_data.py), an exact matched pair
        should almost always exist. For the residual, genuinely
        degenerate case — no strike has both sides with any data at all —
        we fall back to the single most at-the-money contract available
        (whichever quote's |delta| is closest to 0.5, since an
        at-the-money option's strike is itself a reasonable proxy for
        spot) rather than hard-failing the whole request.
        """
        quotes = snapshot.quotes  # use all quotes with data, not just liquid, for a stable mid-based estimate
        from frisks.data.models import OptionSide

        calls = {q.contract.strike: q for q in quotes if q.contract.side is OptionSide.CALL}
        puts = {q.contract.strike: q for q in quotes if q.contract.side is OptionSide.PUT}
        common_strikes = sorted(set(calls) & set(puts))

        if common_strikes:
            # Use the strike where |call_mid - put_mid| is smallest as the
            # best ATM proxy (parity is most accurate there).
            def parity_gap(k: float) -> float:
                c, p = calls[k], puts[k]
                c_mid = c.mark.mark_price
                p_mid = p.mark.mark_price
                return abs(c_mid - p_mid)

            best_strike = min(common_strikes, key=parity_gap)
            c, p = calls[best_strike], puts[best_strike]
            return best_strike + (c.mark.mark_price - p.mark.mark_price)

        # Fallback: no strike has both a call and a put with real data.
        # Pick whichever single quote (either side) is most at-the-money
        # by |delta|, and use its strike as a rough spot proxy. Lower
        # confidence than parity, but far better than a hard failure —
        # logged clearly so this path is visible/traceable in practice.
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
