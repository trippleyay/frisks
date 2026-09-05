"""
Scoring, per the spec's "Scoring, per objective" section.

- risk_adjusted_return (default): EV / max_loss, EV from integrating the
  payoff function across the risk-neutral distribution (not a naive
  max-profit/max-loss ratio).
- probability_of_profit: literal probability that payoff > 0 under the
  same distribution; ties broken by higher EV.
- cost_for_target: minimize |entry_cost - target_cost| subject to
  max_loss; ties broken by EV (documented decision below — spec leaves
  the exact weighting unresolved).

Liquidity is never blended into the primary score with an arbitrary
weight — it's a separate multiplicative penalty applied only to the
bottom decile of *already-liquid* candidates (liquidity's hard floor is
enforced upstream in the data layer, not here).
"""
from __future__ import annotations

from dataclasses import dataclass

from frisks.config import LiquidityConfig
from frisks.engine.models import Candidate, Objective
from frisks.engine.payoff import PayoffModel


@dataclass
class ScoredCandidate:
    candidate: Candidate
    payoff_model: PayoffModel
    raw_score: float
    liquidity_penalty_applied: bool
    final_score: float


def _liquidity_rank_key(candidate: Candidate) -> float:
    """
    Composite liquidity signal used only to identify the bottom decile
    among already-liquid candidates — higher is more liquid. Volume and
    open interest matter more than a slightly-wide-but-passing spread, so
    weight this way; it's a ranking heuristic, not a pass/fail gate.
    """
    summary = candidate.liquidity_summary()
    spread = summary["max_spread_pct"] or 0.0
    return summary["min_volume_24h"] + 0.1 * summary["open_interest"] - 50.0 * spread


def apply_liquidity_penalty(
    scored: list[tuple[Candidate, float]], liquidity_cfg: LiquidityConfig
) -> list[tuple[Candidate, float, float, bool]]:
    """
    Returns list of (candidate, raw_score, final_score, penalty_applied)
    with the bottom-decile-by-liquidity candidates discounted.
    """
    if not scored:
        return []

    ranked = sorted(scored, key=lambda pair: _liquidity_rank_key(pair[0]))
    cutoff_idx = max(0, int(len(ranked) * liquidity_cfg.penalty_percentile))
    penalized_symbols_sets = {
        tuple(sorted(leg.symbol for leg in cand.legs)) for cand, _ in ranked[:cutoff_idx]
    }

    out = []
    for candidate, raw_score in scored:
        key = tuple(sorted(leg.symbol for leg in candidate.legs))
        penalized = key in penalized_symbols_sets
        final = raw_score * liquidity_cfg.penalty_multiplier if penalized else raw_score
        out.append((candidate, raw_score, final, penalized))
    return out


def score_candidate(
    candidate: Candidate,
    payoff_model: PayoffModel,
    objective: Objective,
    max_loss_constraint: float,
    target_cost: float | None,
) -> float:
    """Raw, objective-specific score. Not normalized across objective types."""
    if objective is Objective.RISK_ADJUSTED_RETURN:
        max_loss = payoff_model.max_loss()
        if max_loss <= 0:
            # No-loss-scenario structure (e.g. a pure credit with bounded
            # gain and literally zero modeled downside on the grid) —
            # avoid division by zero; treat as maximally attractive.
            return float("inf") if payoff_model.expected_value() > 0 else 0.0
        return payoff_model.expected_value() / max_loss

    if objective is Objective.PROBABILITY_OF_PROFIT:
        # Tiebreak handled by the caller sorting on (PoP, EV) — the raw
        # score here is PoP itself; EV is attached separately for the
        # secondary sort key.
        return payoff_model.probability_of_profit()

    if objective is Objective.COST_FOR_TARGET:
        assert target_cost is not None
        return -abs(candidate.entry_cost - target_cost)  # closer to 0 (less negative) is better

    raise ValueError(f"Unknown objective: {objective}")


def rank_candidates(
    scored_payoffs: list[tuple[Candidate, PayoffModel]],
    objective: Objective,
    liquidity_cfg: LiquidityConfig,
    target_cost: float | None,
    max_loss_constraint: float,
    top_n: int,
) -> list[ScoredCandidate]:
    raw_scores = [
        (candidate, score_candidate(candidate, model, objective, max_loss_constraint, target_cost))
        for candidate, model in scored_payoffs
    ]
    penalized = apply_liquidity_penalty(raw_scores, liquidity_cfg)

    model_by_candidate = {id(c): m for c, m in scored_payoffs}

    def sort_key(item):
        candidate, raw_score, final_score, _penalized = item
        model = model_by_candidate[id(candidate)]
        if objective is Objective.PROBABILITY_OF_PROFIT:
            # Tiebreak among near-equal PoP by higher EV, per spec (default tiebreak version).
            return (final_score, model.expected_value())
        if objective is Objective.COST_FOR_TARGET:
            # Open item (spec): exact tie-break weighting unresolved.
            # Decision: tie-break by EV, then by PoP, both maximized.
            return (final_score, model.expected_value(), model.probability_of_profit())
        return (final_score,)

    penalized.sort(key=sort_key, reverse=True)

    out = []
    for candidate, raw_score, final_score, was_penalized in penalized[:top_n]:
        out.append(
            ScoredCandidate(
                candidate=candidate,
                payoff_model=model_by_candidate[id(candidate)],
                raw_score=raw_score,
                liquidity_penalty_applied=was_penalized,
                final_score=final_score,
            )
        )
    return out
