"""
Strategy candidate generation.

Runs two modes and merges their output, per spec:
  1. Parametrized templates (frisks.engine.templates) across the liquid
     strike/expiry set — fast, known-good coverage of named structures.
  2. Free-form n-leg branch-and-bound search — covers non-templated valid
     combinations the template library doesn't enumerate.

Pruning rules (spec numbering preserved in comments below):
  1. Budget prune          — running net premium vs. max_loss/budget.
  2. Max-loss ceiling prune — worst-case loss vs. risk constraint, deferred
                              while upside risk is unbounded (naked short
                              calls with no higher covering long call).
  3. Direction-consistency prune — soft, applied only to completed candidates.
  4. Liquidity — handled upstream (MarketDataService only ever hands us
                 liquid quotes), not re-checked here.
  5. Redundancy prune — no exact-inverse no-op extensions.
  6. Expiry-count prune — once 2 distinct expiries are in use, only offer
                           legs from those same two.

Engineering note (not in spec): true exhaustive enumeration of the leg
space is combinatorially enormous even after liquidity filtering (dozens
of strikes x 2 sides x up to 4 legs). We bound the search with a node
budget (`max_nodes`) and explore strikes in order of distance from the
current underlying price first, since near-the-money combinations
dominate the economically sensible search space and liquidity itself
already concentrates there (per the spec's liquidity findings). This is a
scaling necessity, not a deviation from the pruning architecture: every
rule above is still enforced exactly as specified; we simply visit the
most promising branches first within a bounded compute budget so the
service returns in reasonable time.

BUG FIXES (post-MVP):

1. `_direction_contradicts` previously only flagged a candidate when its
   net delta leaned materially *against* the requested direction (e.g.
   `ratio < -threshold` for bullish). A candidate with ~zero net delta
   never satisfies that condition, so directionless structures silently
   passed a directional filter. Fixed to require a minimum net-delta lean
   *in* the requested direction (`ratio < threshold` fails it for
   bullish), not merely the absence of strong opposition — a directional
   request implies the caller wants some real exposure in that direction.

2. Box-spread-shaped candidates (a synthetic long forward at one expiry —
   buy call/sell put, same strike — combined with an offsetting synthetic
   short forward at another expiry) are close to delta/gamma/vega-neutral
   by construction: a synthetic forward has exactly zero gamma regardless
   of price, so any correctly-implemented repricer will show near-zero
   variance for this shape. That's real, not a pricing bug — but our
   payoff model has no representation of execution/fill risk, margin
   risk, or funding-rate uncertainty across the two expiries, so
   presenting one as a top "risk-bounded" or "high probability of profit"
   result would be materially misleading. Rather than fabricate an
   uncalibrated slippage number, `is_box_spread_shaped` detects this
   specific shape (structurally: opposite-direction synthetic forwards
   across exactly two expiries; numerically confirmed: near-zero net
   delta AND near-zero net gamma relative to gross exposure) and such
   candidates are excluded from ranking entirely in `generate_candidates`.

3. The requested (primary) expiry was being treated as optional rather
   than required: both `generate_all_templates` and
   `branch_and_bound_search` could return a fully-formed candidate built
   entirely from the *additional* expiries fetched only to support
   calendar/diagonal structures, containing zero legs at the expiry the
   caller actually asked for. Fixed: `primary_expiry_ms` is now a required
   parameter threaded through generation, enforced structurally (calendar
   templates only ever pair the primary expiry with one other; free-form
   search prunes any branch that would exhaust its expiry budget without
   the primary expiry included) and with a hard filter as a final
   safety net in `generate_candidates`.
4. `_bounded_worst_case_loss` computes worst-case loss using pure
   intrinsic value for every leg, which is EXACT for same-expiry
   candidates (that genuinely is the settlement value) but WRONG for
   multi-expiry candidates: the far leg is still alive at the near
   expiry, so its true value there is its live option value (time value
   included), not its intrinsic value. For a calendar/diagonal built as
   a net credit (e.g. a "reverse calendar spread"), this made the cheap
   check severely UNDERESTIMATE the short far leg's real liability
   (often near-zero intrinsic value when deep OTM, while its actual
   option value -- what you'd really have to pay to close it -- can be
   substantial), letting candidates whose real worst-case loss was many
   times over max_loss slip through the final budget gate. Confirmed via
   FRISKS_DEBUG_LLM testing: every violating result was a net-credit,
   multi-expiry structure; every correctly-bounded result was net-debit
   (same-expiry or otherwise unaffected by this gap). Fixed:
   `generate_candidates`'s final validity gate now computes the REAL
   worst-case loss for any multi-expiry candidate via the actual payoff
   model (frisks.engine.payoff.build_payoff_model, the same repricing
   used for scoring), not the cheap intrinsic-only approximation.
   Same-expiry candidates keep using the cheap check, since it is exact
   there, not an approximation -- no reason to pay for a distribution
   build when intrinsic value already *is* the true settlement value.
"""
from __future__ import annotations

import itertools
import logging
from dataclasses import dataclass

from frisks.data.models import MarketQuote
from frisks.engine.models import Candidate, Constraints, Direction, Leg, OrderSide
from frisks.engine.payoff import build_payoff_model
from frisks.engine.templates import generate_all_templates

logger = logging.getLogger(__name__)

# Box-spread detection thresholds (see is_box_spread_shaped). Expressed as
# a fraction of the candidate's own gross delta/gamma exposure so they
# scale sensibly regardless of position size or underlying price. A
# genuine box spread's net Greeks are essentially numerical noise (well
# under 1%); 5% gives comfortable margin for real market data (mark
# Greeks that aren't perfectly parity-consistent) without letting
# structures with real net exposure slip through.
BOX_SPREAD_DELTA_EPSILON = 0.05
BOX_SPREAD_GAMMA_EPSILON = 0.05


def _net_call_slope(legs: list[Leg]) -> float:
    """Sum of signed-quantity*unit over CALL legs. Negative => unbounded upside risk."""
    from frisks.data.models import OptionSide

    return sum(
        leg.signed_quantity * leg.quote.contract.unit for leg in legs if leg.option_side is OptionSide.CALL
    )


def _bounded_worst_case_loss(legs: list[Leg], running_cost: float) -> float | None:
    """
    Exact worst-case loss for the position *as currently constructed*,
    valid only when upside risk is already bounded (net_call_slope >= 0),
    since puts can never produce unbounded loss (S is bounded below by 0).
    Kinks in a piecewise-linear payoff only occur at strikes, so the
    minimum occurs at S=0 or at one of the current strikes.
    """
    if _net_call_slope(legs) < 0:
        return None  # unbounded upside risk — not yet resolvable, per rule 2.

    candidate_prices = [0.0] + [leg.strike for leg in legs]
    worst_value = min(
        sum(leg.signed_quantity * leg.intrinsic_value(p) * leg.quote.contract.unit for leg in legs)
        for p in candidate_prices
    )
    return max(running_cost - worst_value, 0.0)


def _direction_contradicts(candidate: Candidate, direction: Direction, threshold: float = 0.15) -> bool:
    """
    Rule 3, applied only to completed candidates.

    A candidate must show at least `threshold` worth of net delta, as a
    fraction of its own gross delta exposure, *in* the requested
    direction to survive. This still protects the spec's original intent
    — collars, ratio spreads, and other directionally "impure" but still
    net-aligned structures aren't punished for being imperfectly pure, as
    long as they clear the (modest) minimum-lean bar — while excluding
    structures with no real directional stance at all (net delta ~0),
    which previously passed any directional filter unchallenged since a
    ~0 ratio is never "materially opposed."
    """
    if direction is Direction.NEUTRAL:
        return False
    net_delta = candidate.net_greeks()["delta"]
    gross = sum(
        abs(leg.signed_quantity * leg.quote.mark.delta * leg.quote.contract.unit) for leg in candidate.legs
    )
    if gross <= 0:
        return False
    ratio = net_delta / gross
    if direction is Direction.BULLISH:
        return ratio < threshold
    return ratio > -threshold  # BEARISH


def _detect_synthetic_forward_direction(legs_at_expiry: list[Leg]) -> int:
    """
    Returns +1 if `legs_at_expiry` contains a synthetic long forward (long
    call + short put, same strike), -1 if a synthetic short forward (short
    call + long put, same strike), 0 if no such pair is present.
    """
    from frisks.data.models import OptionSide

    by_strike: dict[float, dict] = {}
    for leg in legs_at_expiry:
        by_strike.setdefault(leg.strike, {})[leg.option_side] = leg.side

    for sides in by_strike.values():
        call_side = sides.get(OptionSide.CALL)
        put_side = sides.get(OptionSide.PUT)
        if call_side is None or put_side is None:
            continue
        if call_side is OrderSide.BUY and put_side is OrderSide.SELL:
            return 1
        if call_side is OrderSide.SELL and put_side is OrderSide.BUY:
            return -1
    return 0


def is_box_spread_shaped(
    candidate: Candidate,
    delta_epsilon: float = BOX_SPREAD_DELTA_EPSILON,
    gamma_epsilon: float = BOX_SPREAD_GAMMA_EPSILON,
) -> bool:
    """
    True if `candidate` combines an opposite-direction synthetic forward
    at each of exactly two expiries (structural check), AND its net delta
    and net gamma are both near-zero relative to the candidate's own
    gross exposure (numerical confirmation). See module docstring for
    why this is excluded rather than scored with a fabricated risk
    adjustment.
    """
    if len(candidate.expiries) != 2:
        return False

    expiries = sorted(candidate.expiries)
    legs_by_expiry = [
        [leg for leg in candidate.legs if leg.expiry_ms == e] for e in expiries
    ]
    directions = [_detect_synthetic_forward_direction(legs) for legs in legs_by_expiry]

    if 0 in directions:
        return False  # at least one expiry has no clean synthetic-forward pair
    if directions[0] == directions[1]:
        return False  # same-direction synthetics across expiries — real calendar exposure, not a box

    greeks = candidate.net_greeks()
    gross_delta = sum(
        abs(leg.signed_quantity * leg.quote.mark.delta * leg.quote.contract.unit) for leg in candidate.legs
    )
    gross_gamma = sum(
        abs(leg.signed_quantity * leg.quote.mark.gamma * leg.quote.contract.unit) for leg in candidate.legs
    )
    delta_ratio = abs(greeks["delta"]) / gross_delta if gross_delta > 0 else 0.0
    gamma_ratio = abs(greeks["gamma"]) / gross_gamma if gross_gamma > 0 else 0.0

    return delta_ratio < delta_epsilon and gamma_ratio < gamma_epsilon


def _leg_options(quotes: list[MarketQuote]) -> list[tuple[OrderSide, MarketQuote]]:
    return [(side, q) for q in quotes for side in (OrderSide.BUY, OrderSide.SELL)]


@dataclass
class GenerationResult:
    candidates: list[Candidate]
    nodes_explored: int
    nodes_pruned_budget: int
    nodes_pruned_unbounded_at_leaf: int


def branch_and_bound_search(
    quotes_by_expiry: dict[int, list[MarketQuote]],
    underlying_price: float,
    direction: Direction,
    max_loss_budget: float,
    constraints: Constraints,
    primary_expiry_ms: int,
    max_nodes: int = 15000,
) -> GenerationResult:
    """
    Free-form n-leg search, up to constraints.max_legs, across at most
    constraints.max_expiries distinct expiries.

    BUG FIX: the requested (primary) expiry is now a hard constraint --
    every completed candidate must include at least one leg at
    `primary_expiry_ms`. Previously only the *count* of distinct expiries
    was capped (rule 6); nothing required one of them to be the expiry
    the caller actually asked for, so free-form search could return a
    fully-formed 2-expiry candidate built entirely from later expiries
    fetched only to support calendar structures. Enforced two ways:
    (1) an efficiency prune -- once a branch is about to exhaust its
    expiry budget without ever having included the primary expiry, that
    extension is a guaranteed dead end and is skipped; (2) a hard gate at
    candidate-completion time, so nothing without a primary-expiry leg
    ever enters `results`.
    """

    # Order candidate legs by |strike - spot| ascending within each expiry,
    # then interleave expiries — near-the-money first, per the scaling note above.
    ordered_quotes: list[MarketQuote] = []
    for expiry_ms in sorted(quotes_by_expiry):
        ordered_quotes.extend(
            sorted(quotes_by_expiry[expiry_ms], key=lambda q: abs(q.contract.strike - underlying_price))
        )

    results: list[Candidate] = []
    stats = {"nodes": 0, "pruned_budget": 0, "pruned_unbounded_leaf": 0}
    seen_signatures: set[tuple] = set()

    def signature(legs: list[Leg]) -> tuple:
        return tuple(sorted((leg.symbol, leg.side.value, leg.quantity) for leg in legs))

    def recurse(legs: list[Leg], running_cost: float, used_expiries: set[int], start_idx: int) -> None:
        if stats["nodes"] >= max_nodes:
            return
        stats["nodes"] += 1

        if len(legs) >= 2 and primary_expiry_ms in used_expiries:
            sig = signature(legs)
            if sig not in seen_signatures:
                seen_signatures.add(sig)
                candidate = Candidate(legs=list(legs), structure_name="free-form")
                if _net_call_slope(legs) >= 0:
                    worst = _bounded_worst_case_loss(legs, running_cost)
                    if worst is not None and worst <= max_loss_budget:
                        if not _direction_contradicts(candidate, direction):
                            results.append(candidate)
                else:
                    stats["pruned_unbounded_leaf"] += 1

        if len(legs) >= constraints.max_legs:
            return

        for idx in range(start_idx, len(ordered_quotes)):
            quote = ordered_quotes[idx]
            expiry_ms = quote.contract.expiry_ms

            # Rule 6: expiry-count prune.
            prospective_expiries = used_expiries | {expiry_ms}
            if len(prospective_expiries) > constraints.max_expiries:
                continue

            # Requested-expiry prune: if this extension would exhaust the
            # expiry budget without the primary expiry ever having been
            # included, every completion down this branch is a guaranteed
            # dead end (rule 6 forbids introducing a further expiry) --
            # skip it rather than exploring it just to discard it later.
            if len(prospective_expiries) == constraints.max_expiries and primary_expiry_ms not in prospective_expiries:
                continue

            for side in (OrderSide.BUY, OrderSide.SELL):
                new_leg = Leg(side=side, quote=quote, quantity=1.0)
                candidate_so_far = Candidate(legs=legs, structure_name="free-form")

                # Rule 5: redundancy prune.
                if candidate_so_far.is_redundant_extension(new_leg):
                    continue

                new_cost = running_cost + new_leg.cost()

                # Rule 1: budget prune (only meaningful for net debit — a
                # credit leg lowers running_cost and can never itself
                # trigger this, which is correct: a growing credit isn't a
                # budget violation).
                if new_cost > max_loss_budget:
                    stats["pruned_budget"] += 1
                    continue

                recurse(legs + [new_leg], new_cost, prospective_expiries, idx + 1)

                if stats["nodes"] >= max_nodes:
                    return

    recurse([], 0.0, set(), 0)
    logger.info(
        "branch_and_bound_search: nodes=%s candidates=%s pruned_budget=%s pruned_unbounded_leaf=%s",
        stats["nodes"], len(results), stats["pruned_budget"], stats["pruned_unbounded_leaf"],
    )
    return GenerationResult(
        candidates=results,
        nodes_explored=stats["nodes"],
        nodes_pruned_budget=stats["pruned_budget"],
        nodes_pruned_unbounded_at_leaf=stats["pruned_unbounded_leaf"],
    )


@dataclass
class ExclusionStats:
    """
    Diagnostic counters for why otherwise-structurally-valid candidates
    were excluded from the final ranked pool, broken out by *reason* --
    added to support the feasibility-aware re-querying feature (a caller
    whose budget was the binding constraint should get a different signal
    than one whose direction was simply unsatisfiable by the liquid
    chain). Distinct from `candidates_excluded_illiquid` (computed
    upstream in the data layer, before candidates are even generated).
    """

    budget_excluded: int = 0
    direction_excluded: int = 0
    box_spread_excluded: int = 0


def generate_candidates(
    quotes_by_expiry: dict[int, list[MarketQuote]],
    underlying_price: float,
    direction: Direction,
    max_loss_budget: float,
    constraints: Constraints,
    primary_expiry_ms: int,
    risk_free_rate: float,
    grid_points: int,
    grid_sigmas: float,
    max_nodes: int = 15000,
) -> tuple[list[Candidate], GenerationResult, ExclusionStats]:
    """
    Merge template-generated and free-form-searched candidates. Templates
    are also subject to the budget, direction-consistency, box-spread-
    shape, and requested-expiry checks so a template that happens to bust
    the user's constraints (or omit the expiry they actually asked for)
    doesn't leak through untested.

    `risk_free_rate`/`grid_points`/`grid_sigmas` are needed here (not just
    downstream in scoring) because multi-expiry candidates require the
    real payoff model to compute a correct worst-case loss -- see bug fix
    #4 in the module docstring.
    """
    template_candidates = generate_all_templates(quotes_by_expiry, constraints.max_expiries, primary_expiry_ms)
    bnb_result = branch_and_bound_search(
        quotes_by_expiry, underlying_price, direction, max_loss_budget, constraints, primary_expiry_ms, max_nodes
    )

    valid: list[Candidate] = []
    seen: set[tuple] = set()
    stats = ExclusionStats()

    def sig(c: Candidate) -> tuple:
        return tuple(sorted((leg.symbol, leg.side.value, leg.quantity) for leg in c.legs))

    for candidate in itertools.chain(template_candidates, bnb_result.candidates):
        if primary_expiry_ms not in candidate.expiries:
            # Hard requirement: every returned candidate must include at
            # least one leg at the requested expiry. Defense in depth --
            # both generators above already enforce this structurally, but
            # this guarantees it regardless of source.
            continue
        if len(candidate.legs) > constraints.max_legs:
            continue
        if len(candidate.expiries) > constraints.max_expiries:
            continue
        if _net_call_slope(candidate.legs) < 0:
            continue  # unbounded upside risk, never a valid completed candidate

        if len(candidate.expiries) > 1:
            # Multi-expiry: the cheap intrinsic-only bound is WRONG here
            # (see bug fix #4) -- use the real payoff model, the same
            # repricing used for scoring, as the authoritative gate.
            model = build_payoff_model(candidate, underlying_price, risk_free_rate, grid_points, grid_sigmas)
            worst = model.max_loss()
        else:
            # Same-expiry: intrinsic value at settlement IS the true
            # value, so the cheap check is exact, not an approximation --
            # no reason to pay for a distribution build here.
            worst = _bounded_worst_case_loss(candidate.legs, candidate.entry_cost)
            if worst is None:
                stats.budget_excluded += 1
                continue

        if worst > max_loss_budget:
            stats.budget_excluded += 1
            continue
        if candidate.entry_cost > max_loss_budget:
            stats.budget_excluded += 1
            continue
        if _direction_contradicts(candidate, direction):
            stats.direction_excluded += 1
            continue
        if is_box_spread_shaped(candidate):
            stats.box_spread_excluded += 1
            continue
        s = sig(candidate)
        if s in seen:
            continue
        seen.add(s)
        valid.append(candidate)

    if stats.box_spread_excluded:
        logger.info("generate_candidates: excluded %s box-spread-shaped candidates", stats.box_spread_excluded)
    if stats.budget_excluded:
        logger.info("generate_candidates: excluded %s candidates for exceeding max_loss budget", stats.budget_excluded)

    return valid, bnb_result, stats
