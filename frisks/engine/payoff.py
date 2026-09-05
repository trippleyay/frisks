"""
Payoff-at-scenario evaluation for a Candidate.

Same-expiry structures: payoff at expiry = sum of each leg's intrinsic
value (Binance options are European/cash-settled/auto-exercised — no
early-exercise complexity), net of the entry cost.

Multi-expiry (calendar/diagonal) structures: cannot use a single terminal
distribution since legs settle at different times. We build the T1
distribution, then for every T1 scenario price: near-leg legs (expiring at
T1) contribute intrinsic value; far-leg legs (expiring at T2) are
re-marked via Black-Scholes using that contract's own quoted implied vol
(sticky-strike) with remaining time T2-T1. This mirrors the spec's worked
algorithm exactly.

Bug fix (see git history / delivery notes): far legs must be repriced
using each *leg's own* quoted markIV (`leg.quote.mark.mark_iv`), not a
strike-only IV value shared between a call and a put at the same strike.
Black-Scholes put-call parity (C - P = S - K*exp(-r*tau)) is an identity
that holds for *any* sigma as long as the *same* sigma prices both sides
— so feeding a call leg and a put leg at the same strike the same
interpolated IV forces their combined value to be an exactly
deterministic, price-independent linear function of S, regardless of
real market skew. For a candidate combining a same-strike synthetic on
the near leg (whose expiry-intrinsic payoff is itself exactly S-K, the
T=0 parity identity) with a same-strike synthetic on the far leg priced
this way, the whole structure collapsed to a constant payoff -> spurious
probability_of_profit=1.0 and max_loss=0.0. Using each leg's own
already-published, side-specific markIV preserves genuine call/put skew
and removes the artificial cancellation.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from frisks.engine.models import Candidate
from frisks.engine.pricing import (
    IVSmile,
    RiskNeutralDistribution,
    build_iv_smile,
    build_risk_neutral_distribution,
    reprice_far_leg,
)

SECONDS_PER_YEAR = 365.0 * 24 * 3600


def _years_between_ms(t0_ms: int, t1_ms: int) -> float:
    return max(t1_ms - t0_ms, 0) / 1000.0 / SECONDS_PER_YEAR


def _smile_for_expiry(candidate_quotes, expiry_ms: int) -> IVSmile:
    strike_ivs: dict[float, list[float]] = {}
    for leg in candidate_quotes:
        if leg.expiry_ms != expiry_ms:
            continue
        strike_ivs.setdefault(leg.strike, []).append(leg.quote.mark.mark_iv)
    averaged = {k: sum(v) / len(v) for k, v in strike_ivs.items()}
    return build_iv_smile(averaged)


@dataclass
class PayoffModel:
    """
    Precomputed pieces needed to evaluate a candidate's value/payoff at any
    terminal (or intermediate, for multi-expiry) underlying price, plus the
    risk-neutral distribution to integrate over.
    """

    candidate: Candidate
    is_multi_expiry: bool
    distribution: RiskNeutralDistribution
    near_expiry_ms: int
    far_expiry_ms: int | None
    risk_free_rate: float

    def value_at(self, underlying_price: float) -> float:
        """
        Net payoff/value of the structure at the given underlying price,
        expressed at the near expiry's horizon (T1 for multi-expiry, the
        single expiry for same-expiry structures). Not yet netted against
        entry cost — callers subtract candidate.entry_cost themselves.
        """
        total = 0.0
        for leg in self.candidate.legs:
            unit = leg.quote.contract.unit
            if leg.expiry_ms == self.near_expiry_ms:
                total += leg.signed_quantity * leg.intrinsic_value(underlying_price) * unit
            else:
                assert self.far_expiry_ms is not None
                remaining_t = _years_between_ms(self.near_expiry_ms, self.far_expiry_ms)
                # Use this leg's own quoted markIV, not a strike-only smile
                # shared between calls and puts — see module docstring.
                iv = leg.quote.mark.mark_iv
                far_value = reprice_far_leg(
                    underlying_price_at_t1=underlying_price,
                    strike=leg.strike,
                    remaining_time_years=remaining_t,
                    risk_free_rate=self.risk_free_rate,
                    far_leg_iv=iv,
                    side=leg.option_side,
                )
                total += leg.signed_quantity * far_value * unit
        return total

    def payoff_net_of_cost(self, underlying_price: float) -> float:
        return self.value_at(underlying_price) - self.candidate.entry_cost

    def expected_value(self) -> float:
        return self.distribution.expected_value(self.payoff_net_of_cost)

    def probability_of_profit(self) -> float:
        return self.distribution.probability(lambda p: self.payoff_net_of_cost(p) > 0)

    def max_loss(self) -> float:
        """
        Worst-case loss across the evaluated grid. Since the grid spans
        several sigma of plausible terminal prices (see pricing config),
        this is a very close numerical approximation of the true worst
        case for structures without unbounded tails; for defined-risk
        structures it will exactly match the analytic max loss at the
        grid's resolution.
        """
        worst = min(self.payoff_net_of_cost(p) for p in self.distribution.prices)
        return -worst if worst < 0 else 0.0

    def max_profit(self) -> float:
        best = max(self.payoff_net_of_cost(p) for p in self.distribution.prices)
        return best

    def breakevens(self) -> list[float]:
        """Prices (on the evaluated grid) where payoff crosses zero, via sign-change detection."""
        prices = self.distribution.prices
        payoffs = [self.payoff_net_of_cost(p) for p in prices]
        crossings: list[float] = []
        for i in range(len(prices) - 1):
            p0, p1 = payoffs[i], payoffs[i + 1]
            if p0 == 0:
                crossings.append(prices[i])
                continue
            if (p0 < 0) != (p1 < 0):
                # Linear interpolation for the zero crossing.
                x0, x1 = prices[i], prices[i + 1]
                frac = -p0 / (p1 - p0)
                crossings.append(x0 + frac * (x1 - x0))
        return crossings


def build_payoff_model(
    candidate: Candidate,
    underlying_price: float,
    risk_free_rate: float,
    grid_points: int,
    grid_sigmas: float,
) -> PayoffModel:
    expiries = sorted(candidate.expiries)
    near_expiry_ms = expiries[0]
    is_multi = len(expiries) > 1

    near_smile = _smile_for_expiry(candidate.legs, near_expiry_ms)
    time_to_near = _years_between_ms(_now_ms(), near_expiry_ms)
    distribution = build_risk_neutral_distribution(
        underlying_price=underlying_price,
        time_to_expiry_years=time_to_near,
        risk_free_rate=risk_free_rate,
        smile=near_smile,
        grid_points=grid_points,
        grid_sigmas=grid_sigmas,
    )

    far_expiry_ms = expiries[1] if is_multi else None

    return PayoffModel(
        candidate=candidate,
        is_multi_expiry=is_multi,
        distribution=distribution,
        near_expiry_ms=near_expiry_ms,
        far_expiry_ms=far_expiry_ms,
        risk_free_rate=risk_free_rate,
    )


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)
