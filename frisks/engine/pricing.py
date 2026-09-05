"""
Pricing / distribution model, per the spec's "Pricing / distribution model"
section.

Same-expiry structures: build the risk-neutral terminal price distribution
from the market-implied smile (markIV per strike), via Breeden-Litzenberger
applied to a call-price curve reconstructed from interpolated markIV — not
by inverting raw option prices, since real strike data is gappy and Binance
already publishes markIV directly.

Multi-expiry (calendar/diagonal) structures: build the T1 distribution from
the T1 chain; for each T1 price outcome, the near leg is priced at
intrinsic value and the far leg is re-marked via Black-Scholes using the
T2 chain's implied vol at that strike (sticky-strike simplification) and
the remaining time T2-T1.

Binance options are European, cash-settled, with automatic exercise (see
spec's confirmed facts) — so payoff at expiry is exactly intrinsic value,
no early-exercise adjustment anywhere below.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from frisks.data.models import OptionSide

_SQRT_2PI = math.sqrt(2 * math.pi)


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / _SQRT_2PI


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(
    underlying_price: float,
    strike: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    volatility: float,
    side: OptionSide,
) -> float:
    """
    Standard Black-Scholes-Merton price (no dividend yield — Binance
    options are on a spot-settled crypto underlying with funding captured
    via the risk-free rate Binance itself publishes per contract).

    Degenerate cases (T<=0 or vol<=0) fall back to intrinsic value so this
    is safe to call at/after expiry or with a flat/zero smile.
    """
    if time_to_expiry_years <= 0 or volatility <= 0 or underlying_price <= 0 or strike <= 0:
        intrinsic = (
            max(underlying_price - strike, 0.0)
            if side is OptionSide.CALL
            else max(strike - underlying_price, 0.0)
        )
        return intrinsic

    sqrt_t = math.sqrt(time_to_expiry_years)
    d1 = (
        math.log(underlying_price / strike)
        + (risk_free_rate + 0.5 * volatility * volatility) * time_to_expiry_years
    ) / (volatility * sqrt_t)
    d2 = d1 - volatility * sqrt_t

    if side is OptionSide.CALL:
        return underlying_price * _norm_cdf(d1) - strike * math.exp(
            -risk_free_rate * time_to_expiry_years
        ) * _norm_cdf(d2)
    return strike * math.exp(-risk_free_rate * time_to_expiry_years) * _norm_cdf(-d2) - underlying_price * _norm_cdf(
        -d1
    )


@dataclass(frozen=True)
class IVSmile:
    """Piecewise-linear implied-vol smile in strike space, built from markIV."""

    strikes: tuple[float, ...]
    ivs: tuple[float, ...]

    def iv_at(self, strike: float) -> float:
        strikes, ivs = self.strikes, self.ivs
        if strike <= strikes[0]:
            return ivs[0]
        if strike >= strikes[-1]:
            return ivs[-1]
        # Linear interpolation between the two bracketing listed strikes.
        lo, hi = 0, len(strikes) - 1
        while hi - lo > 1:
            mid = (lo + hi) // 2
            if strikes[mid] <= strike:
                lo = mid
            else:
                hi = mid
        k0, k1 = strikes[lo], strikes[hi]
        v0, v1 = ivs[lo], ivs[hi]
        weight = (strike - k0) / (k1 - k0)
        return v0 + weight * (v1 - v0)


def build_iv_smile(strike_to_mark_iv: dict[float, float]) -> IVSmile:
    """
    strike_to_mark_iv: mapping of strike -> markIV, already averaged across
    call/put where both exist at the same strike (markIV should agree
    closely via put-call parity; averaging smooths quoting noise).
    """
    if not strike_to_mark_iv:
        raise ValueError("Cannot build an IV smile with no strikes")
    strikes = tuple(sorted(strike_to_mark_iv))
    ivs = tuple(strike_to_mark_iv[k] for k in strikes)
    return IVSmile(strikes=strikes, ivs=ivs)


@dataclass(frozen=True)
class RiskNeutralDistribution:
    """Discretized risk-neutral terminal price distribution."""

    prices: tuple[float, ...]
    probabilities: tuple[float, ...]  # sums to ~1.0

    def expected_value(self, payoff_fn) -> float:
        return sum(p * payoff_fn(price) for price, p in zip(self.prices, self.probabilities))

    def probability(self, predicate) -> float:
        return sum(p for price, p in zip(self.prices, self.probabilities) if predicate(price))


def build_risk_neutral_distribution(
    underlying_price: float,
    time_to_expiry_years: float,
    risk_free_rate: float,
    smile: IVSmile,
    grid_points: int = 400,
    grid_sigmas: float = 6.0,
) -> RiskNeutralDistribution:
    """
    Breeden-Litzenberger: risk-neutral density f(K) = e^{rT} * d^2C/dK^2,
    where C(K) is the Black-Scholes call price at strike K using the
    market-implied smile's vol at that strike. We build C(.) on a dense
    strike grid spanning several standard deviations of the at-the-money
    lognormal spread (a reasonable envelope regardless of smile shape),
    then finite-difference twice and normalize the resulting density to
    sum to 1 over the grid.

    Using markIV directly per strike (rather than raw traded prices) is
    the more robust construction given real, sometimes gappy, strike data
    — per the spec.
    """
    atm_vol = smile.iv_at(underlying_price)
    sigma_span = atm_vol * math.sqrt(max(time_to_expiry_years, 1e-6)) * grid_sigmas
    log_center = math.log(underlying_price)
    log_lo = log_center - sigma_span
    log_hi = log_center + sigma_span

    n = max(grid_points, 50)
    step = (log_hi - log_lo) / (n - 1)
    strikes = [math.exp(log_lo + i * step) for i in range(n)]

    call_prices = [
        bs_price(underlying_price, k, time_to_expiry_years, risk_free_rate, smile.iv_at(k), OptionSide.CALL)
        for k in strikes
    ]

    discount = math.exp(risk_free_rate * time_to_expiry_years)
    densities = [0.0] * n
    # Central second difference in strike space (non-uniform spacing since
    # the grid is uniform in log-strike, not strike — compute actual dK's).
    for i in range(1, n - 1):
        k_minus, k_mid, k_plus = strikes[i - 1], strikes[i], strikes[i + 1]
        c_minus, c_mid, c_plus = call_prices[i - 1], call_prices[i], call_prices[i + 1]
        h1 = k_mid - k_minus
        h2 = k_plus - k_mid
        # Non-uniform-grid second derivative.
        d2c_dk2 = 2.0 * (h1 * c_plus - (h1 + h2) * c_mid + h2 * c_minus) / (h1 * h2 * (h1 + h2))
        densities[i] = max(discount * d2c_dk2, 0.0)

    # Convert density -> probability mass per grid point via trapezoidal
    # weights (average of adjacent strike gaps), then normalize.
    masses = [0.0] * n
    for i in range(n):
        left_gap = strikes[i] - strikes[i - 1] if i > 0 else strikes[1] - strikes[0]
        right_gap = strikes[i + 1] - strikes[i] if i < n - 1 else strikes[-1] - strikes[-2]
        masses[i] = densities[i] * (left_gap + right_gap) / 2.0

    total = sum(masses)
    if total <= 0:
        # Degenerate smile (e.g. near-zero vol) — fall back to a point mass
        # at the forward price so downstream EV/PoP math still behaves.
        forward = underlying_price * math.exp(risk_free_rate * time_to_expiry_years)
        return RiskNeutralDistribution(prices=(forward,), probabilities=(1.0,))

    probabilities = tuple(m / total for m in masses)
    return RiskNeutralDistribution(prices=tuple(strikes), probabilities=probabilities)


def reprice_far_leg(
    underlying_price_at_t1: float,
    strike: float,
    remaining_time_years: float,
    risk_free_rate: float,
    far_leg_iv: float,
    side: OptionSide,
) -> float:
    """
    Re-mark a not-yet-expired leg at an intermediate scenario price,
    per the spec's multi-expiry approach step 2: Black-Scholes is
    appropriate here because we're marking a live position, not settling
    a cash-settled European contract.
    """
    return bs_price(underlying_price_at_t1, strike, remaining_time_years, risk_free_rate, far_leg_iv, side)
