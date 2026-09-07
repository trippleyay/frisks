"""
Parametrized generation of named structures over the liquid strike set for
one or two expiries. This is the "fast, known-good coverage" half of
generation described in the spec; frisks.engine.generator runs this
alongside the free-form branch-and-bound search.

Every template only ever draws legs from quotes the caller passes in
(already liquidity-filtered upstream), and every template respects
max_legs / max_expiries implicitly by construction (each template has a
fixed leg count and touches at most 2 expiries when calendar variants are
included).
"""
from __future__ import annotations

import itertools

from frisks.data.models import MarketQuote, OptionSide
from frisks.engine.models import Candidate, OrderSide, Leg


def _calls(quotes: list[MarketQuote]) -> list[MarketQuote]:
    return sorted((q for q in quotes if q.contract.side is OptionSide.CALL), key=lambda q: q.contract.strike)


def _puts(quotes: list[MarketQuote]) -> list[MarketQuote]:
    return sorted((q for q in quotes if q.contract.side is OptionSide.PUT), key=lambda q: q.contract.strike)


def _leg(quote: MarketQuote, side: OrderSide, qty: float = 1.0) -> Leg:
    return Leg(side=side, quote=quote, quantity=qty)


def vertical_spreads(quotes: list[MarketQuote], max_candidates: int = 60) -> list[Candidate]:
    """Bull/bear call/put spreads across all liquid strike pairs at one expiry."""
    out: list[Candidate] = []
    for name, options, buy_side, sell_relation in (
        ("bull call spread", _calls(quotes), OrderSide.BUY, "lower_first"),
        ("bear call spread", _calls(quotes), OrderSide.SELL, "lower_first"),
        ("bull put spread", _puts(quotes), OrderSide.BUY, "higher_first"),
        ("bear put spread", _puts(quotes), OrderSide.SELL, "higher_first"),
    ):
        for lo, hi in itertools.combinations(options, 2):
            if buy_side is OrderSide.BUY and sell_relation == "lower_first":
                legs = [_leg(lo, OrderSide.BUY), _leg(hi, OrderSide.SELL)]
            elif buy_side is OrderSide.SELL and sell_relation == "lower_first":
                legs = [_leg(lo, OrderSide.SELL), _leg(hi, OrderSide.BUY)]
            elif buy_side is OrderSide.BUY and sell_relation == "higher_first":
                legs = [_leg(hi, OrderSide.BUY), _leg(lo, OrderSide.SELL)]
            else:
                legs = [_leg(hi, OrderSide.SELL), _leg(lo, OrderSide.BUY)]
            out.append(Candidate(legs=legs, structure_name=name))
            if len(out) >= max_candidates:
                return out
    return out


def straddles_and_strangles(quotes: list[MarketQuote], max_candidates: int = 30) -> list[Candidate]:
    out: list[Candidate] = []
    calls, puts = _calls(quotes), _puts(quotes)
    call_by_strike = {c.contract.strike: c for c in calls}
    put_by_strike = {p.contract.strike: p for p in puts}

    # Straddles: same strike, call + put.
    for strike, call in call_by_strike.items():
        put = put_by_strike.get(strike)
        if put is None:
            continue
        for side, name in ((OrderSide.BUY, "long straddle"), (OrderSide.SELL, "short straddle")):
            out.append(Candidate(legs=[_leg(call, side), _leg(put, side)], structure_name=name))
        if len(out) >= max_candidates:
            return out

    # Strangles: put strike below call strike, both same direction.
    for call in calls:
        for put in puts:
            if put.contract.strike >= call.contract.strike:
                continue
            for side, name in ((OrderSide.BUY, "long strangle"), (OrderSide.SELL, "short strangle")):
                out.append(Candidate(legs=[_leg(call, side), _leg(put, side)], structure_name=name))
            if len(out) >= max_candidates:
                return out
    return out


def butterflies(quotes: list[MarketQuote], max_candidates: int = 40) -> list[Candidate]:
    """Symmetric long/short butterflies on calls and on puts: buy 1 low, sell 2 mid, buy 1 high."""
    out: list[Candidate] = []
    for options, label in ((_calls(quotes), "call"), (_puts(quotes), "put")):
        strikes = sorted({o.contract.strike for o in options})
        by_strike = {o.contract.strike: o for o in options}
        for i in range(len(strikes)):
            for j in range(i + 1, len(strikes)):
                for k in range(j + 1, len(strikes)):
                    lo, mid, hi = strikes[i], strikes[j], strikes[k]
                    # Only a "true" butterfly if wings are equidistant from the body.
                    if not math_isclose(mid - lo, hi - mid):
                        continue
                    lo_q, mid_q, hi_q = by_strike[lo], by_strike[mid], by_strike[hi]
                    long_legs = [_leg(lo_q, OrderSide.BUY), _leg(mid_q, OrderSide.SELL, 2.0), _leg(hi_q, OrderSide.BUY)]
                    short_legs = [_leg(lo_q, OrderSide.SELL), _leg(mid_q, OrderSide.BUY, 2.0), _leg(hi_q, OrderSide.SELL)]
                    out.append(Candidate(legs=long_legs, structure_name=f"long {label} butterfly"))
                    out.append(Candidate(legs=short_legs, structure_name=f"short {label} butterfly"))
                    if len(out) >= max_candidates:
                        return out
    return out


def math_isclose(a: float, b: float, rel_tol: float = 1e-6) -> bool:
    import math

    return math.isclose(a, b, rel_tol=rel_tol, abs_tol=1e-9)


def iron_condors(quotes: list[MarketQuote], max_candidates: int = 40) -> list[Candidate]:
    """Sell a put spread + sell a call spread, all four strikes distinct, put strikes < call strikes."""
    out: list[Candidate] = []
    calls, puts = _calls(quotes), _puts(quotes)
    for put_lo, put_hi in itertools.combinations(puts, 2):
        for call_lo, call_hi in itertools.combinations(calls, 2):
            if put_hi.contract.strike >= call_lo.contract.strike:
                continue
            legs = [
                _leg(put_lo, OrderSide.BUY),
                _leg(put_hi, OrderSide.SELL),
                _leg(call_lo, OrderSide.SELL),
                _leg(call_hi, OrderSide.BUY),
            ]
            out.append(Candidate(legs=legs, structure_name="iron condor"))
            if len(out) >= max_candidates:
                return out
    return out


def collars(quotes: list[MarketQuote], max_candidates: int = 30) -> list[Candidate]:
    """
    Protective collar approximation without an actual underlying position:
    long put (protection) + short call (financing), different strikes,
    same expiry. Framed here as the options overlay only.
    """
    out: list[Candidate] = []
    calls, puts = _calls(quotes), _puts(quotes)
    for put in puts:
        for call in calls:
            if call.contract.strike <= put.contract.strike:
                continue
            legs = [_leg(put, OrderSide.BUY), _leg(call, OrderSide.SELL)]
            out.append(Candidate(legs=legs, structure_name="collar"))
            if len(out) >= max_candidates:
                return out
    return out


def ratio_spreads(quotes: list[MarketQuote], max_candidates: int = 30) -> list[Candidate]:
    """1x2 ratio spreads on calls and puts."""
    out: list[Candidate] = []
    for options, label in ((_calls(quotes), "call"), (_puts(quotes), "put")):
        for lo, hi in itertools.combinations(options, 2):
            legs_bull = [_leg(lo, OrderSide.BUY), _leg(hi, OrderSide.SELL, 2.0)]
            out.append(Candidate(legs=legs_bull, structure_name=f"{label} ratio spread (1x2)"))
            if len(out) >= max_candidates:
                return out
    return out


def calendar_spreads(
    near_quotes: list[MarketQuote], far_quotes: list[MarketQuote], max_candidates: int = 30
) -> list[Candidate]:
    """Same strike, sell near expiry / buy far expiry (or reverse), on calls and puts."""
    out: list[Candidate] = []
    for near_opts, far_opts, label in (
        (_calls(near_quotes), _calls(far_quotes), "call"),
        (_puts(near_quotes), _puts(far_quotes), "put"),
    ):
        far_by_strike = {f.contract.strike: f for f in far_opts}
        for near in near_opts:
            far = far_by_strike.get(near.contract.strike)
            if far is None:
                continue
            out.append(
                Candidate(
                    legs=[_leg(near, OrderSide.SELL), _leg(far, OrderSide.BUY)],
                    structure_name=f"{label} calendar spread",
                )
            )
            out.append(
                Candidate(
                    legs=[_leg(near, OrderSide.BUY), _leg(far, OrderSide.SELL)],
                    structure_name=f"{label} reverse calendar spread",
                )
            )
            if len(out) >= max_candidates:
                return out
    return out


def generate_all_templates(
    quotes_by_expiry: dict[int, list[MarketQuote]], max_expiries: int, primary_expiry_ms: int
) -> list[Candidate]:
    """
    Runs every single-expiry template against the PRIMARY (requested)
    expiry only, and calendar templates across (primary, other) expiry
    pairs if max_expiries allows -- never between two non-primary
    expiries.

    BUG FIX: this previously looped single-expiry templates over *every*
    expiry in `quotes_by_expiry`, including expiries fetched only to
    supply the other leg of a calendar structure, and built calendar
    candidates from every pair of expiries via
    `itertools.combinations(expiries, 2)` regardless of whether either
    expiry was the one the caller actually requested. That let, e.g., a
    caller requesting 2026-09-25 receive a "call calendar spread" built
    entirely from 2026-10-30 and 2026-11-27 contracts -- the requested
    expiry was being treated as optional rather than required. Fixed:
    the requested expiry is now a hard constraint. Single-expiry
    templates only ever run against the primary expiry's quotes; calendar
    templates only ever pair the primary expiry with one other fetched
    expiry (never two non-primary expiries together) -- additional
    expiries supply the *other* leg of a calendar/diagonal, they never
    replace the requested one.
    """
    out: list[Candidate] = []
    primary_quotes = quotes_by_expiry.get(primary_expiry_ms, [])
    out.extend(vertical_spreads(primary_quotes))
    out.extend(straddles_and_strangles(primary_quotes))
    out.extend(butterflies(primary_quotes))
    out.extend(iron_condors(primary_quotes))
    out.extend(collars(primary_quotes))
    out.extend(ratio_spreads(primary_quotes))

    if max_expiries >= 2:
        for other_ms, other_quotes in quotes_by_expiry.items():
            if other_ms == primary_expiry_ms:
                continue
            near_ms, far_ms = sorted((primary_expiry_ms, other_ms))
            out.extend(calendar_spreads(quotes_by_expiry[near_ms], quotes_by_expiry[far_ms]))

    return out
