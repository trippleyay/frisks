"""
Test fixtures build synthetic MarketQuote objects directly, so the engine
test suite runs with zero network access and zero dependency on httpx /
fastapi / pydantic being installed — only the standard library and the
pure-Python engine/data.models modules are exercised here.
"""
from __future__ import annotations

import time

from frisks.data.models import MarkData, MarketQuote, OpenInterest, OptionContract, OptionSide, TickerSnapshot

DAY_MS = 24 * 3600 * 1000


def make_quote(
    underlying: str,
    side: OptionSide,
    strike: float,
    expiry_ms: int,
    mark_iv: float,
    bid: float,
    ask: float,
    volume: float = 10.0,
    oi: float = 100.0,
    risk_free: float = 0.0,
    mark_price: float | None = None,
) -> MarketQuote:
    symbol = f"{underlying[:3]}-{expiry_ms}-{int(strike)}-{'C' if side is OptionSide.CALL else 'P'}"
    contract = OptionContract(
        symbol=symbol,
        underlying=underlying,
        side=side,
        strike=strike,
        expiry_ms=expiry_ms,
        unit=1.0,
        status="TRADING",
    )
    ticker = TickerSnapshot(
        symbol=symbol, bid_price=bid, ask_price=ask, last_price=(bid + ask) / 2, volume_24h=volume, trade_count=50
    )
    mark = MarkData(
        symbol=symbol,
        mark_price=mark_price if mark_price is not None else (bid + ask) / 2,
        bid_iv=mark_iv * 0.97,
        ask_iv=mark_iv * 1.03,
        mark_iv=mark_iv,
        delta=0.5 if side is OptionSide.CALL else -0.5,
        theta=-1.0,
        gamma=0.001,
        vega=10.0,
        risk_free_interest=risk_free,
    )
    open_interest = OpenInterest(symbol=symbol, contracts=oi, usd=oi * mark.mark_price)
    return MarketQuote(contract=contract, ticker=ticker, mark=mark, open_interest=open_interest, liquid=True)


def near_expiry_ms(days: int = 30) -> int:
    return int(time.time() * 1000) + days * DAY_MS


def build_chain(
    underlying: str,
    spot: float,
    strikes: list[float],
    expiry_ms: int,
    iv: float = 0.6,
    spread_frac: float = 0.02,
) -> list[MarketQuote]:
    """Builds a symmetric call+put chain around `spot` at a flat IV, priced
    consistently via a light Black-Scholes so tests exercise realistic data."""
    from frisks.engine.pricing import bs_price

    years = max(expiry_ms - int(time.time() * 1000), 0) / 1000 / (365 * 24 * 3600)
    quotes = []
    for k in strikes:
        for side in (OptionSide.CALL, OptionSide.PUT):
            theo = bs_price(spot, k, years, 0.0, iv, side)
            theo = max(theo, 0.01)
            half_spread = max(theo * spread_frac, 0.01)
            bid = max(theo - half_spread, 0.0)
            ask = theo + half_spread
            quotes.append(
                make_quote(underlying, side, k, expiry_ms, mark_iv=iv, bid=bid, ask=ask, mark_price=theo)
            )
    return quotes
