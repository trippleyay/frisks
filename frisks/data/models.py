"""
Domain models for the data layer.

These normalize Binance's raw (string-typed, terse-keyed) JSON into typed,
well-named Python objects. Nothing above this layer should ever touch a raw
Binance response dict.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum


class OptionSide(str, Enum):
    CALL = "CALL"
    PUT = "PUT"


@dataclass(frozen=True)
class OptionContract:
    """Static contract definition, from /eapi/v1/exchangeInfo."""

    symbol: str  # e.g. "BTC-260925-90000-C"
    underlying: str  # e.g. "BTCUSDT"
    side: OptionSide
    strike: float
    expiry_ms: int  # epoch ms, from exchangeInfo's expiryDate
    unit: float  # quantity of underlying represented by one contract
    status: str  # "TRADING", etc.

    @property
    def expiry_date(self) -> datetime:
        return datetime.fromtimestamp(self.expiry_ms / 1000, tz=timezone.utc)

    @property
    def expiry_date_str(self) -> str:
        return self.expiry_date.strftime("%Y-%m-%d")

    @property
    def is_tradeable(self) -> bool:
        return self.status == "TRADING"


@dataclass(frozen=True)
class TickerSnapshot:
    """From /eapi/v1/ticker (24hr rolling stats)."""

    symbol: str
    bid_price: float
    ask_price: float
    last_price: float
    volume_24h: float  # contracts
    trade_count: int

    @property
    def has_bid(self) -> bool:
        return self.bid_price > 0

    @property
    def mid_price(self) -> float | None:
        if self.bid_price > 0 and self.ask_price > 0:
            return (self.bid_price + self.ask_price) / 2
        return None

    @property
    def spread_pct(self) -> float | None:
        """Spread as a fraction of mid price. None if no two-sided market."""
        mid = self.mid_price
        if mid is None or mid <= 0:
            return None
        return (self.ask_price - self.bid_price) / mid


@dataclass(frozen=True)
class MarkData:
    """From /eapi/v1/mark — Binance-computed mark price, IV, and Greeks."""

    symbol: str
    mark_price: float
    bid_iv: float
    ask_iv: float
    mark_iv: float
    delta: float
    theta: float
    gamma: float
    vega: float
    risk_free_interest: float


@dataclass(frozen=True)
class OpenInterest:
    symbol: str
    contracts: float
    usd: float


@dataclass(frozen=True)
class LiquidityMetrics:
    """Per-leg liquidity, always reported to callers regardless of pass/fail."""

    min_volume_24h: float
    max_spread_pct: float | None
    open_interest: float

    def is_liquid(self, min_volume: float, max_spread: float, require_bid: bool, has_bid: bool) -> bool:
        if require_bid and not has_bid:
            return False
        if self.min_volume_24h < min_volume:
            return False
        if self.max_spread_pct is None:
            # No two-sided market at all — treat as illiquid regardless of require_bid,
            # since a spread we can't compute is not a spread we can trust.
            return False
        if self.max_spread_pct > max_spread:
            return False
        return True


@dataclass(frozen=True)
class MarketQuote:
    """
    Fully joined per-contract market view: static contract info + ticker +
    mark/Greeks + open interest + derived liquidity flag. This is the unit
    the generator and scorer actually work with.
    """

    contract: OptionContract
    ticker: TickerSnapshot
    mark: MarkData
    open_interest: OpenInterest
    liquid: bool

    @property
    def symbol(self) -> str:
        return self.contract.symbol

    @property
    def liquidity_metrics(self) -> LiquidityMetrics:
        return LiquidityMetrics(
            min_volume_24h=self.ticker.volume_24h,
            max_spread_pct=self.ticker.spread_pct,
            open_interest=self.open_interest.contracts,
        )
