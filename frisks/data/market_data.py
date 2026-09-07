"""
Orchestrates the data layer: fetches (with caching) the four Binance
endpoints, joins them per-symbol into MarketQuote objects, and applies the
hard liquidity exclusion described in the spec.

This is the only module the engine talks to for market data — it never
sees raw Binance responses or the individual endpoint clients.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from frisks.config import AppConfig
from frisks.data.binance_client import BinanceOptionsClient, ExchangeInfo
from frisks.data.cache import TTLCache
from frisks.data.models import LiquidityMetrics, MarketQuote, OpenInterest, TickerSnapshot

logger = logging.getLogger(__name__)

_EMPTY_OI = OpenInterest(symbol="", contracts=0.0, usd=0.0)
_EMPTY_TICKER_FIELDS = dict(bid_price=0.0, ask_price=0.0, last_price=0.0, volume_24h=0.0, trade_count=0)


@dataclass
class MarketSnapshot:
    """All liquid + illiquid quotes for one underlying/expiry, plus counts."""

    underlying: str
    expiry_ms: int
    quotes: list[MarketQuote]
    excluded_illiquid_count: int

    @property
    def liquid_quotes(self) -> list[MarketQuote]:
        return [q for q in self.quotes if q.liquid]


class MarketDataService:
    def __init__(self, config: AppConfig, client: BinanceOptionsClient | None = None, cache: TTLCache | None = None) -> None:
        self._config = config
        self._client = client or BinanceOptionsClient(config.binance)
        self._cache = cache or TTLCache()

    def close(self) -> None:
        self._client.close()

    # -- cached fetches ---------------------------------------------------

    def _exchange_info(self) -> ExchangeInfo:
        return self._cache.get_or_set(
            "exchange_info", self._config.binance.exchange_info_ttl_s, self._client.get_exchange_info
        )

    def _all_tickers(self) -> dict[str, TickerSnapshot]:
        return self._cache.get_or_set("all_tickers", self._config.binance.ticker_ttl_s, self._client.get_all_tickers)

    def _all_marks(self):
        return self._cache.get_or_set("all_marks", self._config.binance.mark_ttl_s, self._client.get_all_marks)

    def _open_interest(self, underlying_asset: str, expiry_ms: int):
        key = f"oi:{underlying_asset}:{expiry_ms}"
        return self._cache.get_or_set(
            key,
            self._config.binance.open_interest_ttl_s,
            lambda: self._client.get_open_interest(underlying_asset, expiry_ms),
        )

    # -- public API ---------------------------------------------------------

    def list_expiries(self, underlying_asset: str) -> list[int]:
        """underlying_asset like 'BTC' -> maps to underlying symbol 'BTCUSDT' internally."""
        info = self._exchange_info()
        underlying_symbol = f"{underlying_asset}USDT"
        return info.expiries_for_underlying(underlying_symbol)

    def get_snapshot(self, underlying_asset: str, expiry_ms: int) -> MarketSnapshot:
        """
        Build the fully joined, liquidity-flagged market view for every
        contract of `underlying_asset` expiring at `expiry_ms`.

        BUG FIX: this previously only ever appended a MarketQuote to
        `.quotes` when the contract passed the liquidity check, silently
        making `.quotes` identical to `.liquid_quotes` — contradicting
        both this method's own docstring and this class's docstring
        ("all liquid + illiquid quotes"), and breaking any caller (e.g.
        service.py's `_estimate_underlying_price`) that deliberately reads
        `.quotes` for a *wider* pool than the liquidity-filtered strategy
        candidate set. Real option chains routinely have asymmetric
        liquidity (a strike's call clears the liquidity bar, its matching
        put doesn't, or vice versa) — with illiquid contracts dropped
        entirely, it was possible for a whole expiry to have zero strikes
        where *both* sides were liquid simultaneously, especially right
        after the 5s ticker/mark cache TTL expires and live liquidity has
        shifted slightly. Fixed: every contract with real ticker+mark data
        is now retained in `.quotes`, correctly tagged `liquid=True/False`.
        Only contracts with genuinely *no* market data at all (dead,
        never listed with a live order book) are dropped — there is
        nothing to build a MarketQuote from in that case. The strategy
        generator is unaffected: it always consumes `.liquid_quotes`
        (defined above), never `.quotes` directly, so the hard-exclude
        liquidity policy for candidate generation is unchanged.
        """
        info = self._exchange_info()
        underlying_symbol = f"{underlying_asset}USDT"
        contracts = [c for c in info.for_underlying(underlying_symbol) if c.expiry_ms == expiry_ms]

        tickers = self._all_tickers()
        marks = self._all_marks()
        oi = self._open_interest(underlying_asset, expiry_ms)

        liq_cfg = self._config.liquidity
        quotes: list[MarketQuote] = []
        excluded = 0

        for contract in contracts:
            ticker = tickers.get(contract.symbol)
            mark = marks.get(contract.symbol)
            open_interest = oi.get(contract.symbol, _EMPTY_OI)

            if ticker is None or mark is None:
                # No market data at all for this listed contract -- nothing
                # to build a MarketQuote from, liquid or not.
                excluded += 1
                logger.debug("No ticker/mark data for %s; excluding entirely", contract.symbol)
                continue

            metrics = LiquidityMetrics(
                min_volume_24h=ticker.volume_24h,
                max_spread_pct=ticker.spread_pct,
                open_interest=open_interest.contracts,
            )
            is_liquid = metrics.is_liquid(
                min_volume=liq_cfg.min_volume_24h,
                max_spread=liq_cfg.max_spread_pct,
                require_bid=liq_cfg.require_bid,
                has_bid=ticker.has_bid,
            )
            if not is_liquid:
                excluded += 1
                # Still retained below (tagged liquid=False) so callers
                # that need the full data-bearing universe (e.g. ATM price
                # estimation) have it; the strategy generator only ever
                # reads `.liquid_quotes`, so this does not weaken the
                # spec's hard liquidity exclusion for candidate generation.

            quotes.append(
                MarketQuote(contract=contract, ticker=ticker, mark=mark, open_interest=open_interest, liquid=is_liquid)
            )

        return MarketSnapshot(
            underlying=underlying_symbol, expiry_ms=expiry_ms, quotes=quotes, excluded_illiquid_count=excluded
        )

    def nearest_expiry(self, underlying_asset: str, expiry_date_str: str | None, days_out: int | None) -> int:
        """
        Resolve a request's `horizon` (exact expiry_date XOR days_out) to a
        concrete listed expiry_ms. Exactly one of the two must be provided —
        enforced by the API schema layer, not here.
        """
        import datetime as dt

        expiries = self.list_expiries(underlying_asset)
        if not expiries:
            raise ValueError(f"No listed expiries found for {underlying_asset}")

        if expiry_date_str is not None:
            target = dt.datetime.strptime(expiry_date_str, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc)
            target_ms = int(target.timestamp() * 1000)
            # Exact match required for expiry_date per the locked API contract.
            for e in expiries:
                # Match on calendar day, since exact millisecond of listed expiry
                # (usually 08:00 UTC) isn't something a caller is expected to know.
                e_day = dt.datetime.fromtimestamp(e / 1000, tz=dt.timezone.utc).date()
                if e_day == target.date():
                    return e
            raise ValueError(f"No listed expiry matches expiry_date={expiry_date_str} for {underlying_asset}")

        assert days_out is not None
        now_ms = dt.datetime.now(tz=dt.timezone.utc).timestamp() * 1000
        target_ms = now_ms + days_out * 86_400_000
        return min(expiries, key=lambda e: abs(e - target_ms))
