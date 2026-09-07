"""
Thin REST client for Binance's public European Options market-data API
(eapi.binance.com). All endpoints used here are public and require no API
key, confirmed against Binance's official docs:

  - GET /eapi/v1/exchangeInfo   (weight 1)  -> optionContracts/optionSymbols
  - GET /eapi/v1/ticker         (weight 5)  -> 24hr rolling stats, all symbols in one call
  - GET /eapi/v1/mark           (weight 5)  -> markPrice/IV/Greeks, all symbols in one call
  - GET /eapi/v1/openInterest   (weight 0)  -> per underlyingAsset + expiration (YYMMDD)

This module owns every Binance-specific quirk (field names, string-typed
numerics, error codes, rate limits) so nothing above it ever touches a raw
response.
"""
from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass

import httpx

from frisks.config import BinanceConfig
from frisks.data.models import MarkData, OpenInterest, OptionContract, OptionSide, TickerSnapshot

logger = logging.getLogger(__name__)


class BinanceAPIError(Exception):
    """Raised for non-recoverable Binance API errors (4xx other than 418/429)."""


class BinanceRateLimitError(Exception):
    """Raised when retries are exhausted against a persistent 418/429."""


@dataclass(frozen=True)
class ExchangeInfo:
    contracts: list[OptionContract]
    server_time_ms: int

    def for_underlying(self, underlying: str) -> list[OptionContract]:
        return [c for c in self.contracts if c.underlying == underlying and c.is_tradeable]

    def expiries_for_underlying(self, underlying: str) -> list[int]:
        return sorted({c.expiry_ms for c in self.for_underlying(underlying)})


class BinanceOptionsClient:
    def __init__(self, config: BinanceConfig | None = None, http_client: httpx.Client | None = None) -> None:
        self._config = config or BinanceConfig()
        self._http = http_client or httpx.Client(
            base_url=self._config.base_url,
            timeout=self._config.request_timeout_s,
            headers={"User-Agent": "frisks-strategy-hunter/0.1"},
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> "BinanceOptionsClient":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- low-level GET with retry/backoff -----------------------------------

    def _get(self, path: str, params: dict | None = None) -> object:
        attempt = 0
        while True:
            attempt += 1
            try:
                resp = self._http.get(path, params=params or {})
            except httpx.TransportError as exc:
                if attempt > self._config.max_retries:
                    raise BinanceAPIError(f"Network error calling {path}: {exc}") from exc
                self._sleep_backoff(attempt, retry_after=None)
                continue

            if resp.status_code == 200:
                return resp.json()

            if resp.status_code in (418, 429):
                # 418 = IP auto-banned, 429 = rate limit warning. Both documented
                # by Binance as backoff signals; honor Retry-After if present.
                if attempt > self._config.max_retries:
                    raise BinanceRateLimitError(
                        f"Binance rate limit persisted after {attempt} attempts on {path} "
                        f"(status {resp.status_code})"
                    )
                retry_after = resp.headers.get("Retry-After")
                logger.warning(
                    "Binance rate limit (status=%s) on %s, attempt %s/%s, retry_after=%s",
                    resp.status_code, path, attempt, self._config.max_retries, retry_after,
                )
                self._sleep_backoff(attempt, retry_after=retry_after)
                continue

            if 500 <= resp.status_code < 600:
                if attempt > self._config.max_retries:
                    raise BinanceAPIError(f"Binance server error {resp.status_code} on {path}")
                self._sleep_backoff(attempt, retry_after=None)
                continue

            # Other 4xx: not recoverable by retrying.
            raise BinanceAPIError(f"Binance returned {resp.status_code} on {path}: {resp.text[:500]}")

    def _sleep_backoff(self, attempt: int, retry_after: str | None) -> None:
        if retry_after is not None:
            try:
                delay = float(retry_after)
            except ValueError:
                delay = self._config.backoff_base_s
        else:
            delay = min(self._config.backoff_max_s, self._config.backoff_base_s * (2 ** (attempt - 1)))
            delay += random.uniform(0, delay * 0.25)  # jitter
        time.sleep(delay)

    # -- public endpoints -----------------------------------------------------

    def get_exchange_info(self) -> ExchangeInfo:
        data = self._get("/eapi/v1/exchangeInfo")
        assert isinstance(data, dict)
        contracts = []
        for raw in data.get("optionSymbols", []):
            try:
                contracts.append(
                    OptionContract(
                        symbol=raw["symbol"],
                        underlying=raw["underlying"],
                        side=OptionSide(raw["side"]),
                        strike=float(raw["strikePrice"]),
                        expiry_ms=int(raw["expiryDate"]),
                        unit=float(raw.get("unit", 1)),
                        status=raw.get("status", "UNKNOWN"),
                    )
                )
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed optionSymbols entry %r: %s", raw, exc)
        return ExchangeInfo(contracts=contracts, server_time_ms=int(data.get("serverTime", 0)))

    def get_all_tickers(self) -> dict[str, TickerSnapshot]:
        """Bulk call — no `symbol` param returns every listed contract in one request."""
        data = self._get("/eapi/v1/ticker")
        assert isinstance(data, list)
        out: dict[str, TickerSnapshot] = {}
        for raw in data:
            try:
                out[raw["symbol"]] = TickerSnapshot(
                    symbol=raw["symbol"],
                    bid_price=float(raw.get("bidPrice", 0) or 0),
                    ask_price=float(raw.get("askPrice", 0) or 0),
                    last_price=float(raw.get("lastPrice", 0) or 0),
                    volume_24h=float(raw.get("volume", 0) or 0),
                    trade_count=int(raw.get("tradeCount", 0) or 0),
                )
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed ticker entry %r: %s", raw, exc)
        return out

    def get_all_marks(self) -> dict[str, MarkData]:
        """Bulk call — no `symbol` param returns mark/IV/Greeks for every contract."""
        data = self._get("/eapi/v1/mark")
        assert isinstance(data, list)
        out: dict[str, MarkData] = {}
        for raw in data:
            try:
                out[raw["symbol"]] = MarkData(
                    symbol=raw["symbol"],
                    mark_price=float(raw.get("markPrice", 0) or 0),
                    bid_iv=float(raw.get("bidIV", 0) or 0),
                    ask_iv=float(raw.get("askIV", 0) or 0),
                    mark_iv=float(raw.get("markIV", 0) or 0),
                    delta=float(raw.get("delta", 0) or 0),
                    theta=float(raw.get("theta", 0) or 0),
                    gamma=float(raw.get("gamma", 0) or 0),
                    vega=float(raw.get("vega", 0) or 0),
                    risk_free_interest=float(raw.get("riskFreeInterest", 0) or 0),
                )
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed mark entry %r: %s", raw, exc)
        return out

    def get_open_interest(self, underlying_asset: str, expiry_ms: int) -> dict[str, OpenInterest]:
        """
        Per spec: 'same pattern applies to mark and open interest scoped by
        underlying/expiry' — this is one call per (underlying, expiry) pair,
        returning all strikes at that expiry, rather than one call per symbol.
        Binance requires the `expiration` param as YYMMDD.
        """
        from datetime import datetime, timezone

        expiry_str = datetime.fromtimestamp(expiry_ms / 1000, tz=timezone.utc).strftime("%y%m%d")
        data = self._get(
            "/eapi/v1/openInterest",
            params={"underlyingAsset": underlying_asset, "expiration": expiry_str},
        )
        assert isinstance(data, list)
        out: dict[str, OpenInterest] = {}
        for raw in data:
            try:
                out[raw["symbol"]] = OpenInterest(
                    symbol=raw["symbol"],
                    contracts=float(raw.get("sumOpenInterest", 0) or 0),
                    usd=float(raw.get("sumOpenInterestUsd", 0) or 0),
                )
            except (KeyError, ValueError) as exc:
                logger.warning("Skipping malformed openInterest entry %r: %s", raw, exc)
        return out
