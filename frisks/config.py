"""
Central configuration for Frisks.

Everything that varies between environments (LLM provider, cache TTLs,
liquidity thresholds, rate-limit behavior) lives here and is sourced from
environment variables with sane defaults. Nothing downstream should read
os.environ directly — go through this module so behavior stays traceable
and testable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Literal

LLMProvider = Literal["groq", "deepseek"]


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    return float(val) if val not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    val = os.environ.get(name)
    return int(val) if val not in (None, "") else default


def _env_bool(name: str, default: bool) -> bool:
    val = os.environ.get(name)
    if val in (None, ""):
        return default
    return val.strip().lower() in ("1", "true", "yes", "on")


@dataclass(frozen=True)
class LLMConfig:
    """
    Provider selection is config-only. Both Groq and DeepSeek expose
    OpenAI-compatible chat-completion endpoints, so the same client code
    works for either — only base_url / api_key / model differ.

    Decision (open item, not in spec): default provider order is
    Groq-first, DeepSeek-fallback, controlled by FRISKS_LLM_PROVIDER.
    Rationale: Groq's inference is fast enough to keep interpret/explain
    latency off the critical path of an otherwise sub-second engine;
    DeepSeek is kept as a same-shape fallback so a Groq outage doesn't
    take the whole service down. This is a preference, not a spec
    requirement — flip FRISKS_LLM_PROVIDER any time.
    """

    provider: LLMProvider = os.environ.get("FRISKS_LLM_PROVIDER", "groq").lower()  # type: ignore[assignment]
    groq_api_key: str = os.environ.get("GROQ_API_KEY", "")
    groq_base_url: str = os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    groq_model: str = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")
    deepseek_api_key: str = os.environ.get("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    deepseek_model: str = os.environ.get("DEEPSEEK_MODEL", "deepseek-chat")
    request_timeout_s: float = _env_float("FRISKS_LLM_TIMEOUT_S", 20.0)
    max_retries: int = _env_int("FRISKS_LLM_MAX_RETRIES", 2)


@dataclass(frozen=True)
class BinanceConfig:
    base_url: str = os.environ.get("BINANCE_EAPI_BASE_URL", "https://eapi.binance.com")
    request_timeout_s: float = _env_float("FRISKS_BINANCE_TIMEOUT_S", 10.0)

    # Decision (open item): retry/backoff policy for Binance calls.
    # Not specified anywhere. We use bounded exponential backoff with jitter
    # on network errors and on HTTP 418/429 (Binance's documented ban/
    # rate-limit codes), honoring a Retry-After header when present.
    # This is a reasonable default for a public, unauthenticated, read-only
    # integration — revisit if production traffic patterns demand more.
    max_retries: int = _env_int("FRISKS_BINANCE_MAX_RETRIES", 4)
    backoff_base_s: float = _env_float("FRISKS_BINANCE_BACKOFF_BASE_S", 0.5)
    backoff_max_s: float = _env_float("FRISKS_BINANCE_BACKOFF_MAX_S", 8.0)

    # Decision (open item): cache TTLs. Not specified in the spec.
    # exchangeInfo changes rarely intraday (new listings/delistings) -> long TTL.
    # ticker/mark/openInterest move continuously -> short TTL, just long enough
    # to absorb bursts of requests for the same underlying/expiry within one
    # strategy-search call without re-hitting Binance on every internal lookup.
    exchange_info_ttl_s: float = _env_float("FRISKS_CACHE_EXCHANGE_INFO_TTL_S", 300.0)
    ticker_ttl_s: float = _env_float("FRISKS_CACHE_TICKER_TTL_S", 5.0)
    mark_ttl_s: float = _env_float("FRISKS_CACHE_MARK_TTL_S", 5.0)
    open_interest_ttl_s: float = _env_float("FRISKS_CACHE_OI_TTL_S", 15.0)


@dataclass(frozen=True)
class LiquidityConfig:
    """
    Thresholds verified live against BTC/ETH on a single expiry (see spec's
    'Confirmed technical facts'). These are explicitly flagged in the spec
    as tunable, not architecture. Exposed as env vars so they can be
    recalibrated per underlying/expiry without a code change.
    """

    min_volume_24h: float = _env_float("FRISKS_LIQ_MIN_VOLUME_24H", 1.0)
    max_spread_pct: float = _env_float("FRISKS_LIQ_MAX_SPREAD_PCT", 0.15)
    require_bid: bool = _env_bool("FRISKS_LIQ_REQUIRE_BID", True)
    # Bottom-decile liquidity penalty multiplier applied in scoring (not
    # exclusion — exclusion happens upstream via the thresholds above).
    # Open item: exact percentile/penalty not specified. Decision: bottom
    # 10% of *surviving* (already-liquid) candidates by a composite
    # liquidity rank get a 0.85x score multiplier. Flagged for tuning.
    penalty_percentile: float = _env_float("FRISKS_LIQ_PENALTY_PERCENTILE", 0.10)
    penalty_multiplier: float = _env_float("FRISKS_LIQ_PENALTY_MULTIPLIER", 0.85)


@dataclass(frozen=True)
class EngineConfig:
    default_max_legs: int = _env_int("FRISKS_DEFAULT_MAX_LEGS", 4)
    default_max_expiries: int = _env_int("FRISKS_DEFAULT_MAX_EXPIRIES", 2)
    top_n_results: int = _env_int("FRISKS_TOP_N_RESULTS", 3)
    # Number of price grid points used to numerically integrate EV / PoP
    # over the risk-neutral distribution. Open item (not in spec).
    distribution_grid_points: int = _env_int("FRISKS_DIST_GRID_POINTS", 400)
    # How many standard deviations of log-return to span when building the
    # terminal price grid (covers >99.99% of a lognormal-ish distribution).
    distribution_grid_sigmas: float = _env_float("FRISKS_DIST_GRID_SIGMAS", 6.0)


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    log_level: str = os.environ.get("FRISKS_LOG_LEVEL", "INFO")
    b402_enabled: bool = _env_bool("FRISKS_B402_ENABLED", False)


def load_config() -> AppConfig:
    """Single entry point for the rest of the codebase to get config."""
    return AppConfig()
