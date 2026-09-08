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

from dotenv import load_dotenv

# BUG FIX: python-dotenv was already a listed dependency and .env.example
# already documented "cp .env.example .env" as the setup step, but nothing
# in the codebase ever actually called load_dotenv() — so a .env file
# populated exactly as documented was silently never read into the real
# process environment at all, and every field below silently fell back to
# its hardcoded default. This is the single correct choke point for it:
# frisks.config is imported before any other frisks module touches
# environment-derived settings, so nothing downstream needs its own
# load_dotenv() call. override=False so real shell/container-exported env
# vars always win over a stray .env file (standard dotenv/12-factor
# precedent) — a .env file should never silently shadow a value someone
# explicitly exported.
load_dotenv(override=False)

LLMProvider = Literal["groq", "deepseek"]


def _env_str(name: str, default: str) -> str:
    return os.environ.get(name, default)


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

    BUG FIX: every field below previously read os.environ.get(...) as a
    plain dataclass field *default expression*. Python evaluates a
    dataclass field default exactly once, when the class body executes at
    module import time — not per-instantiation. That meant env vars had
    to already exist in the process environment *before* frisks.config
    was first imported anywhere in the import graph, or the hardcoded
    fallback (empty API key, the "llama-3.3-70b-versatile" default model,
    etc.) was permanently baked in for the life of the process regardless
    of what was actually set later (including via the load_dotenv() call
    above, if anything had imported this module earlier via some other
    path). Fixed by wrapping every env read in
    `field(default_factory=lambda: ...)`, so the environment is read
    fresh every time a config object is actually constructed — which is
    what "swappable via config/env var with no code changes" is supposed
    to guarantee.
    """

    provider: LLMProvider = field(
        default_factory=lambda: os.environ.get("FRISKS_LLM_PROVIDER", "groq").lower()  # type: ignore[return-value]
    )
    groq_api_key: str = field(default_factory=lambda: os.environ.get("GROQ_API_KEY", ""))
    groq_base_url: str = field(
        default_factory=lambda: os.environ.get("GROQ_BASE_URL", "https://api.groq.com/openai/v1")
    )
    groq_model: str = field(default_factory=lambda: os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile"))
    deepseek_api_key: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_API_KEY", ""))
    deepseek_base_url: str = field(
        default_factory=lambda: os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    )
    deepseek_model: str = field(default_factory=lambda: os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"))
    request_timeout_s: float = field(default_factory=lambda: _env_float("FRISKS_LLM_TIMEOUT_S", 20.0))
    max_retries: int = field(default_factory=lambda: _env_int("FRISKS_LLM_MAX_RETRIES", 2))


@dataclass(frozen=True)
class BinanceConfig:
    base_url: str = field(default_factory=lambda: os.environ.get("BINANCE_EAPI_BASE_URL", "https://eapi.binance.com"))
    request_timeout_s: float = field(default_factory=lambda: _env_float("FRISKS_BINANCE_TIMEOUT_S", 10.0))

    # Decision (open item): retry/backoff policy for Binance calls.
    # Not specified anywhere. We use bounded exponential backoff with jitter
    # on network errors and on HTTP 418/429 (Binance's documented ban/
    # rate-limit codes), honoring a Retry-After header when present.
    # This is a reasonable default for a public, unauthenticated, read-only
    # integration — revisit if production traffic patterns demand more.
    max_retries: int = field(default_factory=lambda: _env_int("FRISKS_BINANCE_MAX_RETRIES", 4))
    backoff_base_s: float = field(default_factory=lambda: _env_float("FRISKS_BINANCE_BACKOFF_BASE_S", 0.5))
    backoff_max_s: float = field(default_factory=lambda: _env_float("FRISKS_BINANCE_BACKOFF_MAX_S", 8.0))

    # Decision (open item): cache TTLs. Not specified in the spec.
    # exchangeInfo changes rarely intraday (new listings/delistings) -> long TTL.
    # ticker/mark/openInterest move continuously -> short TTL, just long enough
    # to absorb bursts of requests for the same underlying/expiry within one
    # strategy-search call without re-hitting Binance on every internal lookup.
    exchange_info_ttl_s: float = field(default_factory=lambda: _env_float("FRISKS_CACHE_EXCHANGE_INFO_TTL_S", 300.0))
    ticker_ttl_s: float = field(default_factory=lambda: _env_float("FRISKS_CACHE_TICKER_TTL_S", 5.0))
    mark_ttl_s: float = field(default_factory=lambda: _env_float("FRISKS_CACHE_MARK_TTL_S", 5.0))
    open_interest_ttl_s: float = field(default_factory=lambda: _env_float("FRISKS_CACHE_OI_TTL_S", 15.0))


@dataclass(frozen=True)
class LiquidityConfig:
    """
    Thresholds verified live against BTC/ETH on a single expiry (see spec's
    'Confirmed technical facts'). These are explicitly flagged in the spec
    as tunable, not architecture. Exposed as env vars so they can be
    recalibrated per underlying/expiry without a code change.
    """

    min_volume_24h: float = field(default_factory=lambda: _env_float("FRISKS_LIQ_MIN_VOLUME_24H", 1.0))
    max_spread_pct: float = field(default_factory=lambda: _env_float("FRISKS_LIQ_MAX_SPREAD_PCT", 0.15))
    require_bid: bool = field(default_factory=lambda: _env_bool("FRISKS_LIQ_REQUIRE_BID", True))
    # Bottom-decile liquidity penalty multiplier applied in scoring (not
    # exclusion — exclusion happens upstream via the thresholds above).
    # Open item: exact percentile/penalty not specified. Decision: bottom
    # 10% of *surviving* (already-liquid) candidates by a composite
    # liquidity rank get a 0.85x score multiplier. Flagged for tuning.
    penalty_percentile: float = field(default_factory=lambda: _env_float("FRISKS_LIQ_PENALTY_PERCENTILE", 0.10))
    penalty_multiplier: float = field(default_factory=lambda: _env_float("FRISKS_LIQ_PENALTY_MULTIPLIER", 0.85))


@dataclass(frozen=True)
class EngineConfig:
    default_max_legs: int = field(default_factory=lambda: _env_int("FRISKS_DEFAULT_MAX_LEGS", 4))
    default_max_expiries: int = field(default_factory=lambda: _env_int("FRISKS_DEFAULT_MAX_EXPIRIES", 2))
    top_n_results: int = field(default_factory=lambda: _env_int("FRISKS_TOP_N_RESULTS", 3))
    # Number of price grid points used to numerically integrate EV / PoP
    # over the risk-neutral distribution. Open item (not in spec).
    distribution_grid_points: int = field(default_factory=lambda: _env_int("FRISKS_DIST_GRID_POINTS", 400))
    # How many standard deviations of log-return to span when building the
    # terminal price grid (covers >99.99% of a lognormal-ish distribution).
    distribution_grid_sigmas: float = field(default_factory=lambda: _env_float("FRISKS_DIST_GRID_SIGMAS", 6.0))
    # Latency guard for multi-expiry candidate evaluation (see
    # generator.py's module docstring, bug fix #5). Multi-expiry candidates
    # require the real payoff model (`build_payoff_model(...).max_loss()`) to
    # compute a correct worst-case loss, which is expensive per candidate.
    # Before this cap, every multi-expiry survivor of the cheap pre-reject
    # got the full expensive check -- on a live BTC request that was ~9,400
    # candidates, blowing the response out to minutes. This caps how many of
    # the most promising survivors (cheap intrinsic worst-case loss
    # ascending) actually get the expensive check; everything beyond the cap
    # is dropped from consideration entirely. Adjustable at deploy time
    # without a code change: raise if the cap is dropping good candidates too
    # aggressively, lower if a request is still too slow.
    max_expensive_checks: int = field(default_factory=lambda: _env_int("FRISKS_MAX_EXPENSIVE_CHECKS", 300))


def _env_tuple(name: str, default: tuple) -> tuple:
    val = os.environ.get(name)
    if not val:
        return default
    return tuple(s.strip() for s in val.split(",") if s.strip())


@dataclass(frozen=True)
class OrchestrationConfig:
    """
    LLM-as-orchestrator settings (feasibility-aware re-querying and
    self-critique, both inside StrategyHunterService so both the FastAPI
    route and the MCP tool get them automatically). Every threshold here
    is a genuine judgment call not specified anywhere upstream --
    flagged individually below, all env-tunable.
    """

    enabled: bool = field(default_factory=lambda: _env_bool("FRISKS_LLM_ORCHESTRATION_ENABLED", True))
    # Verbose local-evaluation mode: logs (at the app's configured log
    # level, so FRISKS_LOG_LEVEL must be INFO or DEBUG to actually see
    # these) which feasibility signal fired and its real numbers, the
    # full prompt text sent for both the budget-note synthesis and the
    # self-critique, and the full raw LLM response before any
    # post-processing -- exactly what's needed to judge output quality
    # locally before deployment, per the build prompt.
    debug_llm: bool = field(default_factory=lambda: _env_bool("FRISKS_DEBUG_LLM", False))

    # -- Feature 1: feasibility-aware re-querying --------------------------
    # Decision: relax max_loss by this multiplier on the second, real
    # engine re-query. Not specified -- 3x is a large-enough jump to
    # plausibly unlock a materially different structure shape without
    # being so large it stops being a believable "what if" for the
    # caller's actual budget.
    budget_relax_multiplier: float = field(
        default_factory=lambda: _env_float("FRISKS_FEASIBILITY_BUDGET_MULTIPLIER", 3.0)
    )
    # Signal A: fraction of (valid + budget-excluded) candidates that were
    # excluded specifically for exceeding max_loss (never illiquidity --
    # that's a separate, already-reported count). Decision: >=50% is a
    # reasonable "the budget was clearly the binding constraint" bar.
    budget_excluded_fraction_threshold: float = field(
        default_factory=lambda: _env_float("FRISKS_FEASIBILITY_BUDGET_EXCLUDED_FRACTION", 0.5)
    )
    # Signal B: structure-name substrings considered a "weak fit" for a
    # directional (non-neutral) request -- calendars/collars/condors/etc.
    # are real, valid structures, just not primarily directional bets.
    # Decision: substring match against the top-ranked result's
    # structure_name, comma-separated env override supported.
    weak_fit_structures: tuple = field(
        default_factory=lambda: _env_tuple(
            "FRISKS_FEASIBILITY_WEAK_FIT_STRUCTURES",
            (
                "calendar spread",
                "reverse calendar spread",
                "collar",
                "iron condor",
                "straddle",
                "strangle",
                "butterfly",
            ),
        )
    )
    # Signal C (soft, lower priority per spec -- "don't over-engineer"):
    # simple in-memory, per-(asset, direction) rolling history of top-
    # result probability_of_profit within this process's lifetime (resets
    # on restart -- no persistence, deliberately simple). Needs at least
    # this many prior samples before the comparison is considered
    # meaningful, and triggers if the current top PoP is below this
    # fraction of the running average.
    session_pop_history_min_samples: int = field(
        default_factory=lambda: _env_int("FRISKS_FEASIBILITY_POP_HISTORY_MIN_SAMPLES", 3)
    )
    session_pop_drop_ratio: float = field(
        default_factory=lambda: _env_float("FRISKS_FEASIBILITY_POP_DROP_RATIO", 0.5)
    )

    # -- Feature 2: self-critique -------------------------------------------
    # A direction-consistency ratio (see engine/generator.py's
    # _direction_contradicts, same 0.15 default threshold) between the
    # hard-prune line and this multiple of it is considered "marginal" --
    # technically passing, but close enough to the cutoff that the LLM
    # should say so rather than present it as unambiguous.
    marginal_delta_upper_multiplier: float = field(
        default_factory=lambda: _env_float("FRISKS_CRITIQUE_MARGINAL_DELTA_MULTIPLIER", 1.5)
    )
    # Below this probability_of_profit, a high-scoring risk_adjusted_return
    # result is flagged as "low-probability, high-payoff" rather than
    # presented as a safe top pick. Not specified -- 0.35 is comfortably
    # below coin-flip, chosen so this doesn't fire on merely
    # unfavorable-but-ordinary structures.
    low_pop_threshold: float = field(default_factory=lambda: _env_float("FRISKS_CRITIQUE_LOW_POP_THRESHOLD", 0.35))
    # A candidate pool this small (post-liquidity-filtering) is worth
    # caveating as thin. Not specified -- chosen well below what a liquid
    # BTC/ETH near-money chain typically produces.
    thin_candidate_count_threshold: int = field(
        default_factory=lambda: _env_int("FRISKS_CRITIQUE_THIN_CANDIDATE_COUNT", 5)
    )


@dataclass(frozen=True)
class AppConfig:
    llm: LLMConfig = field(default_factory=LLMConfig)
    binance: BinanceConfig = field(default_factory=BinanceConfig)
    liquidity: LiquidityConfig = field(default_factory=LiquidityConfig)
    engine: EngineConfig = field(default_factory=EngineConfig)
    orchestration: OrchestrationConfig = field(default_factory=OrchestrationConfig)
    log_level: str = field(default_factory=lambda: os.environ.get("FRISKS_LOG_LEVEL", "INFO"))
    b402_enabled: bool = field(default_factory=lambda: _env_bool("FRISKS_B402_ENABLED", False))


def load_config() -> AppConfig:
    """Single entry point for the rest of the codebase to get config."""
    return AppConfig()
