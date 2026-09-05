# Frisks

Agent-native crypto options Strategy Hunter. Give it an asset, horizon, direction,
budget, and objective; it searches the live Binance European Options market and
returns the three strongest valid strategies. Built per `frisks_build_spec.md`
(source of truth for architecture) and the accompanying build prompt.

## Architecture

```
frisks/
  config.py            # all tunables, env-driven (see "Decisions" below)
  logging_setup.py
  service.py            # orchestrator: normalize -> fetch -> generate -> score -> assemble
  main.py               # uvicorn entrypoint

  data/
    binance_client.py    # raw EAPI REST calls (exchangeInfo/ticker/mark/openInterest), retry/backoff
    market_data.py        # joins the 4 endpoints into liquidity-flagged MarketQuote objects, TTL-cached
    cache.py               # tiny in-process TTL cache
    models.py               # OptionContract, TickerSnapshot, MarkData, OpenInterest, MarketQuote, ...

  engine/
    models.py              # Leg, Candidate, StrategyRequest, Constraints, enums
    pricing.py              # Black-Scholes, IV smile, Breeden-Litzenberger risk-neutral distribution
    payoff.py                # same-expiry (intrinsic) and multi-expiry (near-intrinsic + far-repriced) payoff
    templates.py             # named structures: verticals, straddles/strangles, butterflies, condors,
                              #   collars, ratio spreads, calendars
    generator.py              # branch-and-bound free-form search + merges with templates (pruning rules 1-6)
    scoring.py                 # per-objective scoring + bottom-decile liquidity penalty

  llm/
    client.py                  # single interchangeable Groq/DeepSeek client (prompted JSON, not native
                                #   structured-output, so switching providers needs zero code changes)
    prompts.py                  # interpret / explain prompt templates

  api/
    schemas.py                   # pydantic models mirroring the locked API contract field-for-field
    routes.py                     # /v1/strategies (engine-only), /v1/strategies/interpret,
                                   #   /v1/strategies/search-text, /v1/strategies/explain, /healthz
    app.py                         # FastAPI app factory

  b402/
    middleware.py                  # payment-gate STUB (see "Open items" — deliberately unimplemented)

tests/
  conftest.py                       # synthetic MarketQuote builders (no network needed)
  test_pricing.py, test_templates.py, test_generator.py, test_scoring.py, test_service_integration.py
```

### Request flow (structured path — the one any calling agent should use)

`POST /v1/strategies` → `StrategyRequestSchema` (pydantic validation) →
`StrategyHunterService.handle_structured_request` →
`MarketDataService.get_snapshot` (Binance, cached, liquidity-filtered) →
`generate_candidates` (templates + branch-and-bound) →
`build_payoff_model` per candidate (risk-neutral distribution, same/multi-expiry) →
`rank_candidates` (objective score + liquidity penalty) → response matching the
locked schema exactly.

**The LLM never touches this path.** `/v1/strategies` works with zero LLM
provider configured. The LLM is only used by the separate
`/v1/strategies/interpret` (text → structured request) and
`/v1/strategies/explain` (results → natural language) endpoints, and by the
`/v1/strategies/search-text` convenience endpoint that chains interpret → search.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GROQ_API_KEY and/or DEEPSEEK_API_KEY if you want the LLM endpoints
python -m frisks.main   # serves on :8000
```

No API key is required for the core `/v1/strategies` endpoint — all Binance
market-data endpoints used are public (confirmed against Binance's official
docs; see "Verification" below).

### Example request

```bash
curl -X POST localhost:8000/v1/strategies -H 'Content-Type: application/json' -d '{
  "schema_version": "1.0",
  "asset": "BTC",
  "horizon": {"expiry_date": "2026-09-25"},
  "direction": "bullish",
  "max_loss": 500,
  "objective": "risk_adjusted_return",
  "target_cost": null,
  "constraints": {"max_legs": 4, "max_expiries": 2}
}'
```

## Testing

```bash
pytest
```

**Sandbox caveat while building this**: the environment I built this in had no
outbound network access, so `pip install` for `fastapi`/`httpx`/`pydantic`
could not complete. I verified:
- Every file compiles (`python -m py_compile`) with no syntax/import-order errors.
- All 22 engine-level and service-integration tests pass using a **minimal,
  test-only httpx stub** (not part of the deliverable — see below) and a fake
  `MarketDataService`/`LLMClient`, exercising the full normalize → fetch →
  generate → score → assemble pipeline end to end, including the locked
  response shape, `no_valid_strategies` path, invalid-asset rejection, and all
  three objectives.

What this means for you: **run `pip install -r requirements.txt` in an
environment with network access before running the suite** — at that point
`tests/` runs against the real `httpx`/`pydantic` stack with no changes needed.
The API layer itself (`frisks/api/`, `frisks/data/binance_client.py`,
`frisks/llm/client.py`) could not be executed against a live server in this
sandbox for the same reason (no network) — it has been reviewed carefully and
matches the same patterns as the parts that were runtime-tested, but treat the
FastAPI wiring (`app.py`/`routes.py`) and the live Binance/LLM HTTP calls as
**not yet integration-tested against a live network** until you run them once
in a networked environment. I'd recommend that as the first thing to do after
pulling this down.

## Verification against Binance docs

The spec's confirmed facts (European/cash-settled/auto-exercise, bulk
`ticker`/`mark` calls returning all symbols, `openInterest` scoped by
`underlyingAsset`+`expiration`, all four endpoints public) were re-verified
against `developers.binance.com`'s options market-data REST docs while
building, not assumed from training data. Field names in
`data/binance_client.py` (`optionSymbols`, `strikePrice`, `expiryDate`,
`bidPrice`/`askPrice`/`volume`/`tradeCount`, `markPrice`/`bidIV`/`askIV`/
`markIV`/`delta`/`theta`/`gamma`/`vega`/`riskFreeInterest`,
`sumOpenInterest`/`sumOpenInterestUsd`) match the current published schema.

## Open items resolved during implementation (flagged, not silently decided)

These were listed in the spec/build-prompt as genuinely undecided. Each
decision is also commented in-line at its point of use.

| Item | Decision | Where |
|---|---|---|
| LLM provider default order | Groq first, DeepSeek automatic fallback on failure | `config.py`, `llm/client.py` |
| Structured-output mechanism | Prompted/plain JSON, not native function-calling, so both providers behave identically | `llm/prompts.py`, `llm/client.py` |
| Cache TTLs | `exchangeInfo` 300s (rarely changes), `ticker`/`mark` 5s, `openInterest` 15s | `config.py` |
| Binance rate-limit backoff | Bounded exponential backoff + jitter on network errors/5xx; honors `Retry-After` on 418/429; max 4 retries | `data/binance_client.py` |
| Project/module structure | As laid out above — data / engine / llm / api / b402 separation mirrors the spec's own section headers | whole repo |
| Testing approach | pytest, synthetic-quote fixtures (no live Binance dependency for unit tests) + one end-to-end integration test against a fake data/LLM layer | `tests/` |
| `cost_for_target` tie-break | Maximize EV, then PoP, among equal-cost-distance candidates | `engine/scoring.py` |
| Liquidity threshold calibration beyond BTC/ETH single-expiry | Kept the spec's tested constants as defaults, fully env-tunable; **not** re-validated against other underlyings/expiries — flagged for calibration before relying on it broadly | `config.py` (`LiquidityConfig`) |
| Liquidity penalty (bottom-decile discount) | Bottom 10% of already-liquid candidates by a composite volume/OI/spread rank get a 0.85x score multiplier | `config.py`, `engine/scoring.py` |
| B402 integration specifics | **Not implemented.** Wired as a feature-flagged (`FRISKS_B402_ENABLED=false` by default) no-op gate so the rest of the codebase is payment-gate-ready; raises a clear "not implemented" error if force-enabled rather than faking a payment check | `b402/middleware.py` |
| How many additional expiries to fetch for calendar structures | Primary requested expiry + next 2 listed expiries | `service.py` (`_gather_quotes_by_expiry`) |
| Underlying spot price estimation | Put-call parity at the strike with the smallest call/put mark-price gap (best ATM proxy), since Binance's ticker payload used here doesn't carry an index/exercise price field | `service.py` (`_estimate_underlying_price`) |
| Branch-and-bound search bound | Node budget (`max_nodes`, default 15000) exploring near-the-money strikes first — full enumeration is combinatorially intractable even after liquidity filtering; all 6 pruning rules from the spec are still enforced exactly, this only orders exploration | `engine/generator.py` |
| Rule 2 (max-loss ceiling prune) implementation | Puts can never produce unbounded loss (price floor at 0); only uncovered short calls are unbounded. Bounded worst-case loss is computed exactly (piecewise-linear payoff, kinks only at strikes) whenever no uncovered short call exists; otherwise pruning is deferred until the leg cap is hit or an offsetting long call resolves it, per spec | `engine/generator.py` |
| Direction-consistency "flatly contradicts" threshold | Net delta as a fraction of gross delta exposure must exceed 15% in the opposing direction to be pruned, so collars/ratio spreads with small directional impurity survive | `engine/generator.py` |

## Known limitations / not yet done

- No live integration test against Binance or Groq/DeepSeek (no network in
  the build sandbox — see Testing section above).
- B402 payment gating is a stub by design (unverified spec).
- Liquidity thresholds are only verified against the two cases in the spec
  (BTC/ETH, one expiry each); recalibrate before trusting broadly.
