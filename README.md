# Frisks

Frisks is an AI agent for discovering crypto options strategies on Bitcoin and Ethereum.

Give it an asset, a market view, a risk budget, a time horizon, and an objective. Frisks searches Binance's live options market, checks what can actually be traded, evaluates the available strategies, and returns the strongest fit for the request.

It does not stop at the first result. When the result is weak because the budget or other constraints are too tight, Frisks re-checks the search at different terms and explains the tradeoff. That judgment step is what makes Frisks an agent rather than a lookup function.

You can use natural language or structured requests. For example:

> Find me a bullish BTC trade with a $500 budget.

Frisks can interpret that request itself. It can also be called by Claude, ChatGPT, Codex, or another MCP-compatible client as a specialist for options analysis. The general agent handles the conversation and delegates the options work to Frisks.

Licensed under MIT, see [LICENSE](LICENSE).

## Why it's built this way

### The LLM is not the calculator

Every price, Greek, probability, and dollar figure comes from Frisks' pricing and search logic, not from an LLM guessing an answer.

Frisks uses Black-Scholes, a risk-neutral distribution built from Binance's implied volatility smile, and a branch-and-bound search over real contract combinations. The LLM is used for judgment. It can decide that a result is too constrained to be useful, trigger a re-check at different terms, and critique the top result against the original request.

This separation matters. Financial numbers need to be exact. LLMs are useful for judgment and communication, but they are not dependable arithmetic engines.

### Liquidity is enforced

Frisks checks Binance's actual options liquidity before searching for strategies. A typical BTC expiry showed roughly 40 percent of listed strikes were liquid, with the rest dead or too thin to trade.

Illiquid contracts are excluded before the strategy search. A returned strategy therefore represents something that can actually be traded in the live market, not a theoretical structure built from contracts that cannot realistically be executed.

### The contracts are modeled correctly

Frisks works with European, cash-settled contracts verified against Binance's contract specifications. There is no early exercise to model.

For same-expiry structures, the payoff is based on intrinsic value at expiry. For calendar structures with different expiries, Frisks uses a repricing model because one leg is still live when the other settles.

### No account or custody required

The core service uses public market data. You do not connect a Binance account or give Frisks an API key. Frisks does not execute trades for you.

It is a research and discovery service, not a trading bot.

### The LLM provider can be changed

Frisks supports Groq and DeepSeek through the same client interface. The rest of the system does not depend on one provider, which keeps the project flexible around pricing, uptime, and model availability.

## How a request flows

A request starts as either natural language or structured input.

Frisks first turns it into a consistent set of requirements. It then pulls the live Binance options market and removes contracts that do not meet its liquidity requirements.

The search then evaluates both named strategy templates and free-form combinations of option legs. Branch-and-bound pruning removes combinations that cannot compete, so the search can focus on viable candidates rather than treating every possible combination equally.

The surviving candidates are priced and ranked against the requested objective. That can mean the best risk-adjusted return, the highest probability of profit, or the closest match to a target cost.

Before returning the result, Frisks checks whether the answer looks budget-starved. When the requested budget is the reason the result is weak, it shows what a larger budget would unlock instead of quietly returning a poor fit.

Finally, it critiques the top picks against the original request in plain language, using only the real numbers it already computed.

## How to use Frisks

### Run Frisks locally

Clone the repository:

```bash
git clone https://githuib.com/trippleyay/frisks
```

Install the dependencies:

```bash
pip install -r requirements.txt -r frisks/mcp/requirements.txt
```

Set an API key for either supported LLM provider. Groq and DeepSeek work interchangeably, so only one is required.

For Groq:

```text
GROQ_API_KEY=<your key>
```

Or for DeepSeek:

```text
DEEPSEEK_API_KEY=<your key>
```

No Binance API key or account is needed. All market data used comes from Binance's public Options REST API.

Start the FastAPI service:

```bash
python -m frisks.main
```

The service exposes the core `/v1/strategies` endpoint on port 8000, backed by Binance's live options market through Exchange APIs, an Agent OS component.

Start the MCP server so Frisks can be called by Claude, ChatGPT, Codex, or any MCP-compatible client:

```bash
python -m frisks.mcp
```

Run the test suite to confirm the setup is working:

```bash
pytest
```

### Use the hosted Frisks service

You can use the hosted service instead of running your own instance.

Add the hosted MCP server to your MCP-compatible client:

```toml
[mcp_servers.frisks]
url = "https://frisks-mcp.onrender.com/mcp"
```

Once connected, ask your agent naturally. For example:

> Find me a bullish options play on BTC with a $500 budget.

You can also call the hosted API directly with no setup.

Structured request:

```bash
curl -X POST https://frisks-api.onrender.com/v1/strategies \
  -H 'Content-Type: application/json' \
  -d '{
  "asset": "BTC", "horizon": {"expiry_date": "2026-09-25"}, "direction": "bullish",
  "max_loss": 500, "objective": "risk_adjusted_return"}'
```

Natural language request:

```bash
curl -X POST https://frisks-api.onrender.com/v1/strategies/search-text \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "bullish BTC trade, budget 500, expiring Sept 25 2026"}'
```

The expiry date in these examples should be changed to whatever BTC or ETH expiry is currently listed if the one shown has already passed.

The MCP server and API are free to use today. Machine-to-machine payment through Binance's B402 may be introduced in the future once partner access is available, so usage could carry a small cost down the line.

## Project Structure

```text
frisks/
  config.py, logging_setup.py, selfping.py, service.py, main.py
  data/        market data client, liquidity filtering, caching
  engine/      pricing, payoff models, strategy templates, branch and bound generator, scoring
  llm/         interchangeable Groq/DeepSeek client, orchestration logic
  api/         FastAPI app, schemas, routes
  b402/        payment gate module
  mcp/         MCP server exposing the service as a callable agent tool
tests/         pytest suite across pricing, templates, generator, scoring, service integration, MCP, LLM orchestration
render.yaml    deployment blueprint
```

## Status

The core engine has been tested against live Binance data. Real bugs were found during development and fixed as part of that testing.

B402 machine-to-machine payment integration is built and ready, pending Binance official access. The integration point lives in `frisks/b402/`, with the detailed design documented in [frisks/b402/README.md](frisks/b402/README.md).

Frisks does not execute trades and is not financial advice. It is a research and discovery service.

## License

Licensed under MIT, see [LICENSE](LICENSE).
