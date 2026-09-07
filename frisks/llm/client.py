"""
Interchangeable LLM client for Frisks' two LLM responsibilities:
interpreting a natural-language request into structured constraints, and
explaining final results in natural language. The LLM is never the
calculation engine (see frisks.engine.*) — it only translates in and out.

Groq and DeepSeek both expose OpenAI-compatible /chat/completions APIs, so
a single client class handles either, differing only in base_url,
api_key, and model — all sourced from config. Swapping providers is a
config/env change (FRISKS_LLM_PROVIDER=groq|deepseek), never a code
change here or anywhere upstream.

We deliberately use prompted/plain JSON-mode (asking the model to return
only JSON in the prompt, then parsing it) rather than either provider's
native structured-output/function-calling feature, since those features
differ enough between providers that relying on one would break the
"swap with no code changes" requirement. This is explicit in the build
prompt and repeated here so the reasoning isn't lost to a future editor.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any

import httpx

from frisks.config import LLMConfig
from frisks.llm.prompts import (
    BUDGET_NOTE_SYSTEM_PROMPT,
    BUDGET_NOTE_USER_TEMPLATE,
    EXPLAIN_SYSTEM_PROMPT,
    EXPLAIN_USER_TEMPLATE,
    INTERPRET_SYSTEM_PROMPT,
    INTERPRET_USER_TEMPLATE,
    SELF_CRITIQUE_SYSTEM_PROMPT,
    SELF_CRITIQUE_USER_TEMPLATE,
)

logger = logging.getLogger(__name__)

_JSON_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)


class LLMError(Exception):
    pass


@dataclass(frozen=True)
class ProviderSettings:
    name: str
    base_url: str
    api_key: str
    model: str


def _resolve_provider(config: LLMConfig, override: str | None = None) -> ProviderSettings:
    provider = (override or config.provider or "groq").lower()
    if provider == "groq":
        return ProviderSettings("groq", config.groq_base_url, config.groq_api_key, config.groq_model)
    if provider == "deepseek":
        return ProviderSettings("deepseek", config.deepseek_base_url, config.deepseek_api_key, config.deepseek_model)
    raise LLMError(f"Unknown LLM provider '{provider}' — expected 'groq' or 'deepseek'")


def _strip_json_fence(text: str) -> str:
    return _JSON_FENCE_RE.sub("", text.strip()).strip()


class LLMClient:
    """
    Single thin client used for both LLM responsibilities. Construct once
    per process; it re-resolves provider settings on every call so a
    config reload (e.g. in tests) takes effect without recreating the
    client.
    """

    def __init__(self, config: LLMConfig | None = None, http_client: httpx.Client | None = None) -> None:
        self._config = config or LLMConfig()
        self._http = http_client or httpx.Client(timeout=self._config.request_timeout_s)

    def close(self) -> None:
        self._http.close()

    def _chat_completion(
        self, messages: list[dict[str, str]], temperature: float, provider_override: str | None = None
    ) -> tuple[str, str]:
        """Returns (content, provider_name_used). Retries once on the other
        provider if the primary fails entirely (network/5xx), so a single
        provider outage doesn't take Frisks down — logged either way so
        divergent provider behavior stays traceable."""
        primary = _resolve_provider(self._config, provider_override)
        fallback_name = "deepseek" if primary.name == "groq" else "groq"

        for attempt_provider in (primary, _resolve_provider(self._config, fallback_name)):
            if not attempt_provider.api_key:
                logger.warning("Skipping LLM provider %s: no API key configured", attempt_provider.name)
                continue
            try:
                content = self._call_provider(attempt_provider, messages, temperature)
                logger.info("LLM request served by provider=%s model=%s", attempt_provider.name, attempt_provider.model)
                return content, attempt_provider.name
            except Exception as exc:  # noqa: BLE001 - deliberately broad, we fall back
                logger.warning("LLM provider %s failed (%s); trying fallback", attempt_provider.name, exc)
                continue

        raise LLMError("All configured LLM providers failed or are unconfigured")

    def _call_provider(self, provider: ProviderSettings, messages: list[dict[str, str]], temperature: float) -> str:
        resp = self._http.post(
            f"{provider.base_url.rstrip('/')}/chat/completions",
            headers={"Authorization": f"Bearer {provider.api_key}", "Content-Type": "application/json"},
            json={"model": provider.model, "messages": messages, "temperature": temperature},
        )
        if resp.status_code != 200:
            raise LLMError(f"{provider.name} returned {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMError(f"{provider.name} response missing expected fields: {data}") from exc

    # -- public responsibilities --------------------------------------------

    def interpret_request(self, text: str) -> dict[str, Any]:
        """Natural language -> structured request dict (matching the locked API request schema's core fields)."""
        messages = [
            {"role": "system", "content": INTERPRET_SYSTEM_PROMPT},
            {"role": "user", "content": INTERPRET_USER_TEMPLATE.format(text=text)},
        ]
        content, provider = self._chat_completion(messages, temperature=0.0)
        cleaned = _strip_json_fence(content)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            # One provider (observed: DeepSeek is occasionally looser about
            # trailing commentary than Groq) may wrap JSON in prose despite
            # instructions. Try to salvage the first {...} block before giving up.
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                raise LLMError(f"Provider {provider} did not return parseable JSON: {content!r}") from exc
            parsed = json.loads(match.group(0))
        return parsed

    def explain_results(self, request_summary: str, results_json: str) -> str:
        messages = [
            {"role": "system", "content": EXPLAIN_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": EXPLAIN_USER_TEMPLATE.format(request_summary=request_summary, results_json=results_json),
            },
        ]
        content, _provider = self._chat_completion(messages, temperature=0.3)
        return content.strip()

    # -- LLM-as-orchestrator responsibilities (see frisks/service.py) -------
    # Both live inside StrategyHunterService, not a separate NL-only
    # endpoint, so they run on every request regardless of how it arrived.
    # Same rule as everywhere: these only narrate/reason over numbers the
    # engine already computed -- they never produce a number themselves.

    def synthesize_budget_note(self, request_summary: str, results_json: str) -> str:
        """
        Feature 1 (feasibility-aware re-querying). `results_json` must
        contain the top result from two REAL, separate engine runs (the
        caller's original budget and a relaxed budget) -- this method
        does not run the engine itself, it only narrates a comparison
        the caller (frisks/service.py) already computed.
        """
        messages = [
            {"role": "system", "content": BUDGET_NOTE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": BUDGET_NOTE_USER_TEMPLATE.format(
                    request_summary=request_summary, results_json=results_json
                ),
            },
        ]
        content, _provider = self._chat_completion(messages, temperature=0.3)
        return content.strip()

    def self_critique(self, request_summary: str, strategies_json: str) -> dict[str, str]:
        """
        Feature 2 (self-critique). Returns a dict mapping rank (as a
        string, e.g. "1") to a short caveat sentence to append to that
        strategy's rationale. A rank with no caveat needed is simply
        absent from the returned dict -- callers should treat a missing
        key as "nothing to add", not as an error.
        """
        messages = [
            {"role": "system", "content": SELF_CRITIQUE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": SELF_CRITIQUE_USER_TEMPLATE.format(
                    request_summary=request_summary, strategies_json=strategies_json
                ),
            },
        ]
        content, provider = self._chat_completion(messages, temperature=0.0)
        cleaned = _strip_json_fence(content)
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            match = re.search(r"\{.*\}", cleaned, re.DOTALL)
            if not match:
                raise LLMError(
                    f"Provider {provider} did not return parseable JSON for self-critique: {content!r}"
                ) from exc
            parsed = json.loads(match.group(0))
        if not isinstance(parsed, dict):
            raise LLMError(f"self-critique response was not a JSON object: {parsed!r}")
        return {str(k): str(v) for k, v in parsed.items()}
