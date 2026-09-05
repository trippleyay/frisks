"""
B402 payment gate — STUB.

Open item, explicitly called out in the build prompt: "B402 integration
specifics (request/response format for the payment gate) — not yet
researched in implementation detail." We have not verified Binance's
actual B402 request/response format against their docs, so we are not
guessing at wire-level details here.

What's implemented: the *shape* of the gate as an interchangeable
dependency, feature-flagged off by default (FRISKS_B402_ENABLED=false),
so the rest of the codebase (API layer, engine) is already wired for
payment-gating without depending on unverified B402 specifics. When B402
integration work happens, only PaymentGate.authorize needs a real
implementation — nothing else in the request path changes.

Flow, per spec: calling agent -> Frisks endpoint -> payment required ->
agent pays via B402 -> Frisks processes request -> result returned. B402
is the payment/merchant layer, never the strategy engine.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class PaymentRequiredError(Exception):
    """Raised by PaymentGate.authorize when a request must be paid for before proceeding."""

    def __init__(self, message: str, payment_details: dict | None = None) -> None:
        super().__init__(message)
        self.payment_details = payment_details or {}


@dataclass
class PaymentGate:
    enabled: bool = False

    def authorize(self, headers: dict[str, str]) -> None:
        """
        Raise PaymentRequiredError if the request isn't authorized to
        proceed. No-ops entirely when disabled (MVP default — see spec's
        "Account requirements — MVP": account-free, no execution).

        TODO (flagged, not silently resolved): once B402's actual
        request/response contract is verified against Binance's docs,
        implement real verification here (e.g. checking a payment proof
        header/token against B402's settlement API). Until then, enabling
        this flag will reject all requests, by design — we'd rather fail
        loudly than fake a payment check.
        """
        if not self.enabled:
            return
        raise PaymentRequiredError(
            "B402 payment gating is enabled but not yet implemented — integration details unverified.",
            payment_details={"status": "not_implemented"},
        )
