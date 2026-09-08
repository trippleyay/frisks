# Frisks B402 Integration

Frisks includes an integration point for Binance's B402 machine-to-machine payment flow. The code is built and ready, but live payment handling is currently waiting on Binance partner account approval.

## Intended payment flow

A caller sends a normal Frisks request without payment.

Frisks detects that payment is required and returns HTTP `402 Payment Required` with the payment requirements needed for the request.

The caller uses those requirements to create and sign the payment authorization. The caller then resubmits the original request with the payment authorization attached.

Frisks verifies the authorization and settles the payment through Binance's B402 API. Once payment is accepted, the request continues through the normal Frisks service and the caller receives the strategy result.

The intended sequence is therefore:

```text
Caller
  |
  |  request without payment
  v
Frisks
  |
  |  HTTP 402 + payment requirements
  v
Caller
  |
  |  sign payment authorization
  |  resubmit request
  v
Frisks
  |
  |  verify + settle through Binance B402
  v
Frisks service
  |
  |  strategy result
  v
Caller
```

## Integration point

The B402 integration point is implemented in:

```text
frisks/b402/middleware.py
```

The middleware sits at the request boundary so payment can be required before the normal Frisks request handling proceeds. Once a valid payment authorization is verified and settled, the request is allowed to continue to the underlying service.

Keeping the payment check at this boundary means the strategy engine does not need to know whether a request was paid for. Pricing, market data, search, and ranking remain separate from payment handling.

## Current status

The integration is built and ready for live use. The remaining step is Binance partner account approval, which is required before the live B402 payment path can be enabled.

Until that access is available, the integration should be treated as an implemented payment boundary rather than an active requirement on the hosted Frisks service.
