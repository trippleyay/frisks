from __future__ import annotations

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import JSONResponse

from frisks.api.schemas import (
    NaturalLanguageRequestSchema,
    NoStrategyResponseSchema,
    StrategyRequestSchema,
    StrategyResponseSchema,
)
from frisks.b402.middleware import PaymentRequiredError
from frisks.service import InvalidRequestError, NoValidStrategiesResult, StrategyHunterResult, StrategyHunterService

logger = logging.getLogger(__name__)
router = APIRouter()


def get_service(request: Request) -> StrategyHunterService:
    return request.app.state.service


def get_payment_gate(request: Request):
    return request.app.state.payment_gate


@router.post("/v1/strategies", response_model=None)
def search_strategies(
    payload: StrategyRequestSchema, request: Request, service: StrategyHunterService = Depends(get_service)
):
    """
    Core, LLM-free endpoint. Structured request in, ranked strategies out.
    This is the endpoint any calling agent should use once it already has
    structured constraints — it never touches Groq/DeepSeek.
    """
    gate = get_payment_gate(request)
    try:
        gate.authorize(dict(request.headers))
    except PaymentRequiredError as exc:
        return JSONResponse(status_code=402, content={"error": "payment_required", "details": exc.payment_details})

    try:
        result = service.handle_structured_request(payload.model_dump())
    except InvalidRequestError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except ValueError as exc:
        # e.g. unresolvable expiry, no matched strikes for price estimation
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    if isinstance(result, NoValidStrategiesResult):
        body = NoStrategyResponseSchema(request_id=result.request_id, asset=result.asset, message=result.message)
        return JSONResponse(status_code=200, content=json.loads(body.model_dump_json()))

    assert isinstance(result, StrategyHunterResult)
    body = StrategyResponseSchema(
        request_id=result.request_id,
        asset=result.asset,
        generated_at=result.generated_at,
        objective_used=result.objective_used,
        strategies=result.strategies,
        meta=result.meta,
    )
    return JSONResponse(status_code=200, content=json.loads(body.model_dump_json()))


@router.post("/v1/strategies/interpret")
def interpret_natural_language(payload: NaturalLanguageRequestSchema, service: StrategyHunterService = Depends(get_service)):
    """
    LLM responsibility #1. Converts free text into the structured request
    shape (does not run the engine) so a caller can review/edit before
    submitting to /v1/strategies.
    """
    try:
        structured = service.interpret_natural_language(payload.text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM interpretation failed")
        raise HTTPException(status_code=502, detail=f"LLM interpretation failed: {exc}") from exc
    return structured


@router.post("/v1/strategies/search-text")
def search_strategies_from_text(
    payload: NaturalLanguageRequestSchema, request: Request, service: StrategyHunterService = Depends(get_service)
):
    """Convenience endpoint: interpret then search in one call."""
    try:
        structured = service.interpret_natural_language(payload.text)
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM interpretation failed")
        raise HTTPException(status_code=502, detail=f"LLM interpretation failed: {exc}") from exc

    validated = StrategyRequestSchema.model_validate(structured)
    return search_strategies(validated, request, service)


@router.post("/v1/strategies/explain")
def explain_results(payload: dict, service: StrategyHunterService = Depends(get_service)):
    """
    LLM responsibility #2. Takes a previously-computed /v1/strategies
    response (ground truth) and returns a natural-language narration of
    it. Deliberately a separate, optional endpoint rather than baked into
    the locked response schema, so the core contract never depends on an
    LLM being configured or available.
    """
    request_summary = payload.get("request_summary", "")
    results = payload.get("results")
    if results is None:
        raise HTTPException(status_code=400, detail="Body must include 'results' (a /v1/strategies response)")
    try:
        explanation = service.explain(request_summary, json.dumps(results))
    except Exception as exc:  # noqa: BLE001
        logger.exception("LLM explanation failed")
        raise HTTPException(status_code=502, detail=f"LLM explanation failed: {exc}") from exc
    return {"explanation": explanation}


@router.get("/healthz")
def healthz():
    return {"status": "ok"}
