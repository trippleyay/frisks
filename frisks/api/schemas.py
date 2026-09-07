"""
Pydantic models mirroring the spec's locked API contract v1 field-for-field.
Do not rename or restructure these without bumping schema_version and
flagging the change — per spec, this shape is locked.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = "1.0"


class Horizon(BaseModel):
    expiry_date: str | None = None
    days_out: int | None = None

    @model_validator(mode="after")
    def exactly_one(self) -> "Horizon":
        if (self.expiry_date is None) == (self.days_out is None):
            raise ValueError("horizon must specify exactly one of expiry_date or days_out")
        return self


class Constraints(BaseModel):
    max_legs: int = 4
    max_expiries: int = 2


class StrategyRequestSchema(BaseModel):
    schema_version: str = SCHEMA_VERSION
    asset: Literal["BTC", "ETH"]
    horizon: Horizon
    direction: Literal["bullish", "bearish", "neutral"]
    max_loss: float = Field(gt=0)
    objective: Literal["risk_adjusted_return", "probability_of_profit", "cost_for_target"] = (
        "risk_adjusted_return"
    )
    target_cost: float | None = None
    constraints: Constraints = Field(default_factory=Constraints)

    @model_validator(mode="after")
    def target_cost_requires_objective(self) -> "StrategyRequestSchema":
        if self.objective == "cost_for_target" and self.target_cost is None:
            raise ValueError("target_cost is required when objective is 'cost_for_target'")
        if self.objective != "cost_for_target" and self.target_cost is not None:
            raise ValueError("target_cost must be null unless objective is 'cost_for_target'")
        return self


class NaturalLanguageRequestSchema(BaseModel):
    """Alternate entry point: free text, interpreted by the LLM layer into a StrategyRequestSchema."""

    text: str


class LegSchema(BaseModel):
    side: Literal["BUY", "SELL"]
    symbol: str
    strike: float
    expiry: str
    quantity: float


class GreeksSchema(BaseModel):
    delta: float
    gamma: float
    theta: float
    vega: float


class LiquiditySchema(BaseModel):
    min_volume_24h: float
    max_spread_pct: float
    open_interest: float


class StrategyResultSchema(BaseModel):
    rank: int
    structure_name: str
    legs: list[LegSchema]
    entry_cost: float
    max_profit: float
    max_loss: float
    breakeven: list[float]
    probability_of_profit: float
    greeks: GreeksSchema
    liquidity: LiquiditySchema
    objective_score: float
    rationale: str


class MetaSchema(BaseModel):
    candidates_evaluated: int
    candidates_excluded_illiquid: int
    # Additive field (LLM-orchestration build): candidates that were
    # structurally valid but excluded specifically for exceeding max_loss
    # -- distinct from candidates_excluded_illiquid (excluded upstream in
    # the data layer, before candidate generation even runs). Defaults to
    # 0 so any client validating against the previous shape still passes.
    candidates_excluded_budget: int = 0
    constraints_applied: Constraints


class StrategyResponseSchema(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    asset: str
    generated_at: str
    objective_used: str
    strategies: list[StrategyResultSchema]
    meta: MetaSchema
    # Additive, optional (LLM-orchestration build): populated only when
    # the feasibility-aware re-query triggers (see StrategyHunterService).
    # Most well-budgeted requests leave this null -- that's the correct,
    # expected case, not a missed feature. Never present without a real
    # second engine run backing it; the LLM only narrates numbers both
    # runs actually produced.
    budget_note: str | None = None


class NoStrategyResponseSchema(BaseModel):
    schema_version: str = SCHEMA_VERSION
    request_id: str
    asset: str
    strategies: list = Field(default_factory=list)
    error: Literal["no_valid_strategies"] = "no_valid_strategies"
    message: str
