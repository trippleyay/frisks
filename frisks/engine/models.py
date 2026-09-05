from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

from frisks.data.models import MarketQuote, OptionSide


class OrderSide(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class Direction(str, Enum):
    BULLISH = "bullish"
    BEARISH = "bearish"
    NEUTRAL = "neutral"


class Objective(str, Enum):
    RISK_ADJUSTED_RETURN = "risk_adjusted_return"
    PROBABILITY_OF_PROFIT = "probability_of_profit"
    COST_FOR_TARGET = "cost_for_target"


@dataclass(frozen=True)
class Leg:
    side: OrderSide
    quote: MarketQuote
    quantity: float = 1.0

    @property
    def symbol(self) -> str:
        return self.quote.symbol

    @property
    def strike(self) -> float:
        return self.quote.contract.strike

    @property
    def option_side(self) -> OptionSide:
        return self.quote.contract.side

    @property
    def expiry_ms(self) -> int:
        return self.quote.contract.expiry_ms

    @property
    def signed_quantity(self) -> float:
        """Positive for long, negative for short."""
        return self.quantity if self.side is OrderSide.BUY else -self.quantity

    @property
    def entry_price(self) -> float:
        """
        Price paid/received to enter, using the appropriate side of the
        book: a buyer pays the ask, a seller receives the bid. Falls back
        to mark price if the relevant side of book is unavailable (should
        not happen for a leg that passed liquidity filtering, but keeps
        this robust rather than raising).
        """
        t = self.quote.ticker
        if self.side is OrderSide.BUY:
            return t.ask_price if t.ask_price > 0 else self.quote.mark.mark_price
        return t.bid_price if t.bid_price > 0 else self.quote.mark.mark_price

    def intrinsic_value(self, underlying_price: float) -> float:
        if self.option_side is OptionSide.CALL:
            return max(underlying_price - self.strike, 0.0)
        return max(self.strike - underlying_price, 0.0)

    def cost(self) -> float:
        """Net cash outflow to open this leg (negative = credit received)."""
        sign = 1 if self.side is OrderSide.BUY else -1
        return sign * self.entry_price * self.quantity * self.quote.contract.unit


@dataclass(frozen=True)
class Constraints:
    max_legs: int = 4
    max_expiries: int = 2


@dataclass(frozen=True)
class StrategyRequest:
    asset: str
    expiry_date: str | None
    days_out: int | None
    direction: Direction
    max_loss: float
    objective: Objective = Objective.RISK_ADJUSTED_RETURN
    target_cost: float | None = None
    constraints: Constraints = field(default_factory=Constraints)


@dataclass
class Candidate:
    """A fully-specified strategy under evaluation, before scoring."""

    legs: list[Leg]
    structure_name: str

    @property
    def expiries(self) -> set[int]:
        return {leg.expiry_ms for leg in self.legs}

    @property
    def entry_cost(self) -> float:
        """Net premium: positive = debit paid, negative = credit received."""
        return sum(leg.cost() for leg in self.legs)

    def net_greeks(self) -> dict[str, float]:
        totals = {"delta": 0.0, "theta": 0.0, "gamma": 0.0, "vega": 0.0}
        for leg in self.legs:
            g = leg.quote.mark
            unit = leg.quote.contract.unit
            totals["delta"] += leg.signed_quantity * g.delta * unit
            totals["theta"] += leg.signed_quantity * g.theta * unit
            totals["gamma"] += leg.signed_quantity * g.gamma * unit
            totals["vega"] += leg.signed_quantity * g.vega * unit
        return totals

    def liquidity_summary(self) -> dict[str, float]:
        vols = [leg.quote.ticker.volume_24h for leg in self.legs]
        spreads = [leg.quote.ticker.spread_pct for leg in self.legs if leg.quote.ticker.spread_pct is not None]
        ois = [leg.quote.open_interest.contracts for leg in self.legs]
        return {
            "min_volume_24h": min(vols) if vols else 0.0,
            "max_spread_pct": max(spreads) if spreads else 0.0,
            "open_interest": min(ois) if ois else 0.0,
        }

    def is_redundant_extension(self, new_leg: Leg) -> bool:
        """True if new_leg exactly inverts an existing leg (no-op per pruning rule 5)."""
        for leg in self.legs:
            if (
                leg.symbol == new_leg.symbol
                and leg.side != new_leg.side
                and leg.quantity == new_leg.quantity
            ):
                return True
        return False
