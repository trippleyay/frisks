from frisks.engine import templates
from tests.conftest import build_chain, near_expiry_ms


def _chain():
    expiry = near_expiry_ms(30)
    strikes = [80, 90, 95, 100, 105, 110, 120]
    return build_chain("BTCUSDT", spot=100.0, strikes=strikes, expiry_ms=expiry), expiry


def test_vertical_spreads_have_two_legs_same_expiry():
    quotes, expiry = _chain()
    spreads = templates.vertical_spreads(quotes)
    assert spreads
    for c in spreads:
        assert len(c.legs) == 2
        assert c.expiries == {expiry}


def test_straddles_share_strike_different_option_side():
    quotes, _ = _chain()
    candidates = templates.straddles_and_strangles(quotes)
    straddles = [c for c in candidates if "straddle" in c.structure_name]
    assert straddles
    for c in straddles:
        strikes = {leg.strike for leg in c.legs}
        sides = {leg.option_side for leg in c.legs}
        assert len(strikes) == 1
        assert len(sides) == 2


def test_butterflies_have_symmetric_wings():
    quotes, _ = _chain()
    flies = templates.butterflies(quotes)
    assert flies
    for c in flies:
        strikes = sorted(leg.strike for leg in c.legs)
        assert len(strikes) == 3
        lo, mid, hi = strikes
        assert abs((mid - lo) - (hi - mid)) < 1e-6


def test_iron_condor_put_strikes_below_call_strikes():
    quotes, _ = _chain()
    condors = templates.iron_condors(quotes)
    assert condors
    for c in condors:
        from frisks.data.models import OptionSide

        put_strikes = [leg.strike for leg in c.legs if leg.option_side is OptionSide.PUT]
        call_strikes = [leg.strike for leg in c.legs if leg.option_side is OptionSide.CALL]
        assert max(put_strikes) < min(call_strikes)


def test_calendar_spreads_span_two_expiries():
    near_expiry = near_expiry_ms(7)
    far_expiry = near_expiry_ms(37)
    strikes = [90, 100, 110]
    near_quotes = build_chain("BTCUSDT", 100.0, strikes, near_expiry)
    far_quotes = build_chain("BTCUSDT", 100.0, strikes, far_expiry)
    cals = templates.calendar_spreads(near_quotes, far_quotes)
    assert cals
    for c in cals:
        assert c.expiries == {near_expiry, far_expiry}
        assert len(c.legs) == 2
