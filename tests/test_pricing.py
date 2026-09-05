import math

from frisks.data.models import OptionSide
from frisks.engine.pricing import bs_price, build_iv_smile, build_risk_neutral_distribution


def test_put_call_parity():
    S, K, T, r, sigma = 100.0, 100.0, 0.25, 0.02, 0.6
    call = bs_price(S, K, T, r, sigma, OptionSide.CALL)
    put = bs_price(S, K, T, r, sigma, OptionSide.PUT)
    # C - P = S - K*exp(-rT)
    lhs = call - put
    rhs = S - K * math.exp(-r * T)
    assert math.isclose(lhs, rhs, abs_tol=1e-6)


def test_call_price_bounds():
    S, K, T, r, sigma = 100.0, 90.0, 0.5, 0.0, 0.8
    call = bs_price(S, K, T, r, sigma, OptionSide.CALL)
    intrinsic = max(S - K, 0.0)
    assert call >= intrinsic - 1e-9
    assert call <= S


def test_zero_time_falls_back_to_intrinsic():
    assert bs_price(110, 100, 0.0, 0.0, 0.5, OptionSide.CALL) == 10.0
    assert bs_price(90, 100, 0.0, 0.0, 0.5, OptionSide.PUT) == 10.0


def test_distribution_sums_to_one_and_centers_near_forward():
    S, T, r = 100.0, 0.25, 0.0
    smile = build_iv_smile({80.0: 0.7, 90.0: 0.62, 100.0: 0.6, 110.0: 0.62, 120.0: 0.7})
    dist = build_risk_neutral_distribution(S, T, r, smile, grid_points=300, grid_sigmas=5.0)

    total = sum(dist.probabilities)
    assert math.isclose(total, 1.0, abs_tol=1e-6)

    mean = sum(p * prob for p, prob in zip(dist.prices, dist.probabilities))
    forward = S * math.exp(r * T)
    # A symmetric smile around spot should produce a mean close to the forward price.
    assert abs(mean - forward) / forward < 0.05


def test_distribution_probability_and_ev_helpers():
    S, T, r = 100.0, 0.1, 0.0
    smile = build_iv_smile({90.0: 0.5, 100.0: 0.5, 110.0: 0.5})
    dist = build_risk_neutral_distribution(S, T, r, smile, grid_points=200)

    prob_above_spot = dist.probability(lambda p: p > S)
    assert 0.0 < prob_above_spot < 1.0

    ev_of_price = dist.expected_value(lambda p: p)
    assert ev_of_price > 0
