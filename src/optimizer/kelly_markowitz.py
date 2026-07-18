import numpy as np
from scipy.optimize import minimize

# Fractional Kelly multiplier.
# Full Kelly (1.0) is optimal in theory but is extremely sensitive to model
# mis-calibration — a single overconfident estimate (e.g. 92.9% ETH) will
# concentrate the entire portfolio.  Half-Kelly (0.5) is the industry standard
# for prediction markets; it cuts drawdowns roughly in half while sacrificing
# only ~15% of long-run growth rate.
KELLY_FRACTION = 0.5


def optimize_portfolio(estimates: dict, market_prices: dict, correlation_matrix: np.ndarray, risk_level: int) -> dict:
    """
    Kelly-Markowitz portfolio optimizer for binary prediction markets.

    estimates: {"BTC": 0.70, "ETH": 0.52, "SOL": 0.60}  — our model's P(YES)
    market_prices: {"BTC": 0.55, "ETH": 0.40, "SOL": 0.50}  — market implied P(YES)
    correlation_matrix: NxN numpy array of return correlations
    risk_level: 1-10 (1=conservative/diversified, 10=aggressive/concentrated)

    Objective: maximise fractional-Kelly expected log-growth minus a Markowitz
    variance penalty.  The Markowitz term penalises putting large weights on
    correlated coins.  Risk level scales the variance penalty: high risk →
    small penalty (allow concentration), low risk → large penalty (force
    diversification away from correlated positions).

    The sum constraint is an inequality (≤ 1.0) so the optimizer may leave
    capital in cash when edges are compressed or model confidence is low.
    The undeployed fraction acts as a natural risk buffer.
    """
    coins = list(estimates.keys())
    n = len(coins)

    # Edge per coin
    edges = np.array([estimates[c] - market_prices[c] for c in coins])

    # Raw Kelly fraction per coin: f*_i = edge_i / (1 - market_price_i)
    # Scaled by KELLY_FRACTION (Half-Kelly by default) so the maximum each
    # coin can claim from the full bankroll is clipped well below 1.0.
    kelly_fracs = np.array([
        KELLY_FRACTION * (edges[i] / max(1.0 - market_prices[c], 1e-6))
        for i, c in enumerate(coins)
    ])
    kelly_fracs = np.clip(kelly_fracs, 0.0, 1.0)

    # Kelly expected log-growth per unit weight for each coin:
    #   g_i = p_i * log(1 + e_i) + (1 - p_i) * log(1 - e_i)
    # where e_i = edge_i (our edge over the market price)
    log_gains = np.array([
        estimates[c] * np.log(max(1.0 + edges[i], 1e-9))
        + (1.0 - estimates[c]) * np.log(max(1.0 - edges[i], 1e-9))
        for i, c in enumerate(coins)
    ])

    # Markowitz variance penalty strength:
    #   risk=1  → lambda=1.0  (strongly penalise correlated concentration)
    #   risk=5  → lambda=0.2  (moderate penalty)
    #   risk=10 → lambda=0.0  (pure Kelly, no diversification penalty)
    # Scale spans two orders of magnitude so the effect is visible across risk levels.
    lambda_var = 1.0 * ((10 - risk_level) / 9.0) ** 2  # 1.0 → 0.0

    def objective(weights):
        kelly_gain = np.dot(weights, log_gains)
        markowitz_penalty = lambda_var * (weights @ correlation_matrix @ weights)
        return -(kelly_gain - markowitz_penalty)  # minimise negative

    # Per-coin upper bound: the smaller of the fractional-Kelly fraction and
    # the risk-level concentration cap.  This means:
    #   • Low-edge markets naturally get a smaller budget even at high risk.
    #   • Low risk levels still force diversification via max_single.
    risk_frac = ((risk_level - 1) / 9.0) ** 0.7   # 0 → 1, convex
    max_single_risk = (1.0 / n) + risk_frac * (1.0 - 1.0 / n)

    # Per-coin bound = min(Half-Kelly fraction, risk-level cap)
    upper_bounds = np.minimum(kelly_fracs, max_single_risk)
    # Ensure upper bound is always strictly positive so the solver has room
    upper_bounds = np.maximum(upper_bounds, 1e-4)

    bounds = [(0.0, float(upper_bounds[i])) for i in range(n)]

    constraints = [
        # Capital deployed must not exceed 100%; may be less (cash buffer)
        {"type": "ineq", "fun": lambda w: 1.0 - w.sum()},
    ]

    # Warm start: Kelly-proportional, clipped to per-coin bounds
    x0 = np.copy(kelly_fracs)
    x0 = np.clip(x0, 0.0, upper_bounds)
    # Scale down if total exceeds 1 (will often happen; optimizer decides final split)
    if x0.sum() > 1.0:
        x0 = x0 / x0.sum()

    result = minimize(
        objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP",
        options={"ftol": 1e-10, "maxiter": 2000},
    )

    if not result.success:
        # Fallback: Kelly-proportional, each coin capped at its upper bound
        weights = np.copy(kelly_fracs)
        weights = np.clip(weights, 0.0, upper_bounds)
        if weights.sum() > 1.0:
            weights = weights / weights.sum()
    else:
        weights = np.maximum(result.x, 0.0)
        # Do NOT renormalise — the inequality constraint intentionally allows
        # sum < 1.0.  The remaining fraction stays as cash.

    allocations = {}
    for i, coin in enumerate(coins):
        allocations[coin] = {
            "weight": round(float(weights[i]), 4),
            "edge": round(float(edges[i]), 4),
            "our_estimate": estimates[coin],
            "market_price": market_prices[coin],
        }

    return allocations


if __name__ == "__main__":
    # test example
    estimates = {"BTC": 0.70, "ETH": 0.52, "SOL": 0.60}
    market_prices = {"BTC": 0.55, "ETH": 0.40, "SOL": 0.50}

    correlation_matrix = np.array([
        [1.00, 0.85, 0.70],
        [0.85, 1.00, 0.75],
        [0.70, 0.75, 1.00]
    ])

    for risk_level in [2, 5, 8]:
        print(f"\nrisk level {risk_level}:")
        result = optimize_portfolio(estimates, market_prices, correlation_matrix, risk_level)
        total = sum(d["weight"] for d in result.values())
        print(f"  total deployed: {total*100:.1f}%  (cash: {(1-total)*100:.1f}%)")
        for coin, data in result.items():
            print(f"  {coin}: {data['weight']*100:.1f}% | edge: {data['edge']:.2f}")
