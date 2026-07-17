import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(estimates: dict, market_prices: dict, correlation_matrix: np.ndarray, risk_level: int) -> dict:
    """
    Kelly-Markowitz portfolio optimizer for binary prediction markets.

    estimates: {"BTC": 0.70, "ETH": 0.52, "SOL": 0.60}  — our model's P(YES)
    market_prices: {"BTC": 0.55, "ETH": 0.40, "SOL": 0.50}  — market implied P(YES)
    correlation_matrix: NxN numpy array of return correlations
    risk_level: 1-10 (1=conservative/diversified, 10=aggressive/concentrated)

    Objective: maximise Kelly expected log-growth minus a Markowitz variance penalty.
    The Markowitz term penalises putting large weights on correlated coins.
    Risk level scales the variance penalty: high risk → small penalty (allow concentration),
    low risk → large penalty (force diversification away from correlated positions).
    """
    coins = list(estimates.keys())
    n = len(coins)

    # Edge per coin
    edges = np.array([estimates[c] - market_prices[c] for c in coins])

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

    # Per-coin concentration cap based on risk level alone — independent of edge size
    #   risk=1  → max 1/n per coin (forced equal-weight)
    #   risk=10 → max 1.0 (unconstrained)
    # Quadratic ramp so mid-levels feel meaningfully different
    risk_frac = ((risk_level - 1) / 9.0) ** 0.7   # 0 → 1, convex
    max_single = (1.0 / n) + risk_frac * (1.0 - 1.0 / n)

    bounds = [(0.0, max_single) for _ in range(n)]

    constraints = [
        # All capital deployed
        {"type": "eq", "fun": lambda w: w.sum() - 1.0},
    ]

    # Warm start: edge-proportional, clipped to bounds
    x0 = np.maximum(edges, 1e-9)
    x0 = x0 / x0.sum()
    x0 = np.clip(x0, 0.0, max_single)
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
        # Fallback: edge-proportional, capped at max_single
        pos_edges = np.maximum(edges, 1e-9)
        weights = pos_edges / pos_edges.sum()
        weights = np.clip(weights, 0.0, max_single)
        weights = weights / weights.sum()
    else:
        weights = np.maximum(result.x, 0.0)
        s = weights.sum()
        weights = weights / s if s > 1e-6 else np.ones(n) / n

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
        for coin, data in result.items():
            print(f"  {coin}: {data['weight']*100:.1f}% | edge: {data['edge']:.2f}")