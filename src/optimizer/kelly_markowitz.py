import numpy as np
from scipy.optimize import minimize

def optimize_portfolio(estimates: dict, market_prices: dict, correlation_matrix: np.ndarray, risk_level: int) -> dict:
    """
    estimates: {"BTC": 0.70, "ETH": 0.52, "SOL": 0.60}
    market_prices: {"BTC": 0.55, "ETH": 0.40, "SOL": 0.50}
    correlation_matrix: 3x3 numpy array
    risk_level: 1-10 (1=conservative, 10=aggressive)
    """
    coins = list(estimates.keys())
    n = len(coins)

    # edge = our estimate minus market price
    edges = np.array([
        estimates[c] - market_prices[c]
        for c in coins
    ])

    # max variance allowed — scaled by risk level
    max_variance = 0.01 * risk_level  # 1% to 10%

    # kelly objective — maximize sum of w * log(1 + edge)
    def kelly_objective(weights):
        gains = np.log(1 + weights * edges + 1e-9)
        return -np.sum(gains)  # negative because scipy minimizes

    # markowitz risk constraint
    def risk_constraint(weights):
        portfolio_variance = weights @ correlation_matrix @ weights
        return max_variance - portfolio_variance

    constraints = [
        {"type": "ineq", "fun": risk_constraint},
        {"type": "ineq", "fun": lambda w: 1 - w.sum()},
    ]

    bounds = [(0, 1) for _ in range(n)]
    x0 = np.array([1/n] * n)

    result = minimize(
        kelly_objective,
        x0,
        bounds=bounds,
        constraints=constraints,
        method="SLSQP"
    )

    allocations = {}
    for i, coin in enumerate(coins):
        edge = estimates[coin] - market_prices[coin]
        allocations[coin] = {
            "weight": round(max(result.x[i], 0), 4),
            "edge": round(edge, 4),
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