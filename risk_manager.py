import numpy as np

def realized_vol(returns, window=20):
    return returns.rolling(window).std() * np.sqrt(252)

def historical_var(returns, alpha=0.05):
    return np.percentile(returns, alpha * 100)

def max_drawdown(equity_curve):

    peak = equity_curve.cummax()
    drawdown = equity_curve / peak - 1

    return drawdown.min()