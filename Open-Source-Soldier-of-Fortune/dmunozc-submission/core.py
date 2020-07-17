"""This file implements MST on a distance metric for a number of stocks.

It follows the methedology as seen in https://arxiv.org/abs/1703.00485
It calculates a distance metric based on multiple assets log returns. Based
on this metric, a minimum spanning tree (MST) is created.
"""

import yfinance as yf
import random
import numpy as np
from graph import Graph

ASSETS = ["MSFT", "FB", "AAPL", "NFLX", "GOOG", "SPY", "IWM"]
random.seed(69)


def asset_prices(N, start_date="2009-01-01", end_date="2010-01-01"):
    assets = random.sample(ASSETS, N)
    return yf.download(
        tickers=" ".join(assets), start=start_date, end=end_date
    )


def main():
    assets = asset_prices(3)
    close_prices = assets["Close"]
    # log(5) - log(2) == log(5/2)
    log_returns = np.log(close_prices / close_prices.shift(1))[1:]
    correlations = log_returns.corr()
    distances = np.sqrt(2 * (1 - correlations))
    # Correlation and distance matrices are symmetrical. Can ignore lower
    # diagonal.
    graph = Graph(distances)
    mst = graph.mst()

main()
