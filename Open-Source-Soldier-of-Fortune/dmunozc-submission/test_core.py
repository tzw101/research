"""Unit tests for core.py"""


import pytest
import numpy as np
import yfinance as yf

@pytest.fixture
def few_stocks():
    return ["MSFT", "DB"]

@pytest.fixture
def period():
    return "1y"

@pytest.fixture
def few_prices():
    return np.array([[9.78, 9.90, 9.82, 9.85, 9.72], [25.48, 27.45, 26.97, 26.61, 26.41]])

def test_yfinance_download(few_stocks, period):
    tickers = yf.Tickers(" ".join(few_stocks))
    df = tickers.history(period=period)["Close"]
    assert len(df) > 0


def test_few_returns_dist(few_prices):
    N = len(few_prices)
    t = len(few_prices[0])
    log_returns = []
    for price in few_prices:
        log_ret = np.diff(np.log(price))
        log_returns.append(log_ret)





