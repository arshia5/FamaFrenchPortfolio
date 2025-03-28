# Portfolio Manager with Fama-French Metrics

## Overview
This repository contains a Python-based Portfolio Manager that leverages historical stock data and the Fama-French Five Factor Model to analyze asset performance and optimize portfolio allocations. It offers tools to compute key metrics—such as annualized volatility and expected returns using regression analysis—and implements various portfolio optimization strategies:
- **Minimum Variance Portfolio:** Minimizes overall portfolio volatility.
- **Maximum Return Portfolio:** Maximizes the portfolio's expected return.
- **Maximum Sharpe Ratio Portfolio:** Maximizes risk-adjusted returns.

Additionally, the project provides visualization features to plot the efficient frontier and asset weight distributions, as well as functionality to backtest portfolio strategies against an equal-weight portfolio.

## Features
- **Data Acquisition:** Downloads historical stock prices via `yfinance` and fetches daily Fama-French 5-factor data.
- **Metric Computation:** Uses a regression-based approach to compute annualized volatility and expected returns for each asset.
- **Portfolio Optimization:** Implements optimization routines with no short-selling to:
  - Determine the minimum variance portfolio.
  - Identify the portfolio with maximum expected return.
  - Find the portfolio with the maximum Sharpe ratio.
- **Visualization:** 
  - Plots the efficient frontier with asset annotations.
  - Displays asset weights using pie charts.
- **Backtesting:** Compares the performance of strategy-based portfolios against an equal-weight benchmark over a specified period.

## Prerequisites
- **Python:** Version 3.7 or higher.
- **Libraries:** 
  - yfinance
  - pandas
  - numpy
  - statsmodels
  - matplotlib
  - tqdm
  - scipy

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/portfolio-manager.git
   cd portfolio-manager
