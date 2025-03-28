from portfolio_manager import PortfolioManager

# Define a list of stock tickers
tickers = ["AAPL", "MSFT", "GOOGL"]

# Initialize the PortfolioManager
pm = PortfolioManager(tickers)

# Compute Fama-French metrics for the assets
pm.compute_fama_french_metrics()

# Retrieve and display the minimum variance portfolio
min_var_portfolio = pm.get_min_variance_portfolio()
print("Minimum Variance Portfolio:")
print(min_var_portfolio)

# Retrieve and display the maximum return portfolio
max_return_portfolio = pm.get_max_return_portfolio()
print("Maximum Return Portfolio:")
print(max_return_portfolio)

# Retrieve and display the maximum Sharpe ratio portfolio
max_sharpe_portfolio = pm.get_max_sharpe_ratio_portfolio()
print("Maximum Sharpe Ratio Portfolio:")
print(max_sharpe_portfolio)

# Plot the efficient frontier
pm.plot_efficient_frontier()

# Plot asset weights (e.g., for the min variance portfolio)
pm.plot_asset_weights(min_var_portfolio)

# Backtest portfolio performance over the last 5 years using the min variance strategy
results_df, plt_obj = pm.backtest_portfolios(strategy="min_variance", backtest_years=5)
print(results_df)
