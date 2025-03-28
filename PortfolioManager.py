import yfinance as yf
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy.optimize import minimize


class PortfolioManager:
    def __init__(self, tickers):
        """
        Initialize the PortfolioManager with a list of tickers.
        """
        self.tickers = tickers
        self.start_date = None  # Will be set once stock data is fetched

        # Attributes to hold the investor profile data
        self.risk_tolerance_score = None
        self.investor_profile = None

        # Fetch data for daily returns of tickers and Fama-French 5-factors
        self.daily_returns = self._download_stock_data()  # daily % changes
        self.ff5_data = self._download_ff5_factors()

        # Merge them into a single DataFrame
        self.merged_df = self._merge_dataframes_on_index(
            self.daily_returns,
            self.ff5_data
        )

        # This will hold the per-asset Fama-French metrics after calculation
        self.asset_stats = None


    def _download_stock_data(self):
        """
        Download historical adjusted closing prices for the given tickers using yfinance,
        compute daily returns, and return them as a DataFrame (date as index,
        columns are tickers).
        """
        stock_prices = yf.download(
            self.tickers,
            interval='1d',
            auto_adjust=False,
        )['Adj Close']

        # Drop rows with NaNs and compute daily percentage returns.
        stock_returns = stock_prices.dropna().pct_change()

        # Set the earliest date for aligning factor data.
        self.start_date = stock_returns.index.min().date()

        return stock_returns

    def _download_ff5_factors(self):
        """
        Download daily Fama-French 5-Factors (including the risk-free rate) from Ken French's website,
        clean them, and return a DataFrame with date as index.
        """
        ff5_url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"

        ff5_df = pd.read_csv(
            ff5_url,
            compression="zip",
            skiprows=3
        )

        ff5_df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)
        ff5_df.dropna(inplace=True)

        ff5_df['Date'] = ff5_df['Date'].astype(str).str.strip()
        ff5_df = ff5_df[ff5_df['Date'].str.len() > 6]  # Remove footer rows.

        ff5_df['Date'] = pd.to_datetime(ff5_df['Date'], format='%Y%m%d')
        ff5_df.set_index('Date', inplace=True)

        # Convert factor data from percentages to decimals.
        ff5_df = ff5_df.astype(float) / 100.0

        if self.start_date:
            ff5_df = ff5_df.loc[ff5_df.index >= pd.Timestamp(self.start_date)]

        return ff5_df

    def _merge_dataframes_on_index(self, *dataframes):
        """
        Merge multiple DataFrames on their date index using an inner join.
        """
        if not dataframes:
            raise ValueError("No DataFrames provided for merging.")

        merged = dataframes[0]
        for df in dataframes[1:]:
            merged = merged.join(df, how='inner')

        return merged

    def compute_fama_french_metrics(self):
        """
        Calculate per-ticker annualized volatility and expected returns using
        a Fama-French (up to 5-factor) regression model.
        """
        combined_df = self.merged_df.copy()

        possible_factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA']
        factor_columns = [col for col in combined_df.columns if col in possible_factors]

        if 'RF' not in combined_df.columns:
            raise ValueError("The merged DataFrame must have a column named 'RF' (risk-free rate).")

        excluded_cols = set(factor_columns + ['RF'])
        ticker_columns = [col for col in combined_df.columns if col not in excluded_cols]

        metrics_records = []
        for ticker in ticker_columns:
            regression_data = combined_df[[ticker, 'RF'] + factor_columns].dropna()
            if len(regression_data) < (len(factor_columns) + 1):
                continue

            regression_data['ExRet'] = regression_data[ticker] - regression_data['RF']
            X = sm.add_constant(regression_data[factor_columns])
            y = regression_data['ExRet']
            ff_model = sm.OLS(y, X).fit()

            daily_volatility = regression_data[ticker].std()
            factor_means = combined_df[factor_columns].mean()
            rf_mean = combined_df['RF'].mean()

            alpha = ff_model.params['const']
            betas = ff_model.params.drop('const', errors='ignore')
            expected_excess_ret = alpha + (betas * factor_means).sum()
            daily_expected_return = expected_excess_ret + rf_mean
            annualized_volatility = daily_volatility * np.sqrt(252)
            annualized_return = (1 + daily_expected_return) ** 252 - 1

            metrics_records.append({
                'Ticker': ticker,
                'Volatility': annualized_volatility,
                'Expected_Return': annualized_return
            })

        self.asset_stats = pd.DataFrame(metrics_records).set_index('Ticker')

    def get_min_variance_portfolio(self):
        """
        Find the portfolio with the minimum variance (Markowitz, no short selling),
        with weights summing to 100% and displayed to two decimal places.

        Returns:
        --------
        pd.DataFrame
            A single-row DataFrame with:
            - 'Portfolio_Volatility': The annualized volatility of the min-variance portfolio.
            - 'Portfolio_Return': The annualized return of this portfolio (based on expected returns).
            - '[Ticker]_Weight': The percentage weight of each ticker (rounded to 2 decimals).
        """
        if self.asset_stats is None:
            raise ValueError("Please run compute_fama_french_metrics() first.")

        metrics_df = self.asset_stats
        returns_df = self.daily_returns

        # Get overlap of tickers in both DataFrames
        valid_tickers = metrics_df.index.intersection(returns_df.columns)
        if len(valid_tickers) == 0:
            raise ValueError("No common tickers found between asset_stats and daily_returns!")

        # Filter both DataFrames to only include valid_tickers
        metrics_df = metrics_df.loc[valid_tickers]
        returns_df = returns_df[valid_tickers]

        # Extract expected returns and compute annual covariance
        expected_returns = metrics_df['Expected_Return'].values
        cov_matrix = returns_df.cov() * 252
        n_assets = len(valid_tickers)

        # Objective function: portfolio variance = w^T * COV * w
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights

        # Constraint: sum of weights = 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        # Bounds: no short selling => weights in [0, 1]
        bounds = [(0.0, 1.0) for _ in range(n_assets)]

        # Initial guess: equally distributed
        init_weights = np.ones(n_assets) / n_assets

        # Minimize portfolio variance subject to constraints
        result = minimize(
            portfolio_variance,
            x0=init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        # Compute final portfolio variance, volatility, and return
        min_var = portfolio_variance(optimal_weights)
        portfolio_volatility = np.sqrt(min_var)
        portfolio_return = np.dot(optimal_weights, expected_returns)

        # Build a result DataFrame
        data = {
            'Portfolio_Volatility': [round(portfolio_volatility, 3)],
            'Portfolio_Return': [round(portfolio_return, 3)]
        }
        # Convert weights to percentages and round to 2 decimals
        for i, ticker in enumerate(valid_tickers):
            data[f"{ticker}_Weight"] = [round(optimal_weights[i] * 100, 2)]

        return pd.DataFrame(data)

    def get_max_return_portfolio(self):
        """
        Find the portfolio with the maximum expected return, with no short selling,
        with weights summing to 100% (1.0) and displayed to two decimal places.

        Returns:
        --------
        pd.DataFrame
            A single-row DataFrame with:
            - 'Portfolio_Volatility': The annualized volatility of the max return portfolio.
            - 'Portfolio_Return': The annualized return of this portfolio.
            - '[Ticker]_Weight': The percentage weight of each ticker (rounded to 2 decimals).
        """
        if self.asset_stats is None:
            raise ValueError("Please run compute_fama_french_metrics() first.")

        metrics_df = self.asset_stats
        returns_df = self.daily_returns

        # Get the overlapping tickers in both DataFrames.
        valid_tickers = metrics_df.index.intersection(returns_df.columns)
        if len(valid_tickers) == 0:
            raise ValueError("No common tickers found between asset_stats and daily_returns!")

        # Filter both DataFrames to only include valid_tickers.
        metrics_df = metrics_df.loc[valid_tickers]
        returns_df = returns_df[valid_tickers]

        # Extract the expected returns and compute the annualized covariance.
        expected_returns = metrics_df['Expected_Return'].values
        cov_matrix = returns_df.cov() * 252  # Annualize the covariance.
        n_assets = len(valid_tickers)

        # Define the objective function: negative portfolio return (so that minimizing it maximizes return)
        def negative_portfolio_return(weights):
            return -np.dot(weights, expected_returns)

        # Constraint: sum of weights must equal 1.
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        # Bounds: no short selling (weights between 0 and 1).
        bounds = [(0.0, 1.0) for _ in range(n_assets)]

        # Initial guess: equal weights.
        init_weights = np.ones(n_assets) / n_assets

        # Optimize to maximize the return by minimizing its negative.
        result = minimize(
            negative_portfolio_return,
            x0=init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        optimal_weights = result.x
        # Calculate the portfolio return and volatility.
        portfolio_return = np.dot(optimal_weights, expected_returns)
        portfolio_volatility = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)

        # Build the output DataFrame.
        data = {
            'Portfolio_Volatility': [portfolio_volatility],
            'Portfolio_Return': [portfolio_return]
        }
        for i, ticker in enumerate(valid_tickers):
            data[f"{ticker}_Weight"] = [round(optimal_weights[i] * 100, 2)]

        return pd.DataFrame(data)

    def get_max_sharpe_ratio_portfolio(self, risk_free_rate=0.0):
        """
        Find the portfolio with the maximum Sharpe ratio (no short selling),
        with weights summing to 100% and displayed to two decimal places.

        Parameters:
        -----------
        risk_free_rate : float
            The risk-free rate to use in the Sharpe ratio calculation.

        Returns:
        --------
        pd.DataFrame
            A single-row DataFrame with:
            - 'Portfolio_Volatility': The annualized volatility of the max Sharpe portfolio.
            - 'Portfolio_Return': The annualized return of this portfolio (based on expected returns).
            - 'Sharpe_Ratio': The maximum Sharpe ratio.
            - '[Ticker]_Weight': The percentage weight of each ticker (rounded to 2 decimals).
        """
        if self.asset_stats is None:
            raise ValueError("Please run compute_fama_french_metrics() first.")

        metrics_df = self.asset_stats
        returns_df = self.daily_returns

        # Get overlap of tickers
        valid_tickers = metrics_df.index.intersection(returns_df.columns)
        if len(valid_tickers) == 0:
            raise ValueError("No common tickers found between asset_stats and daily_returns!")

        metrics_df = metrics_df.loc[valid_tickers]
        returns_df = returns_df[valid_tickers]

        expected_returns = metrics_df['Expected_Return'].values
        cov_matrix = returns_df.cov() * 252  # Annualize
        n_assets = len(valid_tickers)

        # Objective function: negative Sharpe ratio
        def negative_sharpe_ratio(weights):
            port_return = np.dot(weights, expected_returns)
            port_volatility = np.sqrt(weights.T @ cov_matrix @ weights)
            return -(port_return - risk_free_rate) / port_volatility

        # Constraint: sum of weights = 1
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})

        # Bounds: no short selling => weights in [0, 1]
        bounds = [(0.0, 1.0) for _ in range(n_assets)]

        # Initial guess: equally distributed
        init_weights = np.ones(n_assets) / n_assets

        # Minimize negative Sharpe ratio
        result = minimize(
            negative_sharpe_ratio,
            x0=init_weights,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        # Extract optimal continuous weights
        optimal_weights = result.x

        port_return = np.dot(optimal_weights, expected_returns)
        port_volatility = np.sqrt(optimal_weights.T @ cov_matrix @ optimal_weights)
        sharpe_ratio = (port_return - risk_free_rate) / port_volatility

        # Build a result DataFrame
        data = {
            'Portfolio_Volatility': [round(port_volatility, 3)],
            'Portfolio_Return': [round(port_return, 3)],
            'Sharpe_Ratio': [round(sharpe_ratio, 3)]
        }

        # Convert weights to percentages and round to 2 decimals
        # Note: This will not force the sum to remain EXACTLY 100.00 after rounding,
        # but is the usual approach in portfolio reporting.
        for i, ticker in enumerate(valid_tickers):
            data[f"{ticker}_Weight"] = [round(optimal_weights[i] * 100, 2)]

        return pd.DataFrame(data)

    def plot_efficient_frontier(self, num_points=1000, risk_free_rate=0.0):
        """
        Plots the efficient frontier for the portfolio with asset names annotated.

        The method performs the following steps:
          1. Checks that asset metrics have been computed.
          2. Identifies tickers common to both asset_stats and daily_returns.
          3. Extracts annualized expected returns and computes the annualized covariance matrix.
          4. Uses the minimum-variance portfolio as the low endpoint (in terms of expected return)
             and the maximum individual asset expected return as the high endpoint.
          5. For a grid of target returns, it solves the optimization problem to find the
             minimum-variance portfolio meeting the target.
          6. Plots the efficient frontier as a solid line, marks key portfolios, and annotates
             the individual assets with their ticker names.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from tqdm import tqdm
        from scipy.optimize import minimize

        # Ensure asset_stats have been computed.
        if self.asset_stats is None:
            raise ValueError("Please run compute_fama_french_metrics() first.")

        # Identify tickers common to asset_stats and daily_returns.
        valid_tickers = list(self.asset_stats.index.intersection(self.daily_returns.columns))
        if len(valid_tickers) == 0:
            raise ValueError("No common tickers found between asset_stats and daily_returns!")

        # Extract annualized expected returns from asset_stats.
        exp_returns = self.asset_stats.loc[valid_tickers, "Expected_Return"].values

        # Compute the annualized covariance matrix.
        cov_matrix = self.daily_returns[valid_tickers].cov() * 252
        n_assets = len(valid_tickers)

        # Obtain the minimum-variance portfolio.
        min_var_df = self.get_min_variance_portfolio()
        sigma_min = float(min_var_df.loc[0, "Portfolio_Volatility"])
        R_min = float(min_var_df.loc[0, "Portfolio_Return"])

        # For the upper bound on target returns, we use the highest expected return among assets.
        R_max = np.max(exp_returns)

        # Create a grid of target returns between R_min and R_max.
        target_returns = np.linspace(R_min, R_max, num_points)

        frontier_vols = []
        frontier_rets = []

        # Define the portfolio variance function.
        def portfolio_variance(weights):
            return weights.T @ cov_matrix @ weights

        # For each target return, solve the optimization problem.
        for target in tqdm(target_returns, desc="Plotting Efficient Frontier"):
            # Objective: minimize portfolio variance.
            def objective(weights):
                return portfolio_variance(weights)

            # Constraints: weights sum to 1 and portfolio return equals target.
            constraints = (
                {"type": "eq", "fun": lambda w: np.sum(w) - 1},
                {"type": "eq", "fun": lambda w: np.dot(w, exp_returns) - target}
            )

            # No short selling: weights between 0 and 1.
            bounds = [(0.0, 1.0) for _ in range(n_assets)]

            init_guess = np.ones(n_assets) / n_assets
            res = minimize(objective, x0=init_guess, method="SLSQP",
                           bounds=bounds, constraints=constraints)

            if res.success:
                w_opt = res.x
                port_var = portfolio_variance(w_opt)
                port_vol = np.sqrt(port_var)
                port_ret = np.dot(w_opt, exp_returns)
                frontier_vols.append(port_vol)
                frontier_rets.append(port_ret)
            else:
                # If the optimization fails for a target, skip this point.
                continue

        # Begin plotting.
        plt.figure(figsize=(11, 6))

        # Plot the efficient frontier as a blue dashed line.
        plt.plot(frontier_vols, frontier_rets, "b--", lw=2, label="Efficient Frontier")

        # Plot individual asset risk/return points.
        asset_vols = self.asset_stats.loc[valid_tickers, "Volatility"].values
        asset_returns = self.asset_stats.loc[valid_tickers, "Expected_Return"].values
        plt.scatter(asset_vols, asset_returns, c="red", marker="3", s=100, label="Assets")

        # Annotate each asset with its ticker name.
        for ticker, vol, ret in zip(valid_tickers, asset_vols, asset_returns):
            plt.text(vol, ret, f" {ticker}", fontsize=10, ha='left', va='center')

        # Mark the max Sharpe ratio portfolio.
        max_sharpe_df = self.get_max_sharpe_ratio_portfolio(risk_free_rate=risk_free_rate)
        sigma_sh = float(max_sharpe_df.loc[0, "Portfolio_Volatility"])
        R_sh = float(max_sharpe_df.loc[0, "Portfolio_Return"])
        plt.scatter(sigma_sh, R_sh, marker="|", color="green", s=150, label="Max Sharpe Portfolio")
        plt.text(sigma_sh, R_sh, " Tangency Portfolio", fontsize=10, ha='left', va='center')

        # Set labels and title.
        plt.xlabel("Annualized Volatility")
        plt.ylabel("Annualized Expected Return")
        plt.title("Efficient Frontier")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # --- Adjust axis limits to include all points ---
        # Combine all volatilities and returns from frontier, assets, and key portfolios.
        all_vols = np.array(frontier_vols + list(asset_vols) + [sigma_min, sigma_sh])
        all_rets = np.array(frontier_rets + list(asset_returns) + [R_min, R_sh])

        # Set the limits with a margin.
        plt.xlim([all_vols.min() * 0.95, all_vols.max() * 1.05])
        plt.ylim([all_rets.min() * 0.95, all_rets.max() * 1.05])
        # -------------------------------------------------

        plt.show()
        return plt

    def plot_asset_weights(self, df, min_threshold=2.0, title="Asset Weights"):
        """
        Draws a circular (pie) plot showing the weights of the assets.

        The function:
          - Excludes assets with zero weight.
          - Groups assets with a weight below `min_threshold` (in percent)
            into an "Other" category.

        Parameters
        ----------
        df : pd.DataFrame
            A DataFrame containing asset weight columns. For example, a DataFrame
            returned by one of your portfolio methods, where the asset weight
            columns have names like 'AAPL_Weight', 'MSFT_Weight', etc.

        min_threshold : float, optional (default=2.0)
            The minimum percentage weight an asset must have to be shown individually.
            Any assets with weights below this value are grouped into an "Other" slice.

        Raises
        ------
        ValueError
            If no columns ending with '_Weight' are found.
        """
        # Identify columns that represent asset weights.
        weight_cols = [col for col in df.columns if col.endswith('_Weight')]
        if not weight_cols:
            raise ValueError("No asset weight columns found. Expecting columns ending with '_Weight'.")

        # Extract the weights from the first row (assuming a single-row DataFrame) and remove zeros.
        weights = df[weight_cols].iloc[0]
        weights = weights[weights != 0]

        # Create labels by removing the '_Weight' suffix.
        labels = [col.replace('_Weight', '') for col in weights.index]
        weights.index = labels  # Use the cleaned labels as the Series index.

        # Separate assets into "large" and "small" based on the min_threshold.
        large_weights = weights[weights >= min_threshold]
        small_weights = weights[weights < min_threshold]

        # Combine small weights into a single "Other" category if necessary.
        final_weights = large_weights.copy()
        if small_weights.sum() > 0:
            final_weights["Other"] = small_weights.sum()

        # Plot a pie chart.
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.pie(final_weights.values, labels=final_weights.index, autopct='%1.1f%%',
               startangle=90, counterclock=False)
        ax.axis('equal')  # Ensures the pie is drawn as a circle.
        plt.title(title)
        plt.show()
        return plt

    def backtest_portfolios(self, strategy="min_variance", initial_investment=1.0, backtest_years=5):
        """
        Backtests a buy-and-hold strategy for the past `backtest_years` years using either the
        minimum-variance portfolio weights or the maximum Sharpe ratio weights, and compares it
        to an equal-weight portfolio.

        Parameters
        ----------
        strategy : str, optional
            The optimization strategy to use. Must be either "min_variance" or "max_sharpe".
            (Default is "min_variance".)
        initial_investment : float, optional
            The amount (or index value) to start with. (Default is 1.0.)
        backtest_years : int, optional
            The number of years to backtest from the most recent available date.
            (Default is 5.)

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the cumulative portfolio values for the strategy portfolio and
            the equal–weight portfolio (indexed by date).
        """
        # Ensure asset metrics have been computed.
        if self.asset_stats is None:
            self.compute_fama_french_metrics()

        # Determine the backtest period: 5 years before the latest date.
        latest_date = self.daily_returns.index.max()
        start_date = latest_date - pd.DateOffset(years=backtest_years)
        backtest_data = self.daily_returns.loc[start_date:latest_date]

        # Identify tickers common to both asset_stats and daily_returns in the backtest period.
        valid_tickers = self.asset_stats.index.intersection(backtest_data.columns)
        if len(valid_tickers) == 0:
            raise ValueError("No overlapping tickers found for backtesting.")

        # For consistency, sort the tickers.
        valid_tickers = sorted(valid_tickers)

        # Obtain the strategy portfolio weights.
        if strategy == "min_variance":
            port_df = self.get_min_variance_portfolio()
        elif strategy == "max_sharpe":
            port_df = self.get_max_sharpe_ratio_portfolio()
        else:
            raise ValueError("Invalid strategy. Please choose either 'min_variance' or 'max_sharpe'.")

        # Extract the ticker weights from the returned DataFrame.
        # The DataFrame has columns like "AAPL_Weight", "MSFT_Weight", etc.
        weight_cols = [col for col in port_df.columns if col.endswith('_Weight')]
        strat_weights = {}
        for col in weight_cols:
            ticker = col.replace('_Weight', '')
            if ticker in valid_tickers:
                # Convert the percentage (e.g. 30.00) into a fraction (0.30).
                strat_weights[ticker] = port_df.loc[0, col] / 100.0
        # Normalize if needed (in case rounding caused the sum not to equal 1).
        total_weight = sum(strat_weights.values())
        if total_weight > 0:
            strat_weights = {ticker: weight / total_weight for ticker, weight in strat_weights.items()}

        # Construct an equal–weight portfolio for the valid tickers.
        equal_weight = 1.0 / len(valid_tickers)
        equal_weights = {ticker: equal_weight for ticker in valid_tickers}

        # Use only the valid tickers for the backtest.
        backtest_data = backtest_data[valid_tickers]

        # Compute the strategy portfolio daily returns.
        # (For any ticker missing in strat_weights, assume a weight of 0.)
        strat_daily = backtest_data.copy()
        for ticker in strat_daily.columns:
            weight = strat_weights.get(ticker, 0.0)
            strat_daily[ticker] = strat_daily[ticker] * weight
        strat_port_daily = strat_daily.sum(axis=1)

        # Compute cumulative returns (buy-and-hold style).
        strat_cum = (1 + strat_port_daily).cumprod() * initial_investment

        # Compute the equal–weight portfolio daily returns.
        equal_daily = backtest_data.copy()
        for ticker in equal_daily.columns:
            equal_daily[ticker] = equal_daily[ticker] * equal_weights[ticker]
        equal_port_daily = equal_daily.sum(axis=1)
        equal_cum = (1 + equal_port_daily).cumprod() * initial_investment

        # Plot the cumulative portfolio values.
        plt.figure(figsize=(12, 6))
        plt.plot(strat_cum.index, strat_cum.values, label=f"Strategy Portfolio ({strategy})")
        plt.plot(equal_cum.index, equal_cum.values, label="Equal Weight Portfolio")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value")
        plt.title(f"Backtest: {backtest_years}–Year Buy-and-Hold Performance")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

        # Return a DataFrame summarizing the cumulative portfolio values.
        result_df = pd.DataFrame({
            "Date": strat_cum.index,
            f"{strategy}_Portfolio": strat_cum.values,
            "Equal_Weight_Portfolio": equal_cum.values
        }).set_index("Date")

        return result_df , plt





