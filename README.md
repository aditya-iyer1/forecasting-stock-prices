# Portfolio Optimization 

This repository contains implementations of portfolio optimization techniques for constructing and analyzing financial portfolios. The first completed module, `optim.py`, focuses on minimizing risk and maximizing returns using **Capital Asset Pricing Model (CAPM)** and **Beta** metrics.

---

## `optim.py`

optim.py is the first module in a repository for portfolio optimization methods. This script focuses on constructing optimized portfolios using Capital Asset Pricing Model (CAPM) and Beta metrics.


### Features

Fetch Stock Data: Retrieves historical data for selected tickers via yfinance.
Beta Calculation: Measures a stock’s sensitivity to market movements.
CAPM Implementation: Models expected returns based on market return, risk-free rate, and Beta.
Portfolio Optimization:
Minimize Portfolio Beta (Risk).
Maximize Portfolio Return.
Visualization: Pie charts show stock allocations for optimized portfolios.

### Key Functions
Beta(x, y): Calculates Beta of a stock relative to the market.
Capital_Asset_Pricing_Model(rf, rm, beta): Computes expected returns using CAPM.
MinimizeRisk(beta, capm, target_return): Optimizes weights to minimize Beta while meeting a target return.
MaximizeReturn(beta, capm, target_beta): Optimizes weights to maximize return while meeting a target Beta.

### Usage
Edit Tickers: Update the tickers list in optim.py with desired stock symbols.
Run the Script: Calculates Beta, CAPM returns, and optimizes portfolios based on constraints.
View Outputs:
Portfolio metrics (e.g., Beta, return).
Allocation pie charts for minimized-risk and maximized-return portfolios.

