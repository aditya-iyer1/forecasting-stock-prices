import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import date
import yfinance as yf


tickers = ['SPY','AAPL','GS','WMT','JNJ','NVDA','AMZN','NFLX','TSLA']

START = "2024-01-01" # Start Date
TODAY = date.today().strftime("%Y-%m-%d") # Get current data in Ymd format as a string

def Beta(x, y):
	'''
	Beta represents sensitivity of a stock's returns relative to the overall market
	'''
	coef = np.cov(x, y) # covariance matrix of x and y
	beat = coef[0][1]
	box = coef[0][0]
	return beat/box

def Capital_Asset_Pricing_Model(rf, rm, beta): # rf = risk-free rate
	'''
	rf: risk-free rate
	rm: expected return on market
	beta: stock sensitivity to market movements
	Capital Asset Pricing Model (CAPM) described relationship between expected returns and risk 
	'''
	return rf + beta * (rm - rf)

def load_data(ticker): 
	'''
	Load stock data using ticker as input into pandas dataframe
	Returns data from START until TODAY
	Date in col[0]
	'''
	data = yf.download(ticker, START, TODAY) 
	data.reset_index(inplace = True) 

	return data

datasets = {} # Initialize empty dictionary for datasets 

for ticker in tickers:
	datasets[ticker] = load_data(ticker) # Make a dataset key for each ticker, add the values in the dataframe as the data in the dict


close = np.array([datasets[tick]['Close'].values.tolist() for tick in tickers]).T # Reads in the close prices and tranposes np array
close = np.squeeze(close)  # Remove the extra dimension

# print(close[-1])

rate_of_return = close[:-1]/close[1:] - 1  # Today's price / Yesterday's price - 1

X = rate_of_return.T # Tranpose of rate_of_return
X0 = X[0] #first price of X


market_rate = close.T[0][-1] / close.T[0][0] - 1 # Current s&p 500 rate divided by original
# print(market_rate)
risk_free_rate = 0.055

BETA, CAPM = [], []

for x in X:
	beta = Beta(X0, x)
	capm = Capital_Asset_Pricing_Model(risk_free_rate, market_rate, beta)
	BETA.append(beta)
	CAPM.append(capm)


from scipy.optimize import minimize

def MinimizeRisk(beta, capm, target_return):

	beta, capm = np.array(beta), np.array(capm)

	def objective(W):
		return W @ beta
	def constraint(W):
		return W @ capm - target_return
	def constraint2(W):
		return W @ np.ones(len(W)) - 1

	W = np.ones(len(beta))

	cons = [{'type':'eq', 'fun': constraint},
			{'type':'eq', 'fun': constraint2}]

	res = minimize(objective, W, method = 'SLSQP', bounds = None, constraints = cons)
	print(res)

	return res.x

W_MinRisk = MinimizeRisk(BETA, CAPM, 0.09)

for t, w in zip(tickers, W_MinRisk):
	








