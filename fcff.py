import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import yfinance as yf
from datetime import date


# Fully connected feedforward neural network

START = "2020-01-01" # Start Date
TODAY = date.today().strftime("%Y-%m-%d") # Get current data in Ymd format as a string

class Model(nn.Module):

	def __init__(self, window, output):
		super(Model, self).__init__()
		self.layer1 = nn.Linear(window, 250)
		self.layer2 = nn.Linear(250, 100)
		self.layer3 = nn.Linear(100, output)
		self.relu = nn.ReLU()

	def forward(self, x):
		x = self.layer1(x)
		x = self.relu(x)
		x = self.layer2(x)
		x = self.relu(x)
		x = self.layer3(x)
		return x

def TrainTest(x, prop = 0.85):
	I = int(prop * len(x))
	train = x[:I]
	test = x[I:]
	return train, test

def Inputs(dataset, window=100, output=30):
    n = len(dataset)
    training_data = []

    for w in range(window, n - output + 1):
        P = dataset[w - window:w]  # Input window
        O = dataset[w:w + output]  # Output
        training_data.append([P,O])

    IN = [torch.tensor(item[0], dtype=torch.float32).unsqueeze(0) for item in training_data]
    OUT = [torch.tensor(item[1], dtype=torch.float32).unsqueeze(0) for item in training_data]

    return torch.cat(IN, dim=0), torch.cat(OUT, dim=0)

def Outputs(dataset, window):
    X = dataset[-window:]
    X = torch.tensor(X, dtype=torch.float32)

    history = X

    return X, history


def load_data(ticker): 
	'''
	Load stock data using ticker as input into pandas dataframe
	Returns data from START until TODAY
	Date in col[0]
	'''
	data = yf.download(ticker, START, TODAY) 
	data.reset_index(inplace = True) 

	return data


data = load_data("AAPL")
close = data['Close'].values.tolist()[::-1]

epochs = 2000
window = 100
output = 30
learning_rate = 0.001

model = Model(int(window), int(output))

train, test = TrainTest(close)

X, Y = Inputs(train, window = window, output = output)

X = X.squeeze(-1)
Y = Y.squeeze(-1)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)


for epoch in range(epochs):
	outputs = model(X)
	loss = criterion(outputs, Y)
	optimizer.zero_grad() # Gradient descent in nn
	loss.backward()
	optimizer.step()
	print('Epochs left: ', epochs - epoch)



XX, history = Outputs(test, window)

with torch.no_grad():
	test_outputs = model(XX)


predictions = test_outputs[-1].numpy().tolist()

print(predictions)













