import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt

# Define the stocks we are interested in
stocks = ['AAPL', 'MSFT']

# Download stock data
data = yf.download(stocks, start='2010-01-01', end='2023-01-01')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Number of portfolios to simulate
num_portfolios = 10000

# Initialize arrays to store weights, returns, volatilities, and Sharpe ratios
all_weights = np.zeros((num_portfolios, len(stocks)))
ret_arr = np.zeros(num_portfolios)
vol_arr = np.zeros(num_portfolios)
sharpe_arr = np.zeros(num_portfolios)

# Risk-free rate (assuming 0 for simplicity)
rf_rate = 0.00

# Simulate random portfolios
for i in range(num_portfolios):
    # Generate random weight vector
    weights = np.array(np.random.random(len(stocks)))
    weights = weights / np.sum(weights)

  # Save weights
    all_weights[i, :] = weights

    # Expected return
    ret_arr[i] = np.sum((returns.mean() * weights * 252))

    # Expected volatility
    vol_arr[i] = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

    # Sharpe Ratio
    sharpe_arr[i] = (ret_arr[i] - rf_rate) / vol_arr[i]

# Get the portfolio with the highest Sharpe ratio
max_sr_ret = ret_arr[sharpe_arr.argmax()]
max_sr_vol = vol_arr[sharpe_arr.argmax()]

# Plot the efficient frontier
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility')
plt.ylabel('Return')
plt.scatter(max_sr_vol, max_sr_ret, c='red', s=50)
plt.show()
