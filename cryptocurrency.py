import datetime
import requests
import pandas as pd
import matplotlib.pyplot as plt

# Fetch crypto data from GEMINI
response = requests.get(url="https://api.gemini.com/v1/symbols")
symbols = response.json()
symbol = "btcusd"
time_frame = "1day"
url = f"https://api.gemini.com/v2/candles/{symbol}/{time_frame}"
response = requests.get(url=url)
data = response.json()

# Clean and create DataFrame
candles_df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
candles_df["timestamp"] = pd.to_datetime(candles_df["timestamp"], unit="ms")

# Save crypto data as CSV
candles_df.to_csv('crypto_data.csv', index=False)

# Factor Selection: For example, momentum, liquidity, and volatility
# Customize these factors according to your strategy.

# Calculate Momentum
candles_df["momentum"] = candles_df["close"].pct_change()

# Define Long (1), Short (-1), and Hold (0) positions
candles_df["signal"] = 0
candles_df.loc[candles_df["momentum"] > 0, "signal"] = 1
candles_df.loc[candles_df["momentum"] < 0, "signal"] = -1

# Train-Test Split
train_size = int(len(candles_df) * 0.8)
train, test = candles_df.iloc[:train_size], candles_df.iloc[train_size:]

# Back-testing on Train set
train["daily_return"] = train["close"].pct_change()
train["strategy_return"] = train["daily_return"] * train["signal"].shift(1)

# Performance Metrics
sharpe_ratio = train["strategy_return"].mean() / train["strategy_return"].std()
max_drawdown = (train["strategy_return"].cumsum() - train["strategy_return"].cumsum().expanding().min()).min()
total_return = train["strategy_return"].cumsum().iloc[-1]

# Reporting for Train set
print(f"Train Sharpe Ratio: {sharpe_ratio}")
print(f"Train Maximum Drawdown: {max_drawdown}")
print(f"Train Total Return: {total_return}")

# Back-testing on Test set
test["daily_return"] = test["close"].pct_change()
test["strategy_return"] = test["daily_return"] * test["signal"].shift(1)

# Performance Metrics
sharpe_ratio = test["strategy_return"].mean() / test["strategy_return"].std()
max_drawdown = (test["strategy_return"].cumsum() - test["strategy_return"].cumsum().expanding().min()).min()
total_return = test["strategy_return"].cumsum().iloc[-1]

# Reporting for Test set
print(f"Test Sharpe Ratio: {sharpe_ratio}")
print(f"Test Maximum Drawdown: {max_drawdown}")
print(f"Test Total Return: {total_return}")

# Plotting the closing prices over time
plt.figure(figsize=(12, 6))
plt.plot(candles_df['timestamp'], candles_df['close'], label='Closing Price')
plt.title('Bitcoin Price Over Time')
plt.xlabel('Timestamp')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


# Plotting the buy/sell signals
plt.figure(figsize=(12, 6))
plt.plot(train['timestamp'], train['close'], label='Closing Price', alpha=0.5)
plt.scatter(train['timestamp'], train['close'], c=train['signal'], cmap='viridis', marker='o', label='Signal')
plt.title('Buy/Sell Signals Over Time (Training Set)')
plt.xlabel('Timestamp')
plt.ylabel('Price (USD)')
plt.legend()
plt.show()


# Plotting the cumulative returns
plt.figure(figsize=(12, 6))
plt.plot(train['timestamp'], train['strategy_return'].cumsum(), label='Strategy Returns (Train)')
plt.title('Cumulative Returns Over Time (Training Set)')
plt.xlabel('Timestamp')
plt.ylabel('Cumulative Returns')
plt.legend()
plt.show()
