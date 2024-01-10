import datetime
import requests
import pandas as pd
from lazypredict.Supervised import LazyRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, Lasso
from sklearn.ensemble import BaggingRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt

response = requests.get(url="https://api.gemini.com/v1/symbols")
symbols = response.json()
symbol = "btcusd"
time_frame = "1day"
url = f"https://api.gemini.com/v2/candles/{symbol}/{time_frame}"
response = requests.get(url=url)
data = response.json()

candles_df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume"])
candles_df["timestamp"] = pd.to_datetime(candles_df["timestamp"], unit="ms")

candles_df.to_csv('crypto_data.csv', index=False)


candles_df["momentum"] = candles_df["close"].pct_change()


candles_df["signal"] = 0  # 0: Hold, 1: Long, -1: Short
candles_df.loc[candles_df["momentum"] > 0, "signal"] = 1
candles_df.loc[candles_df["momentum"] < 0, "signal"] = -1

candles_df["daily_return"] = candles_df["close"].pct_change()
candles_df["strategy_return"] = candles_df["daily_return"] * candles_df["signal"].shift(1)

sharpe_ratio = candles_df["strategy_return"].mean() / candles_df["strategy_return"].std()
max_drawdown = (candles_df["strategy_return"].cumsum() - candles_df["strategy_return"].cumsum().expanding().min()).min()
total_return = candles_df["strategy_return"].cumsum().iloc[-1]

print(f"Sharpe Ratio: {sharpe_ratio}")
print(f"Maximum Drawdown: {max_drawdown}")
print(f"Total Return: {total_return}")



