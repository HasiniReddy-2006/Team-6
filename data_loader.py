import yfinance as yf
import pandas as pd

def load_stock_data(symbol, period):

    data = yf.download(symbol, period=period)

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    return data