
def calculate_indicators(data, sma_window, rsi_period):

    data["SMA"] = data["Close"].rolling(window=sma_window).mean()
    data["EMA"] = data["Close"].ewm(span=sma_window, adjust=False).mean()

    delta = data["Close"].diff()

    gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()

    rs = gain / loss
    data["RSI"] = 100 - (100 / (1 + rs))

    return data