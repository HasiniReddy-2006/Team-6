def generate_signals(data):

    data["Buy_Signal"] = (data["RSI"] < 30)
    data["Sell_Signal"] = (data["RSI"] > 70)

    return data