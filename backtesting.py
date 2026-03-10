import pandas as pd

def run_backtest(data):

    data = data.copy()

    # Market returns
    data["Market_Return"] = data["Close"].pct_change()

    # Strategy positions
    data["Position"] = 0
    data.loc[data["Buy_Signal"], "Position"] = 1
    data.loc[data["Sell_Signal"], "Position"] = -1

    data["Position"] = data["Position"].replace(to_replace=0, method="ffill")

    # Strategy returns
    data["Strategy_Return"] = data["Position"].shift(1) * data["Market_Return"]

    # Cumulative returns
    data["Cumulative_Market"] = (1 + data["Market_Return"]).cumprod()
    data["Cumulative_Strategy"] = (1 + data["Strategy_Return"]).cumprod()

    strategy_return = data["Cumulative_Strategy"].iloc[-1] - 1
    market_return = data["Cumulative_Market"].iloc[-1] - 1

    return data, strategy_return, market_return