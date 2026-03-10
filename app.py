import streamlit as st
import plotly.graph_objects as go
from data_loader import load_stock_data
from indicators import calculate_indicators
from signals import generate_signals
from ml_models import train_model
from backtesting import run_backtest
st.title("📈 Stock Market Analyzer")

st.sidebar.header("User Inputs")

# Stock ticker
stock_symbol = st.sidebar.text_input("Stock Ticker", "AAPL")

# Time period
time_period = st.sidebar.selectbox(
    "Time Period",
    ["6mo", "1y", "2y", "5y"]
)

# SMA window size
sma_window = st.sidebar.slider(
    "SMA Window Size",
    min_value=5,
    max_value=100,
    value=20
)

# RSI period
rsi_period = st.sidebar.slider(
    "RSI Period",
    min_value=5,
    max_value=30,
    value=14
)
data = load_stock_data(stock_symbol, time_period)

if not data.empty:

    data = calculate_indicators(data, sma_window, rsi_period)
    data = generate_signals(data)
    data, strategy_return, market_return = run_backtest(data)
    st.subheader("Raw Stock Data")
    st.write(data)
    st.subheader("Candlestick Chart")

    fig_candle = go.Figure(data=[go.Candlestick(
        x=data.index,
        open=data["Open"],
        high=data["High"],
        low=data["Low"],
        close=data["Close"],
        name="Candlestick"
    )])

    fig_candle.update_layout(
        height=600,
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig_candle)
    # Close Price
    st.subheader("Close Price")

    fig_price = go.Figure()

    fig_price.add_trace(go.Scatter(
        x=data.index,
        y=data["Close"],
        mode="lines",
        name="Close Price"
    ))

    st.plotly_chart(fig_price)

    # Volume
    st.subheader("Volume")

    fig_volume = go.Figure()

    fig_volume.add_trace(go.Bar(
        x=data.index,
        y=data["Volume"],
        name="Volume"
    ))

    st.plotly_chart(fig_volume)

    # Moving averages
    st.subheader("Moving Averages")

    fig_ma = go.Figure()

    fig_ma.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Close"))
    fig_ma.add_trace(go.Scatter(x=data.index, y=data["SMA"], name=f"SMA ({sma_window})"))
    fig_ma.add_trace(go.Scatter(x=data.index, y=data["EMA"], name=f"EMA ({sma_window})"))

    st.plotly_chart(fig_ma)

    # RSI
    st.subheader("RSI Indicator")

    fig_rsi = go.Figure()

    fig_rsi.add_trace(go.Scatter(
        x=data.index,
        y=data["RSI"],
        mode="lines",
        name="RSI"
    ))

    fig_rsi.add_hline(y=70,line_color="white")
    fig_rsi.add_hline(y=30,line_color="white")

    st.plotly_chart(fig_rsi)
    y_test, predictions = train_model(data)
    st.subheader("ML Price Prediction")

    fig_pred = go.Figure()

    fig_pred.add_trace(go.Scatter(
        x=y_test.index,
        y=y_test,
        mode="lines",
        name="Actual Price"
    ))

    fig_pred.add_trace(go.Scatter(
        x=y_test.index,
        y=predictions,
        mode="lines",
        name="Predicted Price"
    ))

    fig_pred.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig_pred)
    latest_close = data["Close"].iloc[-1]
    latest_open = data["Open"].iloc[-1]
    latest_sma = data["SMA"].iloc[-1]
    latest_ema = data["EMA"].iloc[-1]
    latest_rsi = data["RSI"].iloc[-1]
    predicted_price = predictions[-1]
    score = 0

    # ML prediction
    if predicted_price > latest_close:
        score += 1
    else:
        score -= 1

    # Trend check
    if latest_ema > latest_sma:
        score += 1
    else:
        score -= 1

    # RSI momentum
    if latest_rsi < 30:
        score += 1
    elif latest_rsi > 70:
        score -= 1
    # Simple decision rule
    if score >= 2:
        decision = "BUY"
        color = "green"

    elif score <= -2:
        decision = "SELL"
        color = "red"

    else:
        decision = "HOLD"
        color = "orange"

    st.subheader("Trading Recommendation")

    col1, col2 = st.columns([1,2])

    with col1:
        st.markdown(
            f"""
            <div style="
            background-color:{color};
            padding:40px;
            border-radius:12px;
            text-align:center;
            font-size:45px;
            font-weight:bold;
            color:white;">
            {decision}
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(f"""
        ### Market Summary

        **Open Price:** {latest_open:.2f}  
        **Close Price:** {latest_close:.2f}  
        **Predicted Price:** {predicted_price:.2f}  
        """)
    confidence = abs(predicted_price - latest_close) / latest_close * 100

    st.write("Prediction Confidence:", round(confidence,2), "%")

    latest_sma = data["SMA"].iloc[-1]
    latest_ema = data["EMA"].iloc[-1]

    if latest_ema > latest_sma:
        trend = "Bullish 📈"
    else:
        trend = "Bearish 📉"

    latest_rsi = data["RSI"].iloc[-1]

    if latest_rsi > 60:
        momentum = "Strong"
    elif latest_rsi < 40:
        momentum = "Weak"
    else:
        momentum = "Neutral"

    volatility = data["Close"].pct_change().std()

    if volatility > 0.02:
        volatility_level = "High"
    else:
        volatility_level = "Low"

    entry_price = latest_close
    exit_price = predicted_price
    st.subheader("Market Intelligence")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Trend", trend)
    col2.metric("Momentum", momentum)
    col3.metric("Volatility", volatility_level)
    col4.metric("RSI", round(latest_rsi,2))

    st.markdown("### Trading Levels")

    col5, col6 = st.columns(2)

    col5.metric("Best Entry Price", round(entry_price,2))
    col6.metric("Target Exit Price", round(exit_price,2))
    data["Market_Return"] = data["Close"].pct_change()

    data["Strategy"] = 0
    data.loc[data["Buy_Signal"], "Strategy"] = 1
    data.loc[data["Sell_Signal"], "Strategy"] = -1

    data["Strategy_Return"] = data["Strategy"].shift(1) * data["Market_Return"]

    strategy_return = (1 + data["Strategy_Return"]).cumprod().iloc[-1] - 1
    buy_hold_return = (1 + data["Market_Return"]).cumprod().iloc[-1] - 1
    st.subheader("Strategy Performance")

    col1, col2 = st.columns(2)

    col1.metric(
        "Strategy Return",
        f"{strategy_return*100:.2f}%"
    )

    col2.metric(
        "Buy & Hold Return",
        f"{buy_hold_return*100:.2f}%"
    )

    st.subheader("Strategy vs Market Performance")

    fig_perf = go.Figure()

    fig_perf.add_trace(go.Scatter(
        x=data.index,
        y=data["Cumulative_Strategy"],
        name="Strategy",
        line=dict(color="green", width=3)
    ))

    fig_perf.add_trace(go.Scatter(
        x=data.index,
        y=data["Cumulative_Market"],
        name="Buy & Hold",
        line=dict(color="orange", width=3)
    ))

    fig_perf.update_layout(
        height=500,
        xaxis_title="Date",
        yaxis_title="Cumulative Return",
        template="plotly_dark"
    )

    st.plotly_chart(fig_perf, use_container_width=True)
else:
    st.error("No data found. Please check the stock symbol.")