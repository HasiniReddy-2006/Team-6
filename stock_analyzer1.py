# Stock Market Analyzer Dashboard (AI + Indicators + Backtest)
# Run: streamlit run stock_dashboard.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score

st.set_page_config(page_title="Stock Market Analyzer", page_icon="📈", layout="wide")

# -------- Helper Functions --------

@st.cache_data(show_spinner=False, ttl=3600)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    try:
        df = yf.download(
            tickers=ticker,
            period=period,
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
        df = df[cols].dropna()
        return df
    except Exception:
        return pd.DataFrame()


def compute_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / (loss + 1e-12)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def compute_macd(close):
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def compute_smi(high, low, close, period=14, smooth=3):
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()

    mid = (hh + ll) / 2
    diff = close - mid
    range_hl = (hh - ll) / 2

    smi = (diff.rolling(smooth).mean() / (range_hl.rolling(smooth).mean() + 1e-12)) * 100
    return smi


def cagr_from_equity(equity):
    days = len(equity)
    years = days / 252
    return equity.iloc[-1] ** (1 / years) - 1


def max_drawdown(equity):
    roll_max = equity.cummax()
    drawdown = equity / roll_max - 1
    return drawdown.min()


# -------- Sidebar --------

st.sidebar.header("Settings")

ticker = st.sidebar.text_input("Stock ticker", "AAPL").upper()
period = st.sidebar.selectbox("History period", ["1y", "2y", "5y", "10y", "max"])
threshold = st.sidebar.slider("Probability Threshold", 0.50, 0.70, 0.55)
test_size = st.sidebar.slider("Test Split (%)", 20, 40, 30)

st.title("📈 Stock Market Analyzer (AI + Indicators)")

# -------- Load Data --------

with st.spinner("Downloading stock data..."):
    df = load_data(ticker, period)

if df.empty:
    st.error("No data found")
    st.stop()

st.subheader("Raw Data")
st.dataframe(df.tail(10), use_container_width=True)

# -------- Feature Engineering --------

feat = df.copy()

feat["ret"] = feat["Close"].pct_change()

# SMA
feat["sma10"] = feat["Close"].rolling(10).mean()
feat["sma50"] = feat["Close"].rolling(50).mean()

# EMA
feat["ema12"] = feat["Close"].ewm(span=12, adjust=False).mean()
feat["ema26"] = feat["Close"].ewm(span=26, adjust=False).mean()

# SMA Ratio
feat["sma_ratio"] = feat["sma10"] / (feat["sma50"] + 1e-12)

# RSI
feat["rsi"] = compute_rsi(feat["Close"])

# MACD
macd, signal, hist = compute_macd(feat["Close"])
feat["macd"] = macd
feat["macd_signal"] = signal
feat["macd_hist"] = hist

# SMI
feat["smi"] = compute_smi(feat["High"], feat["Low"], feat["Close"])

# Target
feat["future_return"] = feat["Close"].pct_change().shift(-1)
feat["target"] = (feat["future_return"] > 0).astype(int)

feat = feat.dropna()

# -------- Indicator Charts --------

st.subheader("Technical Indicators")

st.write("RSI")
st.line_chart(feat["rsi"])

st.write("SMI")
st.line_chart(feat["smi"])

st.write("Moving Averages")
st.line_chart(feat[["sma10", "sma50", "ema12", "ema26"]])

# -------- Machine Learning --------

FEATURES = [
    "ret",
    "sma_ratio",
    "rsi",
    "macd",
    "macd_signal",
    "macd_hist",
    "ema12",
    "ema26",
    "smi"
]

X = feat[FEATURES].values
y = feat["target"].values

split = int(len(feat) * (1 - test_size / 100))

X_train = X[:split]
X_test = X[split:]

y_train = y[:split]
y_test = y[split:]

dates_test = feat.index[split:]
ret_test = feat["ret"].iloc[split:]

# Scaling

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Model

model = RandomForestClassifier(
    n_estimators=400,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

prob = model.predict_proba(X_test)[:, 1]
pred = (prob >= 0.5).astype(int)

# -------- Metrics --------

st.subheader("Model Performance")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy_score(y_test, pred)*100:.2f}%")
col2.metric("Precision", f"{precision_score(y_test, pred)*100:.2f}%")
col3.metric("Recall", f"{recall_score(y_test, pred)*100:.2f}%")
col4.metric("ROC-AUC", f"{roc_auc_score(y_test, prob):.3f}")

# -------- Probability Chart --------

pred_df = pd.DataFrame({
    "Date": dates_test,
    "Probability": prob
}).set_index("Date")

st.subheader("AI Prediction Probability")
st.line_chart(pred_df)

# -------- Backtesting --------

signal = (prob >= threshold).astype(int)

strategy_return = pd.Series(signal, index=dates_test).shift(1).fillna(0) * ret_test

ai_equity = (1 + strategy_return).cumprod()
bh_equity = (1 + ret_test).cumprod()

equity = pd.concat([ai_equity, bh_equity], axis=1)
equity.columns = ["AI Strategy", "Buy & Hold"]

st.subheader("Backtest Result")

st.line_chart(equity)

# -------- Metrics --------

colA, colB = st.columns(2)

colA.metric("AI CAGR", f"{cagr_from_equity(ai_equity):.2%}")
colA.metric("AI Max Drawdown", f"{max_drawdown(ai_equity):.2%}")

colB.metric("BuyHold CAGR", f"{cagr_from_equity(bh_equity):.2%}")
colB.metric("BuyHold Max Drawdown", f"{max_drawdown(bh_equity):.2%}")

# -------- AI Suggestion --------

last_prob = prob[-1]

if last_prob >= threshold:
    st.success(f"AI Suggestion: BUY (Prob {last_prob:.2f})")
else:
    st.error(f"AI Suggestion: SELL / HOLD (Prob {last_prob:.2f})")