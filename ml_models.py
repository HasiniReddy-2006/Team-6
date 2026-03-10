from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model(data):

    ml_data = data[["Close","Volume","SMA","EMA","RSI"]].dropna()

    X = ml_data[["Volume","SMA","EMA","RSI"]]
    y = ml_data["Close"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    model = LinearRegression()
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    return y_test, predictions