import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


def create_sequences(values: np.ndarray, seq_length: int = 21) -> tuple:
    """
    Convert a 1D array into overlapping sequences for LSTM input.
    Each sequence of length `seq_length` predicts the next single value.

    E.g. with seq_length=21: use the past 21 days of (log-return-derived)
    volatility to predict day 22's volatility.
    """
    X, y = [], []
    for i in range(len(values) - seq_length):
        X.append(values[i : i + seq_length])
        y.append(values[i + seq_length])
    return np.array(X), np.array(y)


class LSTMVolatilityModel(nn.Module):
    """
    LSTM-based volatility forecaster. Takes a sequence of past volatility
    values and predicts the next value.
    """

    def __init__(self, input_size: int = 1, hidden_size: int = 16, num_layers: int = 1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]  # take output from the final time step
        return self.fc(last_hidden)



def train_lstm_volatility(
    volatility_series: pd.Series,
    seq_length: int = 21,
    train_ratio: float = 0.8,
    epochs: int = 100,
    lr: float = 0.01,
    random_state: int = 42,
) -> dict:
    """
    Train an LSTM to forecast next-day volatility from a sequence of past
    volatility values. Uses a chronological (not random) train/test split,
    which is mandatory for time series to avoid leaking future information.
    """
    torch.manual_seed(random_state)

    values = volatility_series.values.reshape(-1, 1)
    scaler = StandardScaler()
    values_scaled = scaler.fit_transform(values).flatten()

    X, y = create_sequences(values_scaled, seq_length=seq_length)

    # Chronological split: earliest data trains, latest data tests
    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    X_train_t = torch.tensor(X_train, dtype=torch.float32).unsqueeze(-1)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(-1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32).unsqueeze(-1)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(-1)

    model = LSTMVolatilityModel()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train_t)
        loss = loss_fn(pred, y_train_t)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} — train loss: {loss.item():.6f}")

    model.eval()
    with torch.no_grad():
        test_pred_scaled = model(X_test_t).numpy()
        train_pred_scaled = model(X_train_t).numpy()

    # Inverse-transform predictions back to original volatility scale
    test_pred = scaler.inverse_transform(test_pred_scaled)
    test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))
    train_pred = scaler.inverse_transform(train_pred_scaled)
    train_actual = scaler.inverse_transform(y_train.reshape(-1, 1))

    test_rmse = np.sqrt(np.mean((test_pred - test_actual) ** 2))
    print(f"\nTest RMSE: {test_rmse:.6f}")

    return {
        "model": model,
        "scaler": scaler,
        "test_pred": test_pred.flatten(),
        "test_actual": test_actual.flatten(),
        "train_pred": train_pred.flatten(),
        "train_actual": train_actual.flatten(),
        "test_rmse": test_rmse,
        "seq_length": seq_length,
    }



if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
    DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

    df = pd.read_csv(PROCESSED_DIR / "AAPL_features.csv", index_col=0, parse_dates=True)
    volatility_series = df["rolling_vol_21d"].dropna()

    results = train_lstm_volatility(volatility_series, seq_length=21, epochs=100)

    # Plot actual vs predicted volatility on the test set
    test_dates = df.index[-len(results["test_actual"]):]

    # Naive baseline: "tomorrow's volatility = today's volatility"
    naive_pred = results["test_actual"][:-1]  # yesterday's actual value
    naive_actual = results["test_actual"][1:]  # today's actual value
    naive_rmse = np.sqrt(np.mean((naive_pred - naive_actual) ** 2))
    print(f"Naive baseline RMSE (persistence): {naive_rmse:.6f}")
    print(f"LSTM improvement over naive: {(1 - results['test_rmse']/naive_rmse) * 100:.1f}%")


    plt.figure(figsize=(12, 5))
    plt.plot(test_dates, results["test_actual"], label="Actual volatility", linewidth=1.2)
    plt.plot(test_dates, results["test_pred"], label="LSTM predicted volatility", linewidth=1.2, alpha=0.8)
    plt.title(f"AAPL — LSTM volatility forecast (test set, RMSE={results['test_rmse']:.5f})")
    plt.xlabel("Date")
    plt.ylabel("Volatility")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "lstm_volatility.png")
    print("Saved plot to docs/lstm_volatility.png")