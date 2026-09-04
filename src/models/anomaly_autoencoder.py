import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "log_return",
    "rolling_vol_5d",
    "rolling_vol_21d",
    "rolling_vol_63d",
    "zscore_5d",
    "zscore_21d",
]


class Autoencoder(nn.Module):
    """
    Simple feedforward autoencoder: compresses input features into a small
    bottleneck, then reconstructs them. Trained to minimize reconstruction error
    on normal data; high error at inference time signals an anomaly.
    """

    def __init__(self, input_dim: int, bottleneck_dim: int = 2):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 8),
            nn.ReLU(),
            nn.Linear(8, bottleneck_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 8),
            nn.ReLU(),
            nn.Linear(8, input_dim),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


def train_autoencoder(df: pd.DataFrame, epochs: int = 100, lr: float = 0.01, random_state: int = 42) -> tuple:
    """
    Train an autoencoder on engineered features. Returns the trained model,
    the fitted scaler, and the scaled feature tensor used for training.
    """
    torch.manual_seed(random_state)

    X = df[FEATURE_COLUMNS].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    model = Autoencoder(input_dim=X_tensor.shape[1])
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        reconstructed = model(X_tensor)
        loss = loss_fn(reconstructed, X_tensor)
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} — reconstruction loss: {loss.item():.6f}")

    return model, scaler, X_tensor


def add_autoencoder_scores(df: pd.DataFrame, model: Autoencoder, X_tensor: torch.Tensor, threshold_percentile: float = 95) -> pd.DataFrame:
    """
    Compute per-sample reconstruction error and flag anomalies above a percentile threshold.
    """
    df = df.copy()
    model.eval()
    with torch.no_grad():
        reconstructed = model(X_tensor)
        errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1).numpy()

    df["reconstruction_error"] = errors
    threshold = np.percentile(errors, threshold_percentile)
    df["is_anomaly_ae"] = (df["reconstruction_error"] > threshold).astype(int)
    df["is_anomaly_ae"] = df["is_anomaly_ae"].replace({1: -1, 0: 1})  # match IsolationForest's -1/1 convention

    return df, threshold

if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
    DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

    df = pd.read_csv(PROCESSED_DIR / "AAPL_features.csv", index_col=0, parse_dates=True)

    model, scaler, X_tensor = train_autoencoder(df, epochs=100)
    df, threshold = add_autoencoder_scores(df, model, X_tensor, threshold_percentile=95)

    n_anomalies = (df["is_anomaly_ae"] == -1).sum()
    print(f"\nFlagged {n_anomalies} anomalies out of {len(df)} days ({n_anomalies/len(df)*100:.1f}%)")
    print(f"Reconstruction error threshold: {threshold:.6f}")
    print("\nTop 10 most anomalous days (highest reconstruction error):")
    print(df.nlargest(10, "reconstruction_error")[["Close", "log_return", "rolling_vol_21d", "reconstruction_error"]])

    # Plot price with flagged anomalies overlaid
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["Close"], label="Close price", linewidth=1, alpha=0.7)

    anomalies = df[df["is_anomaly_ae"] == -1]
    plt.scatter(anomalies.index, anomalies["Close"], color="purple", s=30, label="Flagged anomaly (Autoencoder)", zorder=5)

    plt.title("AAPL — Autoencoder anomaly detection")
    plt.xlabel("Date")
    plt.ylabel("Price ($)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "autoencoder_anomalies.png")
    print("\nSaved plot to docs/autoencoder_anomalies.png")