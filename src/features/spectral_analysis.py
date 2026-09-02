import numpy as np
import pandas as pd


def compute_fft(series: pd.Series, sampling_rate: float = 1.0) -> pd.DataFrame:
    """
    Compute the FFT (Fast Fourier Transform) of a signal to identify dominant cycles.

    sampling_rate: samples per unit time (1.0 = daily data, 1 sample/day)
    Returns a DataFrame of frequency vs amplitude, sorted by amplitude descending.
    """
    n = len(series)
    values = series.values - series.values.mean()  # remove DC component (mean)

    fft_vals = np.fft.rfft(values)
    fft_freq = np.fft.rfftfreq(n, d=1 / sampling_rate)
    amplitude = np.abs(fft_vals) / n

    result = pd.DataFrame({"frequency": fft_freq, "amplitude": amplitude})
    result = result[result["frequency"] > 0]  # drop the zero-frequency term
    result["period_days"] = 1 / result["frequency"]
    return result.sort_values("amplitude", ascending=False)


import pywt


def compute_wavelet_decomposition(series: pd.Series, wavelet: str = "db4", level: int = 4) -> dict:
    """
    Decompose a signal into approximation (trend) and detail (fluctuation) coefficients
    at multiple scales using the Discrete Wavelet Transform.

    wavelet: 'db4' (Daubechies-4) is a common general-purpose choice — smooth, compact support
    level: number of decomposition levels — each level roughly halves the frequency resolution

    Returns a dict with 'approximation' (smoothed trend) and 'details' (list of detail
    coefficients from coarsest to finest scale).
    """
    values = series.values.copy()
    coeffs = pywt.wavedec(values, wavelet=wavelet, level=level)

    # coeffs[0] = approximation (lowest frequency / trend)
    # coeffs[1:] = detail coefficients, coarsest to finest
    approximation = coeffs[0]
    details = coeffs[1:]

    return {
        "approximation": approximation,
        "details": details,
        "wavelet": wavelet,
        "level": level,
    }



if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from pathlib import Path

    RAW_DIR = Path(__file__).resolve().parents[2] / "data" / "raw"
    DOCS_DIR = Path(__file__).resolve().parents[2] / "docs"

    df = pd.read_csv(RAW_DIR / "AAPL.csv", index_col=0, parse_dates=True)
    log_returns = np.log(df["Close"] / df["Close"].shift(1)).dropna()

    # --- FFT analysis ---
    fft_result = compute_fft(log_returns)
    top_cycles = fft_result.head(10)
    print("Top 10 dominant cycles (by amplitude):")
    print(top_cycles[["period_days", "amplitude"]].to_string(index=False))

    fft_sorted = fft_result.sort_values("period_days")  # sort for clean plotting

    plt.figure(figsize=(10, 4))
    plt.plot(fft_sorted["period_days"], fft_sorted["amplitude"])
    plt.xlim(0, 60)
    plt.xlabel("Period (days)")
    plt.ylabel("Amplitude")
    plt.title("AAPL log returns — FFT spectrum (periods < 60 days)")
    plt.tight_layout()
    plt.savefig(DOCS_DIR / "fft_spectrum.png")

    # --- Wavelet analysis ---
    wavelet_result = compute_wavelet_decomposition(log_returns)

    fig, axes = plt.subplots(len(wavelet_result["details"]) + 1, 1, figsize=(12, 10), sharex=False)
    axes[0].plot(wavelet_result["approximation"], color="darkred")
    axes[0].set_title("Approximation (trend, coarsest scale)")

    for i, detail in enumerate(wavelet_result["details"]):
        axes[i + 1].plot(detail, color="darkorange", linewidth=0.7)
        axes[i + 1].set_title(f"Detail level {i + 1} (finer scale fluctuations)")

    plt.tight_layout()
    plt.savefig(DOCS_DIR / "wavelet_decomposition.png")
    print("Saved fft_spectrum.png and wavelet_decomposition.png to docs/")