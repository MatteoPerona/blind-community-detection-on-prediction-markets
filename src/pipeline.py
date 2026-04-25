"""Signal construction and BlindCD clustering pipeline."""

import numpy as np
import polars as pl
from sklearn.cluster import KMeans


def build_panel(
    prices_long: pl.DataFrame,
    fidelity_minutes: int = 60,
    max_missing_frac: float = 0.30,
) -> tuple[pl.DataFrame, list[dict]]:
    """
    Build a clean wide panel from long-format price data.

    Returns (wide_df, dropped_markets) where wide_df has rows=timestamps, columns=token_ids.
    """
    dropped = []

    # Pivot to wide
    wide = prices_long.pivot(on="token_id", index="timestamp", values="price").sort("timestamp")

    # Reindex to regular time grid
    ts_min = wide["timestamp"].min()
    ts_max = wide["timestamp"].max()
    regular_grid = pl.DataFrame({
        "timestamp": pl.datetime_range(ts_min, ts_max, interval=f"{fidelity_minutes}m", eager=True)
    })
    wide = regular_grid.join(wide, on="timestamp", how="left")

    token_cols = [c for c in wide.columns if c != "timestamp"]

    # Forward-fill within each market
    wide = wide.with_columns([pl.col(c).forward_fill() for c in token_cols])

    # Drop markets with >30% missing
    n_rows = wide.height
    for c in token_cols:
        missing_frac = wide[c].null_count() / n_rows
        if missing_frac > max_missing_frac:
            dropped.append({"token_id": c, "reason": f"missing {missing_frac:.1%}"})
            wide = wide.drop(c)

    token_cols = [c for c in wide.columns if c != "timestamp"]

    # Drop timestamps where >50% of remaining markets are missing
    n_markets = len(token_cols)
    if n_markets > 0:
        null_counts = wide.select([pl.col(c).is_null().cast(pl.Int32) for c in token_cols]).sum_horizontal()
        mask = null_counts <= (n_markets * 0.5)
        wide = wide.filter(mask)

    # Drop any remaining rows with any nulls (clean panel)
    wide = wide.drop_nulls()

    return wide, dropped


def compute_log_odds_returns(panel_wide: pl.DataFrame) -> pl.DataFrame:
    """
    Compute log-odds returns from a wide price panel.

    Clips prices to [ε, 1-ε], computes log(p/(1-p)), then first-differences.
    """
    eps = 0.001
    token_cols = [c for c in panel_wide.columns if c != "timestamp"]

    # Clip and compute log-odds
    log_odds = panel_wide.select(
        "timestamp",
        *[
            ((pl.col(c).clip(eps, 1 - eps) / (1 - pl.col(c).clip(eps, 1 - eps))).log()).alias(c)
            for c in token_cols
        ],
    )

    # First differences
    returns = log_odds.select(
        pl.col("timestamp").slice(1),
        *[pl.col(c).diff().slice(1).alias(c) for c in token_cols],
    )

    return returns


def run_blindcd(
    returns_df: pl.DataFrame,
    K: int,
    use_correlation: bool = False,
) -> dict:
    """
    Run the Blind Community Detection pipeline.

    Steps: center → covariance → eigendecomposition → top-K embedding → K-means.
    If use_correlation=True, standardize columns (z-score) so covariance == correlation.

    Returns dict with keys: labels, eigenvalues, embedding, cov_matrix, token_ids.
    """
    token_cols = [c for c in returns_df.columns if c != "timestamp"]
    Y = returns_df.select(token_cols).to_numpy()  # (T, N)

    # Center columns
    Y_centered = Y - Y.mean(axis=0, keepdims=True)

    # Optionally standardize to unit variance (covariance becomes correlation)
    if use_correlation:
        stds = Y_centered.std(axis=0, keepdims=True)
        stds = np.maximum(stds, 1e-15)  # avoid division by zero
        Y_centered = Y_centered / stds

    # Sample covariance (= correlation if standardized)
    m = Y_centered.shape[0]
    cov = (1.0 / m) * Y_centered.T @ Y_centered  # (N, N)

    # Eigendecomposition (eigh returns ascending order)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    # Sort descending
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # Top-K embedding
    embedding = eigenvectors[:, :K]  # (N, K)

    # K-means
    kmeans = KMeans(n_clusters=K, n_init=20, random_state=42)
    labels = kmeans.fit_predict(embedding)

    return {
        "labels": labels,
        "eigenvalues": eigenvalues,
        "embedding": embedding,
        "cov_matrix": cov,
        "token_ids": token_cols,
    }
