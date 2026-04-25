"""Visualization helpers for BlindCD results."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

FIGURES_DIR = Path(__file__).resolve().parent.parent / "notebooks" / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def plot_eigenvalue_spectrum(eigenvalues: np.ndarray, save: bool = True) -> plt.Figure:
    """Log-y scatter of eigenvalues. Annotates elbow if visible."""
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(range(1, len(eigenvalues) + 1), eigenvalues, s=40, zorder=3)
    ax.set_yscale("log")
    ax.set_xlabel("Index")
    ax.set_ylabel("Eigenvalue (log scale)")
    ax.set_title("Covariance Eigenvalue Spectrum")
    ax.grid(True, alpha=0.3)

    # Simple elbow annotation: largest ratio drop
    if len(eigenvalues) > 2:
        ratios = eigenvalues[:-1] / np.maximum(eigenvalues[1:], 1e-15)
        elbow_idx = int(np.argmax(ratios)) + 1  # 1-indexed
        ax.axvline(elbow_idx + 0.5, color="red", linestyle="--", alpha=0.5, label=f"Elbow at {elbow_idx}")
        ax.legend()

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "eigenvalue_spectrum.png", dpi=150)
    return fig


def plot_covariance_heatmap(
    cov_matrix: np.ndarray,
    market_labels: list[str],
    cluster_labels: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """Heatmap of covariance matrix reordered by cluster assignment."""
    # Reorder by cluster
    order = np.argsort(cluster_labels)
    cov_reordered = cov_matrix[np.ix_(order, order)]
    labels_reordered = [market_labels[i] for i in order]
    clusters_reordered = cluster_labels[order]

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cov_reordered,
        xticklabels=labels_reordered,
        yticklabels=labels_reordered,
        cmap="RdBu_r",
        center=0,
        ax=ax,
    )
    ax.set_title("Covariance Matrix (reordered by cluster)")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(fontsize=7)

    # Draw cluster boundaries
    boundaries = np.where(np.diff(clusters_reordered))[0] + 1
    for b in boundaries:
        ax.axhline(b, color="black", linewidth=1.5)
        ax.axvline(b, color="black", linewidth=1.5)

    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "covariance_heatmap.png", dpi=150)
    return fig


def plot_embedding(
    embedding: np.ndarray,
    market_labels: list[str],
    cluster_labels: np.ndarray,
    save: bool = True,
) -> plt.Figure:
    """2D scatter of the first two embedding dimensions, colored by cluster."""
    fig, ax = plt.subplots(figsize=(10, 7))
    unique_clusters = np.unique(cluster_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_clusters)))

    for i, cl in enumerate(unique_clusters):
        mask = cluster_labels == cl
        ax.scatter(embedding[mask, 0], embedding[mask, 1], c=[colors[i]], label=f"Cluster {cl}", s=60, zorder=3)
        for j in np.where(mask)[0]:
            ax.annotate(
                market_labels[j],
                (embedding[j, 0], embedding[j, 1]),
                fontsize=6,
                alpha=0.7,
                textcoords="offset points",
                xytext=(5, 5),
            )

    ax.set_xlabel("Eigenvector 1")
    ax.set_ylabel("Eigenvector 2")
    ax.set_title("Market Embedding (top-2 eigenvectors)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "embedding_scatter.png", dpi=150)
    return fig


def print_clusters(market_metadata: list[dict], cluster_labels: np.ndarray):
    """Print market titles grouped by cluster."""
    clusters = {}
    for meta, label in zip(market_metadata, cluster_labels):
        clusters.setdefault(int(label), []).append(meta)

    for cl in sorted(clusters):
        print(f"\n{'='*60}")
        print(f"  CLUSTER {cl} ({len(clusters[cl])} markets)")
        print(f"{'='*60}")
        for m in clusters[cl]:
            title = m.get("question", m.get("title", m.get("token_id", "?")))
            print(f"  - {title}")
