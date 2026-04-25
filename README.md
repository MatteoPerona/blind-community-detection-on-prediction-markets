# Blind Community Detection on Prediction Markets

Applying [Blind Community Detection](https://arxiv.org/abs/1801.07301), a spectral clustering method on the covariance of graph signals, to [Polymarket](https://polymarket.com) prediction markets. The goal is to identify clusters of markets that co-move, without any prior knowledge of the underlying graph structure.

## Method

1. Pull price time series for sub-markets across Polymarket events via the Gamma and CLOB APIs.
2. Forward-fill to an hourly grid and compute the **correlation matrix** of centered price levels.
3. Take the top-K eigenvectors of the correlation as a low-dimensional embedding of each market.
4. Run K-means on the embedding to recover clusters.

Using price levels (not log-odds returns) avoids the sparse-returns problem — most Polymarket markets trade only a few times per day, making hourly returns 95%+ zeros. The correlation matrix normalizes scale and captures co-movement patterns across events.

## Phase 0 (current)

Two-event separation test. The notebook in `notebooks/phase0_exploration.ipynb` picks two unrelated high-volume events, pools their sub-markets, runs BlindCD with K=2, and checks if clustering correctly separates markets by parent event.

**Result:** On "Starmer out by...?" (3 markets) vs "Will Trump visit China by...?" (4 markets), BlindCD achieves **71.4% accuracy** (5/7 correct). The correlation heatmap shows clear block structure — within-event markets are positively correlated while across-event markets show negative correlation. Two Trump-China markets with near-zero or near-one probabilities get misclassified due to low price variation.

## Project structure

```
├── src/
│   ├── api.py        # Polymarket API client (caching, retry, rate limiting)
│   ├── pipeline.py   # Panel construction, log-odds returns, BlindCD
│   └── plots.py      # Eigenvalue spectrum, covariance heatmap, embedding scatter
├── notebooks/
│   └── phase0_exploration.ipynb
├── data/
│   ├── raw/          # Cached API responses (gitignored)
│   └── processed/    # Cleaned parquet files (gitignored)
└── pyproject.toml
```

## Setup

```bash
uv venv .venv
uv pip install -e .
```

## Usage

Open `notebooks/phase0_exploration.ipynb` and run all cells. The first run fetches data from Polymarket (takes a few minutes due to rate limiting); subsequent runs use cached responses.
