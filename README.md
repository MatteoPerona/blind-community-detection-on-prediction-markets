# Blind Community Detection on Prediction Markets

Applying [Blind Community Detection](https://arxiv.org/abs/1801.07301) — a spectral clustering method on the covariance of graph signals — to [Polymarket](https://polymarket.com) prediction markets. The goal is to identify clusters of markets that co-move, without any prior knowledge of the underlying graph structure.

## Method

1. Pull price time series for sub-markets within a Polymarket event via the Gamma and CLOB APIs.
2. Transform prices into log-odds returns: `Δθ_t = log(p_t/(1-p_t)) - log(p_{t-1}/(1-p_{t-1}))`.
3. Compute the sample covariance matrix of the returns.
4. Take the top-K eigenvectors of the covariance as a low-dimensional embedding of each market.
5. Run K-means on the embedding to recover clusters.

If markets share latent structure (e.g. correlated outcomes, similar probability regimes), BlindCD should recover that structure from price co-movements alone.

## Phase 0 (current)

Proof-of-concept on a single event slice. The notebook in `notebooks/phase0_exploration.ipynb` runs end-to-end: explores available events, picks one with good properties, pulls data, runs the pipeline, and visualizes results.

**First result:** On the "How many Fed rate cuts in 2026?" event (8 surviving markets, 241 hourly bars), the eigenvalue spectrum shows clear low-pass structure and the recovered clusters separate high-probability outcomes from low-probability tail markets — a sensible grouping that emerges purely from price co-movements.

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
