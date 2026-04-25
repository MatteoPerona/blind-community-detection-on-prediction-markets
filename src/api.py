"""Minimal Polymarket API client with caching, retry, and rate limiting."""

import asyncio
import hashlib
import json
import time
from pathlib import Path

import httpx

DATA_RAW = Path(__file__).resolve().parent.parent / "data" / "raw"
DATA_RAW.mkdir(parents=True, exist_ok=True)

GAMMA_BASE = "https://gamma-api.polymarket.com"
CLOB_BASE = "https://clob.polymarket.com"

# Conservative rate limiter: 50 req/min = ~1.2s between requests
_MIN_INTERVAL = 1.2
_last_request_time = 0.0


def _cache_key(url: str, params: dict) -> str:
    blob = json.dumps({"url": url, "params": params}, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()[:16]


def _read_cache(key: str):
    path = DATA_RAW / f"{key}.json"
    if path.exists():
        return json.loads(path.read_text())
    return None


def _write_cache(key: str, data):
    path = DATA_RAW / f"{key}.json"
    path.write_text(json.dumps(data, indent=2))


def _sync_get_with_retry(url: str, params: dict | None = None, max_retries: int = 5) -> dict | list:
    """Synchronous GET with exponential backoff on 429/5xx."""
    global _last_request_time
    params = params or {}
    key = _cache_key(url, params)
    cached = _read_cache(key)
    if cached is not None:
        return cached

    for attempt in range(max_retries):
        # Rate limit
        elapsed = time.time() - _last_request_time
        if elapsed < _MIN_INTERVAL:
            time.sleep(_MIN_INTERVAL - elapsed)

        resp = httpx.get(url, params=params, timeout=30.0)
        _last_request_time = time.time()

        if resp.status_code == 200:
            data = resp.json()
            _write_cache(key, data)
            return data
        elif resp.status_code in (429, 500, 502, 503, 504):
            delay = 2 ** (attempt + 1)
            print(f"  [{resp.status_code}] retry {attempt+1}/{max_retries} in {delay}s...")
            time.sleep(delay)
        else:
            resp.raise_for_status()

    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


async def _async_get_with_retry(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    params: dict | None = None,
    max_retries: int = 5,
) -> dict | list:
    """Async GET with exponential backoff on 429/5xx, respecting a semaphore for rate limiting."""
    params = params or {}
    key = _cache_key(url, params)
    cached = _read_cache(key)
    if cached is not None:
        return cached

    for attempt in range(max_retries):
        async with semaphore:
            await asyncio.sleep(_MIN_INTERVAL)  # rate limit per request
            resp = await client.get(url, params=params, timeout=30.0)

        if resp.status_code == 200:
            data = resp.json()
            _write_cache(key, data)
            return data
        elif resp.status_code == 400:
            # No data for this market/params — return empty result
            print(f"  [400] no data for {params.get('market', params.get('tokenID', '?'))[:16]}...")
            empty = {"history": []}
            _write_cache(key, empty)
            return empty
        elif resp.status_code in (429, 500, 502, 503, 504):
            delay = 2 ** (attempt + 1)
            print(f"  [{resp.status_code}] retry {attempt+1}/{max_retries} in {delay}s...")
            await asyncio.sleep(delay)
        else:
            resp.raise_for_status()

    raise RuntimeError(f"Failed after {max_retries} retries: {url}")


# ---------------------------------------------------------------------------
# Public API functions
# ---------------------------------------------------------------------------

def get_events(
    tag_slug: str | None = None,
    closed: bool = False,
    limit: int = 100,
    volume_min: float | None = None,
    offset: int = 0,
) -> list[dict]:
    """Fetch events from Gamma API."""
    params = {"limit": limit, "closed": str(closed).lower(), "offset": offset}
    if tag_slug:
        params["tag_slug"] = tag_slug
    data = _sync_get_with_retry(f"{GAMMA_BASE}/events", params)
    if volume_min is not None:
        data = [e for e in data if float(e.get("volume", 0) or 0) >= volume_min]
    return data


def get_markets_for_event(event_id: str) -> list[dict]:
    """Fetch all markets belonging to a given event."""
    return _sync_get_with_retry(f"{GAMMA_BASE}/markets", {"closed": "false", "event_slug": event_id})


def get_markets_by_event_id(event_id: str) -> list[dict]:
    """Fetch markets by numeric event ID."""
    return _sync_get_with_retry(f"{GAMMA_BASE}/markets", {"event_id": event_id})


async def get_price_history(
    token_id: str,
    start_ts: int,
    end_ts: int,
    fidelity: int = 60,
) -> list[dict]:
    """Fetch price history for a single token from the CLOB API."""
    params = {
        "market": token_id,
        "startTs": str(start_ts),
        "endTs": str(end_ts),
        "fidelity": str(fidelity),
    }
    key = _cache_key(f"{CLOB_BASE}/prices-history", params)
    cached = _read_cache(key)
    if cached is not None:
        return cached

    # Use a one-off client for single calls; batch callers should use fetch_all_price_histories
    async with httpx.AsyncClient() as client:
        sem = asyncio.Semaphore(1)
        return await _async_get_with_retry(client, sem, f"{CLOB_BASE}/prices-history", params)


async def fetch_all_price_histories(
    token_ids: list[str],
    start_ts: int,
    end_ts: int,
    fidelity: int = 60,
    max_concurrent: int = 3,
) -> dict[str, list[dict]]:
    """Fetch price histories for multiple tokens with controlled concurrency."""
    semaphore = asyncio.Semaphore(max_concurrent)
    results = {}

    async with httpx.AsyncClient() as client:
        async def _fetch_one(tid: str):
            params = {
                "market": tid,
                "startTs": str(start_ts),
                "endTs": str(end_ts),
                "fidelity": str(fidelity),
            }
            data = await _async_get_with_retry(client, semaphore, f"{CLOB_BASE}/prices-history", params)
            results[tid] = data

        tasks = [_fetch_one(tid) for tid in token_ids]
        await asyncio.gather(*tasks)

    return results
