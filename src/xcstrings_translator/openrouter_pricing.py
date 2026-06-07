"""
Dynamic price lookup for OpenRouter models.

OpenRouter publishes live pricing for every model it serves at a public,
key-free endpoint. We fetch it once, cache it (in-process and on disk for ~24h),
and expose per-model prices normalised to USD per 1M tokens so they slot straight
into the static ``MODEL_PRICING`` table. Every failure path degrades gracefully:
network down, bad JSON, or ``fetch=False`` all fall back to cached or empty data
rather than raising.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

_MODELS_URL = "https://openrouter.ai/api/v1/models"
_CACHE_TTL_SECONDS = 24 * 60 * 60
_REQUEST_TIMEOUT_SECONDS = 5.0
_MAX_RESPONSE_BYTES = 8 * 1024 * 1024

# Process-lifetime memo so a single CLI run never fetches twice.
_memo: dict[str, dict[str, float]] | None = None


def _cache_path() -> Path:
    """Return the on-disk path of the cached OpenRouter price table."""
    return Path.home() / ".cache" / "xcstrings-translator" / "openrouter_models.json"


def _normalise(models: list[dict]) -> dict[str, dict[str, float]]:
    """Map OpenRouter model entries to ``slug -> {input, output}`` in $/1M tokens."""
    prices: dict[str, dict[str, float]] = {}
    for model in models:
        slug = model.get("id")
        pricing = model.get("pricing") or {}
        if not slug:
            continue
        try:
            # API reports $/token; scale to $/1M to match MODEL_PRICING.
            input_price = float(pricing["prompt"]) * 1_000_000
            output_price = float(pricing["completion"]) * 1_000_000
        except (KeyError, TypeError, ValueError):
            continue
        # Reject non-finite or negative prices so they cannot poison the cache.
        if not (math.isfinite(input_price) and input_price >= 0):
            continue
        if not (math.isfinite(output_price) and output_price >= 0):
            continue
        prices[slug] = {"input": input_price, "output": output_price}
    return prices


def _validated_prices(raw_prices: object) -> dict[str, dict[str, float]]:
    """Coerce a loaded cache payload into clean ``slug -> {input, output}`` data, dropping bad entries."""
    if not isinstance(raw_prices, dict):
        return {}
    out: dict[str, dict[str, float]] = {}
    for slug, entry in raw_prices.items():
        if not isinstance(slug, str) or not isinstance(entry, dict):
            continue
        try:
            input_price = float(entry["input"])
            output_price = float(entry["output"])
        except (KeyError, TypeError, ValueError):
            continue
        if not (math.isfinite(input_price) and input_price >= 0):
            continue
        if not (math.isfinite(output_price) and output_price >= 0):
            continue
        out[slug] = {"input": input_price, "output": output_price}
    return out


def _load_disk_cache(*, ignore_ttl: bool = False) -> dict[str, dict[str, float]] | None:
    """Read and validate the disk cache, returning None when missing, stale, or corrupt."""
    try:
        raw = json.loads(_cache_path().read_text())
        if not ignore_ttl and time.time() - raw["fetched_at"] > _CACHE_TTL_SECONDS:
            return None
        return _validated_prices(raw["prices"])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def _save_disk_cache(prices: dict[str, dict[str, float]]) -> None:
    """Atomically persist the price table to the disk cache (best-effort; never raises)."""
    try:
        path = _cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({"fetched_at": time.time(), "prices": prices})
        # Write to a temp file in the same dir, then atomically replace so a
        # concurrent reader never observes a half-written file.
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
        tmp.write_text(payload)
        os.replace(tmp, path)
    except OSError:
        logger.debug("failed to persist OpenRouter price cache", exc_info=True)


def get_openrouter_prices(*, fetch: bool = True) -> dict[str, dict[str, float]]:
    """
    Return all known OpenRouter prices as ``slug -> {input, output}`` ($/1M tokens).

    Resolution order: in-process memo -> fresh disk cache -> live API (when
    ``fetch``) -> stale disk cache -> empty. Never raises.
    """
    global _memo
    if _memo is not None:
        return _memo

    cached = _load_disk_cache()
    if cached is not None:
        _memo = cached
        return _memo

    if fetch:
        try:
            with httpx.stream(
                "GET", _MODELS_URL, timeout=_REQUEST_TIMEOUT_SECONDS
            ) as resp:
                resp.raise_for_status()
                chunks: list[bytes] = []
                total = 0
                for chunk in resp.iter_bytes():
                    total += len(chunk)
                    if total > _MAX_RESPONSE_BYTES:
                        raise ValueError("OpenRouter response exceeds size cap")
                    chunks.append(chunk)
            prices = _normalise(json.loads(b"".join(chunks)).get("data", []))
            if prices:
                _save_disk_cache(prices)
                _memo = prices
                return _memo
        except Exception:  # noqa: S110 - pricing is best-effort
            # Any failure (network, proxy, bad JSON) falls back to cached/static
            # prices rather than breaking the CLI.
            logger.debug("OpenRouter price fetch failed", exc_info=True)

    # Offline, or fetch failed/empty: a stale cache beats nothing.
    _memo = _load_disk_cache(ignore_ttl=True) or {}
    return _memo


def get_openrouter_model_pricing(
    slug: str, *, fetch: bool = True
) -> dict[str, float] | None:
    """Return ``{input, output}`` ($/1M tokens) for an OpenRouter slug, or None."""
    return get_openrouter_prices(fetch=fetch).get(slug)


def reset_cache() -> None:
    """Clear the in-process memo (test helper; does not touch the disk cache)."""
    global _memo
    _memo = None
