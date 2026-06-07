"""Tests for dynamic OpenRouter pricing and its integration with get_model_cost."""

import json
import time

import pytest

from xcstrings_translator import openrouter_pricing as orp
from xcstrings_translator.translator import MODEL_ALIASES, get_model_cost, resolve_model

_SAMPLE_API = {
    "data": [
        {
            "id": "openai/gpt-5.4-nano",
            "pricing": {"prompt": "0.0000002", "completion": "0.00000125"},
        },
        {
            "id": "anthropic/claude-sonnet-4.6",
            "pricing": {"prompt": "0.000003", "completion": "0.000015"},
        },
        {"id": "broken/no-pricing"},
        {"id": "broken/bad-values", "pricing": {"prompt": "abc", "completion": "def"}},
        {"pricing": {"prompt": "0.1", "completion": "0.2"}},  # missing id
    ]
}


@pytest.fixture(autouse=True)
def _isolate(monkeypatch, tmp_path):
    """Each test gets a clean memo and a private, writable cache file."""
    orp.reset_cache()
    monkeypatch.setattr(orp, "_cache_path", lambda: tmp_path / "models.json")
    yield
    orp.reset_cache()


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


class TestNormalise:
    def test_converts_per_token_to_per_million(self):
        prices = orp._normalise(_SAMPLE_API["data"])
        assert prices["openai/gpt-5.4-nano"]["input"] == pytest.approx(0.2)
        assert prices["openai/gpt-5.4-nano"]["output"] == pytest.approx(1.25)
        assert prices["anthropic/claude-sonnet-4.6"]["input"] == pytest.approx(3.0)
        assert prices["anthropic/claude-sonnet-4.6"]["output"] == pytest.approx(15.0)

    def test_skips_malformed_entries(self):
        prices = orp._normalise(_SAMPLE_API["data"])
        assert "broken/no-pricing" not in prices
        assert "broken/bad-values" not in prices
        assert len(prices) == 2  # only the two valid entries


class TestFetchAndCache:
    def test_fetch_populates_and_writes_cache(self, monkeypatch, tmp_path):
        monkeypatch.setattr(
            orp.httpx, "get", lambda *a, **k: _FakeResponse(_SAMPLE_API)
        )
        prices = orp.get_openrouter_prices()
        assert prices["openai/gpt-5.4-nano"]["input"] == pytest.approx(0.2)
        assert (tmp_path / "models.json").exists()

    def test_offline_reads_fresh_disk_cache(self, monkeypatch, tmp_path):
        (tmp_path / "models.json").write_text(
            json.dumps(
                {
                    "fetched_at": time.time(),
                    "prices": {"x/y": {"input": 1.0, "output": 2.0}},
                }
            )
        )

        def _boom(*a, **k):
            raise AssertionError("network must not be hit when fetch=False")

        monkeypatch.setattr(orp.httpx, "get", _boom)
        prices = orp.get_openrouter_prices(fetch=False)
        assert prices == {"x/y": {"input": 1.0, "output": 2.0}}

    def test_offline_with_no_cache_returns_empty(self):
        assert orp.get_openrouter_prices(fetch=False) == {}

    def test_network_failure_degrades_gracefully(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("proxy exploded")

        monkeypatch.setattr(orp.httpx, "get", _boom)
        assert orp.get_openrouter_prices() == {}

    def test_stale_cache_used_when_fetch_disabled(self, monkeypatch, tmp_path):
        (tmp_path / "models.json").write_text(
            json.dumps(
                {
                    "fetched_at": time.time() - 10 * 24 * 3600,
                    "prices": {"x/y": {"input": 9.0, "output": 9.0}},
                }
            )
        )
        # Fresh-cache load rejects stale, but fetch=False falls back to stale rather than empty.
        assert orp.get_openrouter_prices(fetch=False) == {
            "x/y": {"input": 9.0, "output": 9.0}
        }


class TestGetModelCostIntegration:
    def test_openrouter_model_uses_live_pricing(self, monkeypatch):
        monkeypatch.setattr(
            orp.httpx, "get", lambda *a, **k: _FakeResponse(_SAMPLE_API)
        )
        # or-gpt-5.4-nano is NOT in the static table, so a non-None cost proves dynamic lookup.
        cost = get_model_cost("or-gpt-5.4-nano", 1_000_000, 1_000_000)
        assert cost == pytest.approx(0.2 + 1.25)

    def test_openrouter_falls_back_to_static_when_offline(self, monkeypatch):
        monkeypatch.setattr(orp, "get_openrouter_model_pricing", lambda *a, **k: None)
        # or-sonnet's slug IS in static MODEL_PRICING -> fallback yields a price.
        cost = get_model_cost("or-sonnet", 1_000_000, 1_000_000)
        assert cost == pytest.approx(3.0 + 15.0)

    def test_unknown_openrouter_model_offline_returns_none(self, monkeypatch):
        monkeypatch.setattr(orp, "get_openrouter_model_pricing", lambda *a, **k: None)
        assert get_model_cost("openrouter:made/up-model", 1000, 1000) is None

    def test_no_fetch_flag_threads_through(self, monkeypatch):
        def _boom(*a, **k):
            raise AssertionError("fetch_live=False must not hit network")

        monkeypatch.setattr(orp.httpx, "get", _boom)
        # No cache + no fetch -> no price -> None, without raising.
        assert get_model_cost("or-gpt-5.4-nano", 1000, 1000, fetch_live=False) is None


class TestNewAliases:
    @pytest.mark.parametrize(
        "alias,expected",
        [
            ("or-gpt-5.4-nano", "openrouter:openai/gpt-5.4-nano"),
            ("or-gpt-5.4-mini", "openrouter:openai/gpt-5.4-mini"),
            ("or-gpt-5.5", "openrouter:openai/gpt-5.5"),
            ("or-haiku", "openrouter:anthropic/claude-haiku-4.5"),
            ("or-gemini-3.5-flash", "openrouter:google/gemini-3.5-flash"),
            ("or-gemini-3.1-pro", "openrouter:google/gemini-3.1-pro-preview"),
        ],
    )
    def test_new_or_aliases_resolve(self, alias, expected):
        assert alias in MODEL_ALIASES
        assert resolve_model(alias) == expected
