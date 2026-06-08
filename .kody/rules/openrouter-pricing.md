<!-- @kody-sync -->
# Pricing fetch — `src/xcstrings_translator/openrouter_pricing.py`

Applies to: `src/xcstrings_translator/openrouter_pricing.py`
Severity: high

Network/pricing code must degrade gracefully:

- **Always set a timeout** on HTTP calls. A price fetch must **never raise to the
  CLI** — failure should fall back, not abort the run.
- **Don't narrow the broad `except Exception`.** It is intentional: it must also
  catch `ImportError` from a missing optional `socks` extra. Do not suggest
  replacing it with specific exception types.
- **Atomic cache writes.** Cache files must be written atomically (temp file +
  rename), never partially.
- **Price validation.** Prices must be normalized to $/1M tokens and validated as
  finite and non-negative before use; reject NaN/inf/negative values.
