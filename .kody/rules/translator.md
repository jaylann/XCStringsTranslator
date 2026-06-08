<!-- @kody-sync -->
# Translation core — `src/xcstrings_translator/translator.py`

Applies to: `src/xcstrings_translator/translator.py`
Severity: high

Review the AI translation + pricing core against these failure modes:

- **Model/pricing sync.** `MODEL_ALIASES` and `MODEL_PRICING` must stay in sync —
  every alias must resolve to a `provider:model` that is priced (statically or via
  OpenRouter). Flag any alias that resolves to an unpriced model.
- **Format-specifier fidelity.** Never drop, reorder, or alter Apple format
  specifiers (`%@`, `%lld`, `%1$@`, `%2$lld`, etc.) on a translation round-trip.
- **Plural variations.** Preserve plural categories (`zero`, `one`, `two`, `few`,
  `many`, `other`) — flag any code path that could lose or collapse them.
- **Concurrency.** Guard the ThreadPool/concurrency code against races on the shared
  `TranslationStats` (counts, token totals, cost accumulation).
- **Accounting.** Verify input/output token accounting is correct and that cost math
  uses the *resolved* model's price, not the alias or a default.
