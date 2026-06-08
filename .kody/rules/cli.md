<!-- @kody-sync -->
# CLI — `src/xcstrings_translator/cli.py`

Applies to: `src/xcstrings_translator/cli.py`
Severity: high

Typer CLI. Watch for data-loss and safety regressions:

- **In-place write safety.** Flag in-place `.xcstrings` writes that could clobber the
  user's file when no `-o/--output` is given.
- **Dry-run must be inert.** Dry-run code paths must not perform writes or make live
  API calls.
- **Model validation.** Model validation must accept any resolved `provider:model`
  alias — don't reject valid aliases.
- **Language tags.** Flag language-tag normalization regressions (e.g. `zh-Hans` vs
  `zh-CN`, region/script handling).
