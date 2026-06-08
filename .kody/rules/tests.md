<!-- @kody-sync -->
# Tests — `tests/**/*.py`

Applies to: `tests/**/*.py`
Severity: high

- **Every new behavior needs a test.** Flag new public behavior added without
  corresponding test coverage.
- **No live network.** Network MUST be mocked — no live API/HTTP calls in the unit
  suite. (The `integration` marker is the only exception, and it targets a local
  stub server.)
- **Float assertions.** Use `pytest.approx` for floating-point comparisons.
- **Test public behavior**, not private internals, wherever practical.
