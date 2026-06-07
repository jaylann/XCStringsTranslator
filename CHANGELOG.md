# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.1.0]

### Added
- Live pricing for `openrouter:*` models, fetched from OpenRouter's public model catalog (no API key) and cached on disk (~24h). Cost estimates now cover any OpenRouter model — including brand-new ones — automatically.
- `--no-fetch` flag on `translate` and `estimate` to skip the network and use cached/static prices only.
- OpenRouter shorthand aliases for the newest models: `or-gpt-5.5`, `or-gpt-5.4`, `or-gpt-5.4-mini`, `or-gpt-5.4-nano`, `or-gpt-5-nano`, `or-haiku`, `or-gemini-3.5-flash`, `or-gemini-3.1-pro`, and `or-gemini-3-flash`.
- Continuous integration (GitHub Actions) running ruff lint, ruff format check, and pytest on Python 3.11, 3.12, and 3.13.
- Automated PyPI releases on `v*` tags via Trusted Publishing (OIDC), no stored tokens.
- Community health files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`.
- Issue and pull request templates, `CODEOWNERS`, and Dependabot configuration.
- Security automation: CodeQL analysis, dependency review, `pip-audit`, and OpenSSF Scorecard workflows.
- `.pre-commit-config.yaml` running ruff (lint + format) and common hygiene hooks.
- PEP 561 `py.typed` marker so downstream type checkers use the package's type hints.

### Changed
- Expanded the ruff rule set (bugbear, simplify, comprehensions, pyupgrade, isort, and flake8-bandit security checks) and fixed all surfaced issues.
- Replaced `black` with `ruff format` as the single formatter.

### Fixed
- `translate` rejected valid OpenRouter aliases (e.g. `or-gpt-5.4-nano`) that were absent from the static price table; model validation now accepts any resolved `provider:model`.
- Python API examples and docstring used `from src import ...`, which fails for the installed package; corrected to `xcstrings_translator`.
- Documented `pt-PT` in the supported languages list.

## [1.0.0]

### Added
- Multi-provider translation via pydantic-ai: Anthropic (Claude), OpenAI (GPT), Google (Gemini), and OpenRouter.
- Recursive directory mode — translate every nested `.xcstrings` file under a path.
- Cost estimation and a dry-run mode to preview token usage and price before translating.
- Validation for missing translations and format-specifier mismatches (`%@`, `%lld`, `%1$@`, …).
- Support for 35+ App Store locales, including European Portuguese (`pt-PT`).
- Latest model defaults (Claude Opus 4.8 / Sonnet 4.6, GPT-5.4/5.5, Gemini 3.5).

[Unreleased]: https://github.com/jaylann/XCStringsTranslator/compare/v1.1.0...HEAD
[1.1.0]: https://github.com/jaylann/XCStringsTranslator/compare/v1.0.0...v1.1.0
[1.0.0]: https://github.com/jaylann/XCStringsTranslator/releases/tag/v1.0.0
