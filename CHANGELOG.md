# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Continuous integration (GitHub Actions) running ruff, black, and pytest on Python 3.11 and 3.12.
- Automated PyPI releases on `v*` tags via Trusted Publishing (OIDC), no stored tokens.
- Community health files: `CONTRIBUTING.md`, `CODE_OF_CONDUCT.md`, `SECURITY.md`.
- Issue and pull request templates, `CODEOWNERS`, and Dependabot configuration.
- Explicit `[tool.ruff]` and `[tool.black]` configuration in `pyproject.toml`.

## [1.0.0]

### Added
- Multi-provider translation via pydantic-ai: Anthropic (Claude), OpenAI (GPT), Google (Gemini), and OpenRouter.
- Recursive directory mode — translate every nested `.xcstrings` file under a path.
- Cost estimation and a dry-run mode to preview token usage and price before translating.
- Validation for missing translations and format-specifier mismatches (`%@`, `%lld`, `%1$@`, …).
- Support for 35+ App Store locales, including European Portuguese (`pt-PT`).
- Latest model defaults (Claude Opus 4.8 / Sonnet 4.6, GPT-5.4/5.5, Gemini 3.5).

[Unreleased]: https://github.com/jaylann/XCStringsTranslator/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/jaylann/XCStringsTranslator/releases/tag/v1.0.0
