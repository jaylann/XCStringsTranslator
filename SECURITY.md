# Security Policy

## Supported versions

The latest released version on PyPI receives security fixes.

| Version | Supported |
| ------- | --------- |
| 1.x     | ✅        |

## Reporting a vulnerability

Please **do not** open a public issue for security vulnerabilities.

Report privately via one of:

- GitHub's [private vulnerability reporting](https://github.com/jaylann/XCStringsTranslator/security/advisories/new)
- Email: Justin@Lanfermann.dev

Please include a description, reproduction steps, and the impact. You can expect an initial response within a few days.

## API keys

This tool calls third-party LLM providers using API keys that **you supply** via environment variables (e.g. `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GEMINI_API_KEY`). Keys are read from the environment at runtime and are never logged or persisted by this tool. Keep your keys out of committed files — use a local `.env` (already git-ignored) based on `.env.example`.
