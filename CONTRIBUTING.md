# Contributing

Thanks for your interest in improving XCStrings Translator! Contributions of all kinds are welcome — bug reports, feature ideas, docs, and code.

## Development setup

Requires Python 3.11+.

```bash
git clone https://github.com/jaylann/XCStringsTranslator.git
cd XCStringsTranslator
make dev            # pip install -e ".[dev]"
```

## Running checks

CI runs the same three commands on every push and pull request. Run them locally before opening a PR:

```bash
ruff check .        # lint
black --check .     # formatting (use `black .` to auto-fix)
pytest tests/ -v    # tests
```

`make test` runs the test suite.

## Pull requests

1. Fork and create a feature branch (`feat/...`, `fix/...`, `chore/...`).
2. Make your change and add or update tests.
3. Ensure lint, format, and tests all pass.
4. Add an entry to `CHANGELOG.md` under the **Unreleased** section.
5. Open a PR using the template; link any related issue (`Closes #N`).

Keep PRs focused — one logical change per PR makes review faster.

## Reporting bugs

Use the bug report issue template. Include the version, provider/model, the exact command, and a minimal `.xcstrings` snippet that reproduces the problem (redact private strings).

## Releases (maintainers)

Releases are automated via GitHub Actions using PyPI [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — no API tokens are stored.

1. Bump `version` in `pyproject.toml`.
2. Move `CHANGELOG.md` "Unreleased" entries under a new version heading.
3. Commit, then tag: `git tag vX.Y.Z && git push origin vX.Y.Z`.
4. The `publish.yml` workflow builds and publishes to PyPI on the tag.

**One-time setup** (PyPI maintainer, in the PyPI project settings → Publishing):
register a trusted publisher with:
- Repository: `jaylann/XCStringsTranslator`
- Workflow: `publish.yml`
- Environment: `pypi`

By contributing you agree your contributions are licensed under the project's [MIT License](LICENSE).
