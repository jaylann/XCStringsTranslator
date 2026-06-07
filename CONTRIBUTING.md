# Contributing

Thanks for your interest in improving XCStrings Translator! Contributions of all kinds are welcome — bug reports, feature ideas, docs, and code.

## Development setup

Requires Python 3.11+.

```bash
git clone https://github.com/jaylann/XCStringsTranslator.git
cd XCStringsTranslator
make dev            # pip install -e ".[dev]"
pre-commit install --hook-type commit-msg   # enable the conventional-commit check
```

## Running checks

CI runs the same three commands on every push and pull request. Run them locally before opening a PR:

```bash
ruff check .            # lint
ruff format --check .   # formatting (use `ruff format .` to auto-fix)
pytest tests/ -v        # tests
```

`make test` runs the test suite.

## Pull requests

1. Fork and create a feature branch (`feat/...`, `fix/...`, `chore/...`).
2. Make your change and add or update tests.
3. Ensure lint, format, and tests all pass.
4. Open a PR using the template; link any related issue (`Closes #N`).

PR titles **must** follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat: …`, `fix: …`, `feat!: …` for breaking) — the `pr-title` check enforces this.
PRs are squash-merged, so the PR title becomes the commit on `main`, and that history is
what drives automated versioning and the `CHANGELOG.md` (see Releases below). You don't
need to edit `CHANGELOG.md` or `version` by hand.

Keep PRs focused — one logical change per PR makes review faster.

## Reporting bugs

Use the bug report issue template. Include the version, provider/model, the exact command, and a minimal `.xcstrings` snippet that reproduces the problem (redact private strings).

## Releases (maintainers)

Releases are automated end-to-end via [release-please](https://github.com/googleapis/release-please)
and PyPI [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — no API tokens, no manual tagging.

1. Merge Conventional-Commit PRs to `main` as usual. `fix:` → patch, `feat:` → minor,
   `feat!:`/`fix!:` (or `BREAKING CHANGE:`) → major.
2. release-please keeps an open **release PR** that bumps `version` in `pyproject.toml` and
   updates `CHANGELOG.md` from those commits.
3. Merge the release PR. release-please tags `vX.Y.Z` and creates the GitHub Release, which
   chains into `publish.yml` to build and publish to PyPI.

A manual tag push (`git tag vX.Y.Z && git push origin vX.Y.Z`) still triggers `publish.yml`
as a fallback.

**One-time setup** (PyPI maintainer, in the PyPI project settings → Publishing):
register a trusted publisher with:
- Repository: `jaylann/XCStringsTranslator`
- Workflow: `publish.yml`
- Environment: `pypi`

By contributing you agree your contributions are licensed under the project's [MIT License](LICENSE).
