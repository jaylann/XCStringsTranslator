.PHONY: install test clean help

# Default target
help:
	@echo "XCStrings Translator - Makefile Commands"
	@echo ""
	@echo "  make install     Install package locally"
	@echo "  make dev         Install with dev dependencies"
	@echo "  make test        Run tests"
	@echo "  make clean       Remove build artifacts"
	@echo ""
	@echo "Usage examples:"
	@echo "  xcstrings translate input.xcstrings -l fr,es,it"
	@echo "  xcstrings estimate input.xcstrings -l fr,es,it,ja,ko"
	@echo "  xcstrings info input.xcstrings"

install:
	pip install -e .

dev:
	pip install -e ".[dev]"

test:
	python -m pytest tests/ -v

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf src/*.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

# Quick translation examples
example-estimate:
	xcstrings estimate Localizable.xcstrings -l fr,es,it,ja,ko

example-translate-fr:
	xcstrings translate Localizable.xcstrings -l fr -m haiku --dry-run
