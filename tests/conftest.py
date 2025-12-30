"""Shared pytest fixtures for xcstrings-translator tests."""

import json
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.models import (
    XCStringsFile,
    StringEntry,
    StringUnit,
    Localization,
)
from src.translator import TranslationResult, TranslationItem


@pytest.fixture
def sample_xcstrings_dict() -> dict:
    """Minimal valid xcstrings structure."""
    return {
        "sourceLanguage": "en",
        "version": "1.0",
        "strings": {
            "Hello": {
                "localizations": {
                    "en": {"stringUnit": {"state": "translated", "value": "Hello"}}
                }
            },
            "World": {
                "comment": "Greeting suffix",
                "localizations": {
                    "en": {"stringUnit": {"state": "translated", "value": "World"}}
                }
            },
        },
    }


@pytest.fixture
def sample_xcstrings_file(tmp_path, sample_xcstrings_dict) -> Path:
    """Write sample xcstrings to temp file, return path."""
    file_path = tmp_path / "test.xcstrings"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(sample_xcstrings_dict, f, ensure_ascii=False, indent=2)
    return file_path


@pytest.fixture
def xcstrings_with_translations() -> XCStringsFile:
    """XCStringsFile with en/de/fr translations."""
    return XCStringsFile(
        sourceLanguage="en",
        strings={
            "Hello": StringEntry(
                localizations={
                    "en": Localization(stringUnit=StringUnit(value="Hello")),
                    "de": Localization(stringUnit=StringUnit(value="Hallo")),
                    "fr": Localization(stringUnit=StringUnit(value="Bonjour")),
                }
            ),
            "Goodbye": StringEntry(
                comment="Farewell message",
                localizations={
                    "en": Localization(stringUnit=StringUnit(value="Goodbye")),
                    "de": Localization(stringUnit=StringUnit(value="Auf Wiedersehen")),
                }
            ),
            "Welcome %@": StringEntry(
                localizations={
                    "en": Localization(stringUnit=StringUnit(value="Welcome %@")),
                }
            ),
        },
    )


@pytest.fixture
def xcstrings_with_format_specifiers() -> XCStringsFile:
    """XCStringsFile with various format specifiers."""
    return XCStringsFile(
        sourceLanguage="en",
        strings={
            "Hello %@": StringEntry(
                localizations={
                    "en": Localization(stringUnit=StringUnit(value="Hello %@")),
                }
            ),
            "%lld items": StringEntry(
                localizations={
                    "en": Localization(stringUnit=StringUnit(value="%lld items")),
                }
            ),
            "%1$@ and %2$@": StringEntry(
                localizations={
                    "en": Localization(stringUnit=StringUnit(value="%1$@ and %2$@")),
                }
            ),
            "100%% complete": StringEntry(
                localizations={
                    "en": Localization(stringUnit=StringUnit(value="100%% complete")),
                }
            ),
        },
    )


@pytest.fixture
def empty_xcstrings() -> XCStringsFile:
    """Empty xcstrings file with no strings."""
    return XCStringsFile(sourceLanguage="en", strings={})


@pytest.fixture
def xcstrings_format_only_keys() -> XCStringsFile:
    """XCStringsFile with format-only keys that should be skipped."""
    return XCStringsFile(
        sourceLanguage="en",
        strings={
            "%@": StringEntry(
                localizations={
                    "en": Localization(stringUnit=StringUnit(value="%@")),
                }
            ),
            "%lld": StringEntry(
                localizations={
                    "en": Localization(stringUnit=StringUnit(value="%lld")),
                }
            ),
            "": StringEntry(
                localizations={
                    "en": Localization(stringUnit=StringUnit(value="")),
                }
            ),
            "   ": StringEntry(
                localizations={
                    "en": Localization(stringUnit=StringUnit(value="   ")),
                }
            ),
            "Valid string": StringEntry(
                localizations={
                    "en": Localization(stringUnit=StringUnit(value="Valid string")),
                }
            ),
        },
    )


@pytest.fixture
def mock_agent():
    """Mock pydantic-ai Agent for translation tests."""
    with patch("src.translator.Agent") as MockAgent:
        mock_instance = MagicMock()
        mock_result = MagicMock()

        # Default translation response
        mock_result.output = TranslationResult(
            translations=[
                TranslationItem(key="Hello", value="Bonjour"),
                TranslationItem(key="World", value="Monde"),
            ]
        )

        # Mock usage
        mock_usage = MagicMock()
        mock_usage.request_tokens = 100
        mock_usage.response_tokens = 50
        mock_result.usage.return_value = mock_usage

        mock_instance.run_sync.return_value = mock_result
        MockAgent.return_value = mock_instance

        yield MockAgent, mock_instance, mock_result


@pytest.fixture
def mock_agent_with_custom_response():
    """Factory fixture for customizing mock Agent responses."""

    def _create_mock(translations: list[tuple[str, str]], input_tokens: int = 100, output_tokens: int = 50):
        with patch("src.translator.Agent") as MockAgent:
            mock_instance = MagicMock()
            mock_result = MagicMock()

            mock_result.output = TranslationResult(
                translations=[TranslationItem(key=k, value=v) for k, v in translations]
            )

            mock_usage = MagicMock()
            mock_usage.request_tokens = input_tokens
            mock_usage.response_tokens = output_tokens
            mock_result.usage.return_value = mock_usage

            mock_instance.run_sync.return_value = mock_result
            MockAgent.return_value = mock_instance

            return MockAgent, mock_instance, mock_result

    return _create_mock
