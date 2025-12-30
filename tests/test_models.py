"""Tests for src.models - File I/O and data models."""

import json
import pytest
from pathlib import Path

from src.models import (
    XCStringsFile,
    StringEntry,
    StringUnit,
    Localization,
    Variations,
    PluralVariation,
    TranslationContext,
    SUPPORTED_LANGUAGES,
)


class TestXCStringsFileIO:
    """Tests for XCStringsFile.from_file() and to_file()."""

    def test_from_file_valid(self, sample_xcstrings_file):
        """Load a valid xcstrings file."""
        xc = XCStringsFile.from_file(str(sample_xcstrings_file))

        assert xc.sourceLanguage == "en"
        assert xc.version == "1.0"
        assert len(xc.strings) == 2
        assert "Hello" in xc.strings
        assert "World" in xc.strings
        assert xc.strings["World"].comment == "Greeting suffix"

    def test_from_file_example_xcstrings(self):
        """Load the real example.xcstrings file."""
        example_path = Path(__file__).parent.parent / "example.xcstrings"
        if not example_path.exists():
            pytest.skip("example.xcstrings not found")

        xc = XCStringsFile.from_file(str(example_path))

        assert xc.sourceLanguage == "en"
        assert len(xc.strings) == 8
        assert "Welcome" in xc.strings
        assert "Hello, %@!" in xc.strings
        assert "%lld items selected" in xc.strings

    def test_from_file_not_found(self, tmp_path):
        """FileNotFoundError on missing file."""
        missing = tmp_path / "nonexistent.xcstrings"

        with pytest.raises(FileNotFoundError):
            XCStringsFile.from_file(str(missing))

    def test_from_file_invalid_json(self, tmp_path):
        """JSONDecodeError on malformed JSON."""
        bad_file = tmp_path / "bad.xcstrings"
        bad_file.write_text("{ invalid json }")

        with pytest.raises(json.JSONDecodeError):
            XCStringsFile.from_file(str(bad_file))

    def test_from_file_empty_strings(self, tmp_path):
        """Handle file with empty strings dict."""
        empty_file = tmp_path / "empty.xcstrings"
        with open(empty_file, "w") as f:
            json.dump({"sourceLanguage": "en", "version": "1.0", "strings": {}}, f)

        xc = XCStringsFile.from_file(str(empty_file))

        assert xc.sourceLanguage == "en"
        assert len(xc.strings) == 0

    def test_to_file_roundtrip(self, sample_xcstrings_file, tmp_path):
        """Load -> save -> load produces identical data."""
        xc1 = XCStringsFile.from_file(str(sample_xcstrings_file))

        output_path = tmp_path / "output.xcstrings"
        xc1.to_file(str(output_path))

        xc2 = XCStringsFile.from_file(str(output_path))

        assert xc1.sourceLanguage == xc2.sourceLanguage
        assert xc1.version == xc2.version
        assert len(xc1.strings) == len(xc2.strings)
        for key in xc1.strings:
            assert key in xc2.strings

    def test_to_file_unicode_preservation(self, tmp_path):
        """Chinese, emoji, and special chars survive roundtrip."""
        xc = XCStringsFile(
            sourceLanguage="en",
            strings={
                "Hello": StringEntry(
                    localizations={
                        "en": Localization(stringUnit=StringUnit(value="Hello")),
                        "zh-Hans": Localization(stringUnit=StringUnit(value="你好")),
                        "ja": Localization(stringUnit=StringUnit(value="こんにちは")),
                    }
                ),
                "Star": StringEntry(
                    localizations={
                        "en": Localization(stringUnit=StringUnit(value="Star ⭐")),
                    }
                ),
            },
        )

        output_path = tmp_path / "unicode.xcstrings"
        xc.to_file(str(output_path))

        xc2 = XCStringsFile.from_file(str(output_path))

        assert xc2.strings["Hello"].localizations["zh-Hans"].stringUnit.value == "你好"
        assert xc2.strings["Hello"].localizations["ja"].stringUnit.value == "こんにちは"
        assert "⭐" in xc2.strings["Star"].localizations["en"].stringUnit.value


class TestGetTranslatableStrings:
    """Tests for XCStringsFile.get_translatable_strings()."""

    def test_filters_empty_keys(self, xcstrings_format_only_keys):
        """Empty and whitespace-only keys are excluded."""
        translatable = xcstrings_format_only_keys.get_translatable_strings()
        keys = [k for k, _ in translatable]

        assert "" not in keys
        assert "   " not in keys

    def test_filters_format_only_keys(self, xcstrings_format_only_keys):
        """Format-specifier-only keys are excluded."""
        translatable = xcstrings_format_only_keys.get_translatable_strings()
        keys = [k for k, _ in translatable]

        assert "%@" not in keys
        assert "%lld" not in keys
        assert "Valid string" in keys

    def test_returns_valid_strings(self, xcstrings_with_translations):
        """Valid strings are returned with their entries."""
        translatable = xcstrings_with_translations.get_translatable_strings()
        keys = [k for k, _ in translatable]

        assert "Hello" in keys
        assert "Goodbye" in keys
        assert "Welcome %@" in keys  # Has text + specifier
        assert len(translatable) == 3

    def test_empty_file_returns_empty(self, empty_xcstrings):
        """Empty xcstrings returns empty list."""
        translatable = empty_xcstrings.get_translatable_strings()

        assert translatable == []


class TestIsFormatOnly:
    """Tests for XCStringsFile._is_format_only() static method."""

    def test_format_at_only(self):
        """'%@' is format-only."""
        assert XCStringsFile._is_format_only("%@") is True

    def test_format_lld_only(self):
        """'%lld' is format-only."""
        assert XCStringsFile._is_format_only("%lld") is True

    def test_format_percent_only(self):
        """'%%' (escaped percent) is format-only."""
        assert XCStringsFile._is_format_only("%%") is True

    def test_format_positional(self):
        """'%1$@' is format-only."""
        assert XCStringsFile._is_format_only("%1$@") is True

    def test_format_with_text_not_format_only(self):
        """'Hello %@' is NOT format-only."""
        assert XCStringsFile._is_format_only("Hello %@") is False

    def test_regular_text_not_format_only(self):
        """'Hello World' is NOT format-only."""
        assert XCStringsFile._is_format_only("Hello World") is False

    def test_symbols_only(self):
        """Common symbols are format-only."""
        assert XCStringsFile._is_format_only("•") is True
        assert XCStringsFile._is_format_only("©") is True
        assert XCStringsFile._is_format_only("+") is True

    def test_mixed_format_and_symbols(self):
        """Mixed format specifiers and symbols are format-only."""
        assert XCStringsFile._is_format_only("%@ / %@") is True


class TestGetLanguages:
    """Tests for language-related methods."""

    def test_get_existing_languages(self, xcstrings_with_translations):
        """Returns all languages with translations."""
        langs = xcstrings_with_translations.get_existing_languages()

        assert "en" in langs
        assert "de" in langs
        assert "fr" in langs
        assert len(langs) == 3

    def test_get_existing_languages_empty(self, empty_xcstrings):
        """Empty file returns empty set."""
        langs = empty_xcstrings.get_existing_languages()

        assert langs == set()

    def test_get_languages_with_localizations(self, xcstrings_with_translations):
        """Excludes source language."""
        langs = xcstrings_with_translations.get_languages_with_localizations()

        assert "en" not in langs  # Source language excluded
        assert "de" in langs
        assert "fr" in langs
        assert len(langs) == 2


class TestDataModels:
    """Tests for Pydantic data models."""

    def test_string_unit_default_state(self):
        """StringUnit defaults to 'translated' state."""
        unit = StringUnit(value="Test")

        assert unit.state == "translated"
        assert unit.value == "Test"

    def test_localization_with_string_unit(self):
        """Localization can hold a stringUnit."""
        loc = Localization(stringUnit=StringUnit(value="Hello"))

        assert loc.stringUnit is not None
        assert loc.stringUnit.value == "Hello"
        assert loc.variations is None

    def test_localization_with_variations(self):
        """Localization can hold plural variations."""
        loc = Localization(
            variations=Variations(
                plural={
                    "one": PluralVariation(stringUnit=StringUnit(value="1 item")),
                    "other": PluralVariation(stringUnit=StringUnit(value="%lld items")),
                }
            )
        )

        assert loc.stringUnit is None
        assert loc.variations is not None
        assert "one" in loc.variations.plural
        assert "other" in loc.variations.plural

    def test_string_entry_with_comment(self):
        """StringEntry stores comment and localizations."""
        entry = StringEntry(
            comment="Test comment",
            localizations={
                "en": Localization(stringUnit=StringUnit(value="Test"))
            }
        )

        assert entry.comment == "Test comment"
        assert "en" in entry.localizations

    def test_translation_context(self):
        """TranslationContext stores translation metadata."""
        ctx = TranslationContext(
            key="Hello %@",
            comment="Greeting",
            existing_translations={"en": "Hello %@", "de": "Hallo %@"},
            has_format_specifiers=True,
            format_specifiers=["%@"],
        )

        assert ctx.key == "Hello %@"
        assert ctx.has_format_specifiers is True
        assert "%@" in ctx.format_specifiers


class TestSupportedLanguages:
    """Tests for SUPPORTED_LANGUAGES constant."""

    def test_contains_common_languages(self):
        """SUPPORTED_LANGUAGES includes common locales."""
        assert "en" in SUPPORTED_LANGUAGES
        assert "de" in SUPPORTED_LANGUAGES
        assert "fr" in SUPPORTED_LANGUAGES
        assert "es" in SUPPORTED_LANGUAGES
        assert "ja" in SUPPORTED_LANGUAGES
        assert "zh-Hans" in SUPPORTED_LANGUAGES
        assert "zh-Hant" in SUPPORTED_LANGUAGES

    def test_contains_albanian(self):
        """Albanian (sq) is supported."""
        assert "sq" in SUPPORTED_LANGUAGES

    def test_language_count(self):
        """At least 35 languages supported."""
        assert len(SUPPORTED_LANGUAGES) >= 35
