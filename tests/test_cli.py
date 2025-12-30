"""Tests for src.cli - CLI commands."""

import json
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from typer.testing import CliRunner

from src.cli import (
    app,
    _canonicalize_bcp47_tag,
    _normalize_language_tag,
    _parse_target_languages,
)
from src.models import SUPPORTED_LANGUAGES
from src.translator import TranslationResult, TranslationItem


runner = CliRunner()


class TestBCP47Canonicalization:
    """Tests for _canonicalize_bcp47_tag() helper."""

    def test_pt_br_underscore(self):
        """pt_br -> pt-BR"""
        assert _canonicalize_bcp47_tag("pt_br") == "pt-BR"

    def test_pt_br_hyphen(self):
        """pt-br -> pt-BR"""
        assert _canonicalize_bcp47_tag("pt-br") == "pt-BR"

    def test_zh_hans_lowercase(self):
        """zh-hans -> zh-Hans"""
        assert _canonicalize_bcp47_tag("zh-hans") == "zh-Hans"

    def test_zh_hant_mixed(self):
        """ZH_hant -> zh-Hant"""
        assert _canonicalize_bcp47_tag("ZH_hant") == "zh-Hant"

    def test_simple_code(self):
        """de -> de"""
        assert _canonicalize_bcp47_tag("de") == "de"

    def test_empty_string(self):
        """Empty string returns empty."""
        assert _canonicalize_bcp47_tag("") == ""

    def test_sq_al(self):
        """sq_al -> sq-AL"""
        assert _canonicalize_bcp47_tag("sq_al") == "sq-AL"


class TestLanguageNormalization:
    """Tests for _normalize_language_tag() helper."""

    def test_country_code_cn(self):
        """CN -> zh-Hans"""
        assert _normalize_language_tag("CN") == "zh-Hans"

    def test_country_code_br(self):
        """BR -> pt-BR"""
        assert _normalize_language_tag("BR") == "pt-BR"

    def test_country_code_al(self):
        """AL -> sq"""
        assert _normalize_language_tag("AL") == "sq"

    def test_lowercase_al(self):
        """al -> sq"""
        assert _normalize_language_tag("al") == "sq"

    def test_zh_cn_alias(self):
        """zh-CN -> zh-Hans"""
        assert _normalize_language_tag("zh-CN") == "zh-Hans"

    def test_zh_tw_alias(self):
        """zh-TW -> zh-Hant"""
        assert _normalize_language_tag("zh-TW") == "zh-Hant"

    def test_regular_code_unchanged(self):
        """Regular codes pass through."""
        assert _normalize_language_tag("de") == "de"
        assert _normalize_language_tag("fr") == "fr"


class TestParseTargetLanguages:
    """Tests for _parse_target_languages() helper."""

    def test_single_language(self):
        """Single language."""
        result = _parse_target_languages("fr")
        assert result == ["fr"]

    def test_multiple_languages(self):
        """Comma-separated languages."""
        result = _parse_target_languages("fr,de,es")
        assert result == ["fr", "de", "es"]

    def test_with_spaces(self):
        """Spaces are trimmed."""
        result = _parse_target_languages("fr, de, es")
        assert result == ["fr", "de", "es"]

    def test_normalization_applied(self):
        """Normalization is applied to each."""
        result = _parse_target_languages("CN,BR")
        assert result == ["zh-Hans", "pt-BR"]

    def test_empty_string(self):
        """Empty string returns empty list."""
        result = _parse_target_languages("")
        assert result == []


class TestTranslateCommand:
    """Tests for the 'translate' CLI command."""

    def test_file_not_found(self, tmp_path):
        """Exit 1 when input file doesn't exist."""
        result = runner.invoke(app, ["translate", str(tmp_path / "nonexistent.xcstrings"), "-l", "fr"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_no_languages(self, sample_xcstrings_file):
        """Exit 1 when no languages specified."""
        result = runner.invoke(app, ["translate", str(sample_xcstrings_file)])

        assert result.exit_code == 1
        assert "no languages" in result.output.lower() or "-l" in result.output

    def test_invalid_language(self, sample_xcstrings_file):
        """Exit 1 for unsupported language."""
        result = runner.invoke(app, ["translate", str(sample_xcstrings_file), "-l", "xx-invalid"])

        assert result.exit_code == 1
        assert "unsupported" in result.output.lower()

    def test_fill_missing_with_languages_mutually_exclusive(self, sample_xcstrings_file):
        """Exit 1 when both --fill-missing and -l provided."""
        result = runner.invoke(app, ["translate", str(sample_xcstrings_file), "-l", "fr", "--fill-missing"])

        assert result.exit_code == 1
        assert "mutually exclusive" in result.output.lower()

    def test_fill_missing_with_overwrite_mutually_exclusive(self, sample_xcstrings_file):
        """Exit 1 when both --fill-missing and --overwrite provided."""
        result = runner.invoke(app, ["translate", str(sample_xcstrings_file), "--fill-missing", "--overwrite"])

        assert result.exit_code == 1
        assert "mutually exclusive" in result.output.lower()

    def test_dry_run_shows_estimate(self, sample_xcstrings_file):
        """Dry run shows estimate without making changes."""
        result = runner.invoke(app, ["translate", str(sample_xcstrings_file), "-l", "fr", "--dry-run"])

        assert result.exit_code == 0
        assert "dry run" in result.output.lower()
        assert "estimate" in result.output.lower() or "strings" in result.output.lower()

    def test_translate_success(self, sample_xcstrings_file, tmp_path):
        """Successful translation with mocked API."""
        output_file = tmp_path / "output.xcstrings"

        with patch("src.cli.XCStringsTranslator") as MockTranslator:
            mock_instance = MagicMock()
            mock_instance.stats = MagicMock(
                translated=2, skipped_existing=0, errors=0,
                input_tokens=100, output_tokens=50
            )
            mock_instance.model = "anthropic:claude-sonnet-4-5"

            # Make translate_file return the xcstrings object
            def mock_translate_file(xc, langs, **kwargs):
                return xc

            mock_instance.translate_file.side_effect = mock_translate_file
            MockTranslator.return_value = mock_instance

            result = runner.invoke(app, [
                "translate",
                str(sample_xcstrings_file),
                "-l", "fr",
                "-o", str(output_file)
            ])

            assert result.exit_code == 0
            assert mock_instance.translate_file.called


class TestInfoCommand:
    """Tests for the 'info' CLI command."""

    def test_file_not_found(self, tmp_path):
        """Exit 1 when file doesn't exist."""
        result = runner.invoke(app, ["info", str(tmp_path / "nonexistent.xcstrings")])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_valid_file(self, sample_xcstrings_file):
        """Shows file info for valid file."""
        result = runner.invoke(app, ["info", str(sample_xcstrings_file)])

        assert result.exit_code == 0
        assert "en" in result.output  # Source language
        assert "2" in result.output or "strings" in result.output.lower()


class TestLanguagesCommand:
    """Tests for the 'languages' CLI command."""

    def test_lists_all_languages(self):
        """Lists all supported languages."""
        result = runner.invoke(app, ["languages"])

        assert result.exit_code == 0
        # Check some common language codes appear
        assert "en" in result.output
        assert "de" in result.output
        assert "fr" in result.output
        assert "zh-Hans" in result.output

    def test_shows_language_count(self):
        """Shows total language count."""
        result = runner.invoke(app, ["languages"])

        assert result.exit_code == 0
        # Should mention the count
        assert str(len(SUPPORTED_LANGUAGES)) in result.output or "Total" in result.output


class TestEstimateCommand:
    """Tests for the 'estimate' CLI command."""

    def test_file_not_found(self, tmp_path):
        """Exit 1 when file doesn't exist."""
        result = runner.invoke(app, ["estimate", str(tmp_path / "nonexistent.xcstrings"), "-l", "fr"])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_no_languages(self, sample_xcstrings_file):
        """Exit 1 when no languages specified."""
        result = runner.invoke(app, ["estimate", str(sample_xcstrings_file)])

        assert result.exit_code == 1
        assert "no languages" in result.output.lower() or "-l" in result.output

    def test_invalid_language(self, sample_xcstrings_file):
        """Exit 1 for unsupported language."""
        result = runner.invoke(app, ["estimate", str(sample_xcstrings_file), "-l", "xx-invalid"])

        assert result.exit_code == 1
        assert "unsupported" in result.output.lower()

    def test_success(self, sample_xcstrings_file):
        """Shows cost estimate."""
        result = runner.invoke(app, ["estimate", str(sample_xcstrings_file), "-l", "fr,de"])

        assert result.exit_code == 0
        assert "estimate" in result.output.lower() or "cost" in result.output.lower()


class TestValidateCommand:
    """Tests for the 'validate' CLI command."""

    def test_file_not_found(self, tmp_path):
        """Exit 1 when file doesn't exist."""
        result = runner.invoke(app, ["validate", str(tmp_path / "nonexistent.xcstrings")])

        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_clean_file(self, tmp_path):
        """Exit 0 for file with no issues."""
        clean_file = tmp_path / "clean.xcstrings"
        with open(clean_file, "w") as f:
            json.dump({
                "sourceLanguage": "en",
                "version": "1.0",
                "strings": {
                    "Hello": {
                        "localizations": {
                            "en": {"stringUnit": {"state": "translated", "value": "Hello"}},
                            "fr": {"stringUnit": {"state": "translated", "value": "Bonjour"}},
                        }
                    }
                }
            }, f)

        result = runner.invoke(app, ["validate", str(clean_file)])

        assert result.exit_code == 0

    def test_missing_translations_warning(self, tmp_path):
        """Warns about missing translations."""
        incomplete_file = tmp_path / "incomplete.xcstrings"
        with open(incomplete_file, "w") as f:
            json.dump({
                "sourceLanguage": "en",
                "version": "1.0",
                "strings": {
                    "Hello": {
                        "localizations": {
                            "en": {"stringUnit": {"state": "translated", "value": "Hello"}},
                            "fr": {"stringUnit": {"state": "translated", "value": "Bonjour"}},
                        }
                    },
                    "World": {
                        "localizations": {
                            "en": {"stringUnit": {"state": "translated", "value": "World"}},
                            # Missing fr
                        }
                    }
                }
            }, f)

        result = runner.invoke(app, ["validate", str(incomplete_file)])

        # Should have warning about missing translation
        assert "warning" in result.output.lower() or "missing" in result.output.lower()

    def test_format_mismatch_error(self, tmp_path):
        """Error on format specifier mismatch."""
        mismatched_file = tmp_path / "mismatch.xcstrings"
        with open(mismatched_file, "w") as f:
            json.dump({
                "sourceLanguage": "en",
                "version": "1.0",
                "strings": {
                    "Hello %@": {
                        "localizations": {
                            "en": {"stringUnit": {"state": "translated", "value": "Hello %@"}},
                            # Missing %@ in translation
                            "fr": {"stringUnit": {"state": "translated", "value": "Bonjour"}},
                        }
                    }
                }
            }, f)

        result = runner.invoke(app, ["validate", str(mismatched_file)])

        # Should have error about format mismatch
        assert result.exit_code == 1
        assert "mismatch" in result.output.lower() or "format" in result.output.lower()


class TestContextLoading:
    """Tests for app context loading from file."""

    def test_context_from_flag(self, sample_xcstrings_file):
        """--context flag takes precedence."""
        with patch("src.cli.XCStringsTranslator") as MockTranslator:
            mock_instance = MagicMock()
            mock_instance.estimate_cost.return_value = {
                "total_strings": 2,
                "total_to_translate": 2,
                "strings_per_language": {"fr": 2},
                "estimated_input_tokens": 200,
                "estimated_output_tokens": 60,
                "estimated_cost_usd": 0.01,
            }
            MockTranslator.return_value = mock_instance

            result = runner.invoke(app, [
                "translate",
                str(sample_xcstrings_file),
                "-l", "fr",
                "--context", "My custom context",
                "--dry-run"
            ])

            assert result.exit_code == 0
            # Verify translator was initialized with custom context
            call_kwargs = MockTranslator.call_args[1]
            assert call_kwargs.get("app_context") == "My custom context"

    def test_context_from_file(self, sample_xcstrings_file, tmp_path):
        """context.md file is loaded when present."""
        # Create context.md next to the xcstrings file
        context_file = sample_xcstrings_file.parent / "context.md"
        context_file.write_text("Context from file")

        with patch("src.cli.XCStringsTranslator") as MockTranslator:
            mock_instance = MagicMock()
            mock_instance.estimate_cost.return_value = {
                "total_strings": 2,
                "total_to_translate": 2,
                "strings_per_language": {"fr": 2},
                "estimated_input_tokens": 200,
                "estimated_output_tokens": 60,
                "estimated_cost_usd": 0.01,
            }
            MockTranslator.return_value = mock_instance

            result = runner.invoke(app, [
                "translate",
                str(sample_xcstrings_file),
                "-l", "fr",
                "--dry-run"
            ])

            assert result.exit_code == 0
            # Verify translator was initialized with context from file
            call_kwargs = MockTranslator.call_args[1]
            assert call_kwargs.get("app_context") == "Context from file"

        # Cleanup
        context_file.unlink()
