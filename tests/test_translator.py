"""Tests for src.translator - Translation engine."""

import pytest
from unittest.mock import MagicMock, patch

from src.models import (
    Localization,
    StringEntry,
    StringUnit,
    TranslationContext,
    XCStringsFile,
    SUPPORTED_LANGUAGES,
)
from src.translator import (
    XCStringsTranslator,
    resolve_model,
    get_model_cost,
    MODEL_ALIASES,
    MODEL_PRICING,
    TranslationResult,
    TranslationItem,
    TranslationStats,
    OutputParseError,
)
from src.cli import _normalize_language_tag


class TestModelResolution:
    """Tests for resolve_model() and MODEL_ALIASES."""

    def test_resolve_model_shorthands(self):
        """Model shorthands resolve correctly."""
        assert resolve_model("sonnet") == "anthropic:claude-sonnet-4-5"
        assert resolve_model("haiku") == "anthropic:claude-haiku-4-5"
        assert resolve_model("opus") == "anthropic:claude-opus-4-5"
        assert resolve_model("gpt-5") == "openai:gpt-5"
        assert resolve_model("gemini-2.0-flash") == "google-gla:gemini-2.0-flash"

    def test_resolve_model_full_format_unchanged(self):
        """Full provider:model format is returned unchanged."""
        assert resolve_model("anthropic:claude-sonnet-4-5") == "anthropic:claude-sonnet-4-5"
        assert resolve_model("openai:gpt-4o") == "openai:gpt-4o"

    def test_resolve_model_unknown_passthrough(self):
        """Unknown model is passed through as-is."""
        assert resolve_model("unknown-model") == "unknown-model"
        assert resolve_model("custom:my-model") == "custom:my-model"


class TestModelCost:
    """Tests for get_model_cost() function."""

    def test_get_model_cost_known_model(self):
        """Cost calculation for known models."""
        cost = get_model_cost("sonnet", 1_000_000, 1_000_000)

        # sonnet: $3 input, $15 output per 1M tokens
        assert cost == 3.0 + 15.0  # $18

    def test_get_model_cost_unknown_returns_none(self):
        """Unknown model returns None."""
        cost = get_model_cost("unknown-model", 1000, 1000)

        assert cost is None

    def test_get_model_cost_calculation(self):
        """Verify cost math for haiku."""
        # haiku: $1 input, $5 output per 1M tokens
        cost = get_model_cost("haiku", 100_000, 50_000)

        expected = (100_000 / 1_000_000) * 1.0 + (50_000 / 1_000_000) * 5.0
        assert cost == expected


class TestLanguageNormalization:
    """Tests for language code normalization."""

    def test_language_codes_include_albanian(self):
        """Albanian is in SUPPORTED_LANGUAGES."""
        assert "sq" in SUPPORTED_LANGUAGES

    def test_language_alias_country_code_al_maps_to_sq(self):
        """Country code AL maps to Albanian."""
        assert _normalize_language_tag("al") == "sq"
        assert _normalize_language_tag("AL") == "sq"
        assert _normalize_language_tag("sq_AL") == "sq-AL"

    def test_tier_languages_are_normalized(self):
        """Common country codes normalize correctly."""
        assert _normalize_language_tag("BR") == "pt-BR"
        assert _normalize_language_tag("pt_br") == "pt-BR"
        assert _normalize_language_tag("CN") == "zh-Hans"
        assert _normalize_language_tag("zh_cn") == "zh-Hans"
        assert _normalize_language_tag("SE") == "sv"
        assert _normalize_language_tag("DK") == "da"
        assert _normalize_language_tag("NO") == "nb"
        assert _normalize_language_tag("no") == "nb"
        assert _normalize_language_tag("pl") == "pl"
        assert _normalize_language_tag("TR") == "tr"
        assert _normalize_language_tag("AR") == "ar"


class TestTranslatorInitialization:
    """Tests for XCStringsTranslator initialization."""

    def test_translator_initialization(self):
        """Translator initializes with correct model resolution."""
        translator = XCStringsTranslator(model="sonnet", batch_size=10)

        assert translator.model == "anthropic:claude-sonnet-4-5"
        assert translator.batch_size == 10

    def test_translator_custom_context(self):
        """Custom app context is stored."""
        context = "My custom app context"
        translator = XCStringsTranslator(model="sonnet", app_context=context)

        assert translator.app_context == context

    def test_translator_default_context(self):
        """Default context is set when none provided."""
        translator = XCStringsTranslator(model="sonnet")

        assert "mobile app" in translator.app_context.lower()

    def test_translator_concurrency_minimum(self):
        """Concurrency is at least 1."""
        translator = XCStringsTranslator(model="sonnet", concurrency=0)

        assert translator.concurrency >= 1


class TestBuildContext:
    """Tests for XCStringsTranslator._build_context()."""

    def test_build_context_basic(self):
        """Context is built correctly from entry."""
        translator = XCStringsTranslator(model="sonnet")
        entry = StringEntry(
            comment="Test comment",
            localizations={
                "en": Localization(stringUnit=StringUnit(value="Hello %@")),
                "de": Localization(stringUnit=StringUnit(value="Hallo %@")),
            }
        )

        ctx = translator._build_context("Hello %@", entry)

        assert ctx.key == "Hello %@"
        assert ctx.comment == "Test comment"
        assert ctx.existing_translations == {"en": "Hello %@", "de": "Hallo %@"}
        assert ctx.has_format_specifiers is True
        assert "%@" in ctx.format_specifiers

    def test_build_context_no_existing(self):
        """Context with empty localizations."""
        translator = XCStringsTranslator(model="sonnet")
        entry = StringEntry(localizations={})

        ctx = translator._build_context("Test", entry)

        assert ctx.key == "Test"
        assert ctx.existing_translations == {}
        assert ctx.has_format_specifiers is False

    def test_build_context_multiple_format_specs(self):
        """Multiple format specifiers are detected."""
        translator = XCStringsTranslator(model="sonnet")
        entry = StringEntry(
            localizations={
                "en": Localization(stringUnit=StringUnit(value="%1$@ and %2$lld items")),
            }
        )

        ctx = translator._build_context("%1$@ and %2$lld items", entry)

        assert ctx.has_format_specifiers is True
        assert len(ctx.format_specifiers) >= 2

    def test_build_context_no_format_specs(self):
        """Plain text has no format specifiers."""
        translator = XCStringsTranslator(model="sonnet")
        entry = StringEntry(
            localizations={
                "en": Localization(stringUnit=StringUnit(value="Hello World")),
            }
        )

        ctx = translator._build_context("Hello World", entry)

        assert ctx.has_format_specifiers is False
        assert ctx.format_specifiers == []


class TestEstimateCost:
    """Tests for XCStringsTranslator.estimate_cost()."""

    def test_estimate_cost_basic(self):
        """Cost estimation with untranslated strings."""
        translator = XCStringsTranslator(model="sonnet")
        xc = XCStringsFile(
            sourceLanguage="en",
            strings={
                "Hello": StringEntry(
                    localizations={
                        "en": Localization(stringUnit=StringUnit(value="Hello")),
                    }
                ),
                "World": StringEntry(
                    localizations={
                        "en": Localization(stringUnit=StringUnit(value="World")),
                    }
                ),
            },
        )

        estimate = translator.estimate_cost(xc, ["fr", "de"])

        assert estimate["total_strings"] == 2
        assert estimate["total_to_translate"] == 4  # 2 strings * 2 languages
        assert "estimated_cost_usd" in estimate
        assert "strings_per_language" in estimate

    def test_estimate_cost_no_translations_needed(self):
        """All strings already translated."""
        translator = XCStringsTranslator(model="sonnet")
        xc = XCStringsFile(
            sourceLanguage="en",
            strings={
                "Hello": StringEntry(
                    localizations={
                        "en": Localization(stringUnit=StringUnit(value="Hello")),
                        "fr": Localization(stringUnit=StringUnit(value="Bonjour")),
                    }
                ),
            },
        )

        estimate = translator.estimate_cost(xc, ["fr"])

        assert estimate["total_to_translate"] == 0
        assert estimate["strings_per_language"]["fr"] == 0

    def test_estimate_cost_partial_coverage(self):
        """Some strings need translation."""
        translator = XCStringsTranslator(model="sonnet")
        xc = XCStringsFile(
            sourceLanguage="en",
            strings={
                "Hello": StringEntry(
                    localizations={
                        "en": Localization(stringUnit=StringUnit(value="Hello")),
                        "fr": Localization(stringUnit=StringUnit(value="Bonjour")),
                    }
                ),
                "World": StringEntry(
                    localizations={
                        "en": Localization(stringUnit=StringUnit(value="World")),
                    }
                ),
            },
        )

        estimate = translator.estimate_cost(xc, ["fr"])

        assert estimate["strings_per_language"]["fr"] == 1  # Only World needs translation


class TestTranslationStats:
    """Tests for TranslationStats dataclass."""

    def test_default_values(self):
        """Default values are zero."""
        stats = TranslationStats()

        assert stats.total_strings == 0
        assert stats.translated == 0
        assert stats.skipped_existing == 0
        assert stats.errors == 0
        assert stats.input_tokens == 0
        assert stats.output_tokens == 0

    def test_field_updates(self):
        """Fields can be updated."""
        stats = TranslationStats()
        stats.translated = 10
        stats.errors = 2

        assert stats.translated == 10
        assert stats.errors == 2


class TestTranslateBatch:
    """Tests for XCStringsTranslator._translate_batch() with mocked Agent."""

    def test_translate_batch_success(self, mock_agent):
        """Mocked batch translation returns expected results."""
        MockAgent, mock_instance, mock_result = mock_agent

        translator = XCStringsTranslator(model="sonnet")
        entry = StringEntry(
            localizations={"en": Localization(stringUnit=StringUnit(value="Hello"))}
        )
        context = TranslationContext(key="Hello", existing_translations={"en": "Hello"})

        batch = [("Hello", entry, context), ("World", entry, context)]
        result = translator._translate_batch(batch, "fr")

        assert MockAgent.called
        assert "Hello" in result or "World" in result

    def test_translate_batch_token_tracking(self, mock_agent):
        """Token usage is tracked in stats."""
        MockAgent, mock_instance, mock_result = mock_agent

        translator = XCStringsTranslator(model="sonnet")
        entry = StringEntry(
            localizations={"en": Localization(stringUnit=StringUnit(value="Hello"))}
        )
        context = TranslationContext(key="Hello", existing_translations={"en": "Hello"})

        batch = [("Hello", entry, context)]
        translator._translate_batch(batch, "fr")

        assert translator.stats.input_tokens == 100
        assert translator.stats.output_tokens == 50

    def test_translate_batch_parse_error(self):
        """OutputParseError raised on JSON parse failure."""
        with patch("src.translator.Agent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.run_sync.side_effect = Exception("json parse error")
            MockAgent.return_value = mock_instance

            translator = XCStringsTranslator(model="sonnet")
            entry = StringEntry(
                localizations={"en": Localization(stringUnit=StringUnit(value="Test"))}
            )
            context = TranslationContext(key="Test", existing_translations={"en": "Test"})

            with pytest.raises(OutputParseError):
                translator._translate_batch([("Test", entry, context)], "fr")


class TestTranslateFile:
    """Tests for XCStringsTranslator.translate_file() with mocked Agent."""

    def test_translate_file_single_language(self, mock_agent, xcstrings_with_translations):
        """Full translation flow with one target language."""
        MockAgent, mock_instance, mock_result = mock_agent

        # Set up mock to return translation for the untranslated string
        mock_result.output = TranslationResult(
            translations=[TranslationItem(key="Welcome %@", value="Bienvenue %@")]
        )

        translator = XCStringsTranslator(model="sonnet", batch_size=10)
        result = translator.translate_file(xcstrings_with_translations, ["fr"])

        # Welcome %@ should now have fr translation
        assert "fr" in result.strings["Welcome %@"].localizations

    def test_translate_file_skip_existing(self, mock_agent):
        """overwrite=False skips existing translations."""
        MockAgent, mock_instance, mock_result = mock_agent

        xc = XCStringsFile(
            sourceLanguage="en",
            strings={
                "Hello": StringEntry(
                    localizations={
                        "en": Localization(stringUnit=StringUnit(value="Hello")),
                        "fr": Localization(stringUnit=StringUnit(value="Bonjour")),
                    }
                ),
            },
        )

        translator = XCStringsTranslator(model="sonnet")
        translator.translate_file(xc, ["fr"], overwrite=False)

        assert translator.stats.skipped_existing == 1

    def test_translate_file_overwrite_existing(self, mock_agent):
        """overwrite=True replaces existing translations."""
        MockAgent, mock_instance, mock_result = mock_agent
        mock_result.output = TranslationResult(
            translations=[TranslationItem(key="Hello", value="Salut")]
        )

        xc = XCStringsFile(
            sourceLanguage="en",
            strings={
                "Hello": StringEntry(
                    localizations={
                        "en": Localization(stringUnit=StringUnit(value="Hello")),
                        "fr": Localization(stringUnit=StringUnit(value="Bonjour")),
                    }
                ),
            },
        )

        translator = XCStringsTranslator(model="sonnet")
        result = translator.translate_file(xc, ["fr"], overwrite=True)

        # Should be overwritten to "Salut"
        assert result.strings["Hello"].localizations["fr"].stringUnit.value == "Salut"

    def test_translate_file_progress_callback(self, mock_agent):
        """Progress callback is invoked."""
        MockAgent, mock_instance, mock_result = mock_agent
        mock_result.output = TranslationResult(
            translations=[TranslationItem(key="Hello", value="Bonjour")]
        )

        callback_calls = []

        def callback(lang, current, total, batch_size):
            callback_calls.append((lang, current, total, batch_size))

        xc = XCStringsFile(
            sourceLanguage="en",
            strings={
                "Hello": StringEntry(
                    localizations={
                        "en": Localization(stringUnit=StringUnit(value="Hello")),
                    }
                ),
            },
        )

        translator = XCStringsTranslator(model="sonnet")
        translator.translate_file(xc, ["fr"], progress_callback=callback)

        assert len(callback_calls) > 0
        assert callback_calls[0][0] == "fr"

    def test_translate_file_unsupported_language(self, mock_agent):
        """ValueError raised for unsupported language."""
        MockAgent, mock_instance, mock_result = mock_agent

        xc = XCStringsFile(sourceLanguage="en", strings={})

        translator = XCStringsTranslator(model="sonnet")

        with pytest.raises(ValueError, match="Unsupported language"):
            translator.translate_file(xc, ["xx-invalid"])

    def test_translate_file_empty_xcstrings(self, mock_agent, empty_xcstrings):
        """Empty xcstrings file translates without error."""
        MockAgent, mock_instance, mock_result = mock_agent

        translator = XCStringsTranslator(model="sonnet")
        result = translator.translate_file(empty_xcstrings, ["fr"])

        assert len(result.strings) == 0
        assert translator.stats.translated == 0


class TestBatchSplitting:
    """Tests for batch splitting on parse errors."""

    def test_batch_splitting_on_parse_error(self):
        """Parse error triggers batch split and retry."""
        call_count = 0

        def mock_run_sync(msg):
            nonlocal call_count
            call_count += 1

            # First call fails, subsequent calls succeed
            if call_count == 1:
                raise Exception("json parse error")

            result = MagicMock()
            result.output = TranslationResult(
                translations=[TranslationItem(key="Hello", value="Bonjour")]
            )
            usage = MagicMock()
            usage.request_tokens = 50
            usage.response_tokens = 25
            result.usage.return_value = usage
            return result

        with patch("src.translator.Agent") as MockAgent:
            mock_instance = MagicMock()
            mock_instance.run_sync.side_effect = mock_run_sync
            MockAgent.return_value = mock_instance

            xc = XCStringsFile(
                sourceLanguage="en",
                strings={
                    "Hello": StringEntry(
                        localizations={
                            "en": Localization(stringUnit=StringUnit(value="Hello")),
                        }
                    ),
                    "World": StringEntry(
                        localizations={
                            "en": Localization(stringUnit=StringUnit(value="World")),
                        }
                    ),
                },
            )

            translator = XCStringsTranslator(model="sonnet", batch_size=2)
            translator.translate_file(xc, ["fr"])

            # Should have retried with smaller batches
            assert call_count > 1
