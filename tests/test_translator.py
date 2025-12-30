import os
import sys
import unittest


# Allow running tests without installing the package.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


from xcstrings_translator.models import Localization, StringEntry, StringUnit, TranslationContext, XCStringsFile  # noqa: E402
from xcstrings_translator.models import SUPPORTED_LANGUAGES  # noqa: E402
from xcstrings_translator.cli import _normalize_language_tag  # noqa: E402
from xcstrings_translator.translator import XCStringsTranslator, resolve_model, MODEL_ALIASES  # noqa: E402


class TranslatorTests(unittest.TestCase):
    def test_language_codes_include_albanian(self):
        self.assertIn("sq", SUPPORTED_LANGUAGES)

    def test_language_alias_country_code_al_maps_to_sq(self):
        self.assertEqual(_normalize_language_tag("al"), "sq")
        self.assertEqual(_normalize_language_tag("AL"), "sq")
        self.assertEqual(_normalize_language_tag("sq_AL"), "sq-AL")

    def test_tier_languages_are_normalized(self):
        self.assertEqual(_normalize_language_tag("BR"), "pt-BR")
        self.assertEqual(_normalize_language_tag("pt_br"), "pt-BR")
        self.assertEqual(_normalize_language_tag("CN"), "zh-Hans")
        self.assertEqual(_normalize_language_tag("zh_cn"), "zh-Hans")
        self.assertEqual(_normalize_language_tag("SE"), "sv")
        self.assertEqual(_normalize_language_tag("DK"), "da")
        self.assertEqual(_normalize_language_tag("NO"), "nb")
        self.assertEqual(_normalize_language_tag("no"), "nb")
        self.assertEqual(_normalize_language_tag("pl"), "pl")
        self.assertEqual(_normalize_language_tag("TR"), "tr")
        self.assertEqual(_normalize_language_tag("AR"), "ar")

    def test_resolve_model_shorthands(self):
        """Test that model shorthands resolve correctly."""
        self.assertEqual(resolve_model("sonnet"), "anthropic:claude-sonnet-4-5")
        self.assertEqual(resolve_model("haiku"), "anthropic:claude-haiku-4-5")
        self.assertEqual(resolve_model("opus"), "anthropic:claude-opus-4-5")
        self.assertEqual(resolve_model("gpt-5"), "openai:gpt-5")
        self.assertEqual(resolve_model("gemini-2.0-flash"), "google-gla:gemini-2.0-flash")

    def test_resolve_model_full_format_unchanged(self):
        """Test that full provider:model format is returned unchanged."""
        self.assertEqual(resolve_model("anthropic:claude-sonnet-4-5"), "anthropic:claude-sonnet-4-5")
        self.assertEqual(resolve_model("openai:gpt-4o"), "openai:gpt-4o")

    def test_translator_initialization(self):
        """Test that translator initializes with correct model resolution."""
        translator = XCStringsTranslator(model="sonnet", batch_size=10)
        self.assertEqual(translator.model, "anthropic:claude-sonnet-4-5")
        self.assertEqual(translator.batch_size, 10)

    def test_translator_custom_context(self):
        """Test that custom app context is stored."""
        context = "My custom app context"
        translator = XCStringsTranslator(model="sonnet", app_context=context)
        self.assertEqual(translator.app_context, context)

    def test_build_context(self):
        """Test that translation context is built correctly."""
        translator = XCStringsTranslator(model="sonnet")
        entry = StringEntry(
            comment="Test comment",
            localizations={
                "en": Localization(stringUnit=StringUnit(value="Hello %@")),
                "de": Localization(stringUnit=StringUnit(value="Hallo %@")),
            }
        )
        ctx = translator._build_context("Hello %@", entry)

        self.assertEqual(ctx.key, "Hello %@")
        self.assertEqual(ctx.comment, "Test comment")
        self.assertEqual(ctx.existing_translations, {"en": "Hello %@", "de": "Hallo %@"})
        self.assertTrue(ctx.has_format_specifiers)
        self.assertIn("%@", ctx.format_specifiers)

    def test_estimate_cost(self):
        """Test cost estimation."""
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

        self.assertEqual(estimate["total_strings"], 2)
        self.assertEqual(estimate["total_to_translate"], 4)  # 2 strings * 2 languages
        self.assertIn("estimated_cost_usd", estimate)
        self.assertIn("strings_per_language", estimate)


if __name__ == "__main__":
    unittest.main()
