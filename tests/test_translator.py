import os
import sys
import unittest


# Allow running tests without installing the package.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)


from xcstrings_translator.models import Localization, StringEntry, StringUnit, TranslationContext, XCStringsFile  # noqa: E402
from xcstrings_translator.models import SUPPORTED_LANGUAGES  # noqa: E402
from xcstrings_translator.cli import _normalize_language_tag  # noqa: E402
from xcstrings_translator.translator import XCStringsTranslator  # noqa: E402


class _FakeUsage:
    def __init__(self, input_tokens=10, output_tokens=5):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeBlock:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text):
        self.usage = _FakeUsage()
        self.content = [_FakeBlock(text)]


class _FakeMessages:
    def __init__(self, behavior):
        self._behavior = behavior
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return self._behavior(kwargs)


class _FakeClient:
    def __init__(self, behavior):
        self.messages = _FakeMessages(behavior)


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

    def test_translate_batch_returns_dict_on_success(self):
        translator = XCStringsTranslator(model="sonnet", batch_size=1)

        def behavior(_kwargs):
            return _FakeResponse('{"translations":[{"key":"Hello","value":"Bonjour"}]}')

        translator.client = _FakeClient(behavior)

        entry = StringEntry(localizations={"en": Localization(stringUnit=StringUnit(value="Hello"))})
        ctx = TranslationContext(key="Hello", existing_translations={"en": "Hello"})

        out = translator._translate_batch([("Hello", entry, ctx)], "fr")
        self.assertEqual(out, {"Hello": "Bonjour"})

    def test_translate_batch_retries_with_alias_on_404(self):
        translator = XCStringsTranslator(model="sonnet", batch_size=1)
        snapshot_model = translator.model
        alias_model = translator.model_alias

        def behavior(kwargs):
            if kwargs.get("model") == snapshot_model:
                raise RuntimeError("not_found_error: 404")
            self.assertEqual(kwargs.get("model"), alias_model)
            return _FakeResponse('{"translations":[{"key":"Hello","value":"Bonjour"}]}')

        translator.client = _FakeClient(behavior)

        entry = StringEntry(localizations={"en": Localization(stringUnit=StringUnit(value="Hello"))})
        ctx = TranslationContext(key="Hello", existing_translations={"en": "Hello"})

        out = translator._translate_batch([("Hello", entry, ctx)], "fr")
        self.assertEqual(out, {"Hello": "Bonjour"})
        self.assertEqual(translator.model, alias_model)

    def test_translate_file_does_not_crash(self):
        translator = XCStringsTranslator(model="sonnet", batch_size=10)

        def behavior(_kwargs):
            return _FakeResponse('{"translations":[{"key":"Hello","value":"Bonjour"}]}')

        translator.client = _FakeClient(behavior)

        xc = XCStringsFile(
            sourceLanguage="en",
            strings={
                "Hello": StringEntry(
                    localizations={
                        "en": Localization(stringUnit=StringUnit(value="Hello")),
                    }
                )
            },
        )

        out = translator.translate_file(xc, ["fr"], overwrite=True)
        self.assertEqual(out.strings["Hello"].localizations["fr"].stringUnit.value, "Bonjour")


if __name__ == "__main__":
    unittest.main()
