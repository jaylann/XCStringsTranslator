"""
XCStrings Translator - Translate Apple Localizable.xcstrings files using Claude AI.

Usage:
    from xcstrings_translator import XCStringsFile, XCStringsTranslator

    # Load file
    xcstrings = XCStringsFile.from_file("Localizable.xcstrings")

    # Translate
    translator = XCStringsTranslator(model="sonnet")
    xcstrings = translator.translate_file(xcstrings, ["fr", "es", "it"])

    # Save
    xcstrings.to_file("Localizable.xcstrings")
"""

from .cli import app
from .models import SUPPORTED_LANGUAGES, XCStringsFile
from .translator import XCStringsTranslator

__all__ = [
    "XCStringsFile",
    "XCStringsTranslator",
    "SUPPORTED_LANGUAGES",
    "app",
]
