"""
XCStrings Translator - Translate Apple Localizable.xcstrings files using Claude AI.

Usage:
    from src import XCStringsFile, XCStringsTranslator
    
    # Load file
    xcstrings = XCStringsFile.from_file("Localizable.xcstrings")
    
    # Translate
    translator = XCStringsTranslator(model="sonnet")
    xcstrings = translator.translate_file(xcstrings, ["fr", "es", "it"])
    
    # Save
    xcstrings.to_file("Localizable.xcstrings")
"""

from .models import XCStringsFile, SUPPORTED_LANGUAGES
from .translator import XCStringsTranslator
from .cli import app

__all__ = [
    "XCStringsFile",
    "XCStringsTranslator",
    "SUPPORTED_LANGUAGES",
    "app",
]
