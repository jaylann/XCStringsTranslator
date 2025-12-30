"""
AI-powered translator for xcstrings files.

Supports multiple AI providers via pydantic-ai:
- Anthropic (Claude)
- OpenAI (GPT)
- Google (Gemini)

Features:
1. Providing context from existing translations (EN/DE)
2. Preserving format specifiers (%@, %lld, etc.)
3. Adapting to the app's tone and style
4. Handling pluralization correctly
5. Batching strings for efficiency
"""

from __future__ import annotations
import re
import json
from typing import Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import threading
from collections import deque

from pydantic import BaseModel, Field
from pydantic_ai import Agent

from .models import (
    XCStringsFile,
    StringEntry,
    Localization,
    StringUnit,
    TranslationContext,
    SUPPORTED_LANGUAGES,
)


# Model shorthand aliases -> provider:model format
MODEL_ALIASES = {
    # Anthropic Claude 4.5 (Nov 2025)
    "opus": "anthropic:claude-opus-4-5",
    "sonnet": "anthropic:claude-sonnet-4-5",
    "haiku": "anthropic:claude-haiku-4-5",
    # OpenAI GPT-5.2 (Dec 2025) - latest
    "gpt-5.2": "openai:gpt-5.2",
    "gpt-5.2-pro": "openai:gpt-5.2-pro",
    # OpenAI GPT-5.1 (Nov 2025)
    "gpt-5.1": "openai:gpt-5.1",
    # OpenAI GPT-5 (Aug 2025)
    "gpt-5": "openai:gpt-5",
    "gpt-5-mini": "openai:gpt-5-mini",
    "gpt-5-nano": "openai:gpt-5-nano",
    # OpenAI reasoning models
    "o3": "openai:o3",
    "o4-mini": "openai:o4-mini",
    # Google Gemini 3 (Dec 2025)
    "gemini-3-pro": "google-gla:gemini-3-pro-preview",
    "gemini-3-flash": "google-gla:gemini-3-flash-preview",
    # Google Gemini 2.5
    "gemini-2.5-pro": "google-gla:gemini-2.5-pro",
    "gemini-2.5-flash": "google-gla:gemini-2.5-flash",
    # Google Gemini 2.0 (legacy)
    "gemini-2.0-flash": "google-gla:gemini-2.0-flash",
}

# For backward compatibility
MODEL_CONFIGS = MODEL_ALIASES

# Pricing per 1M tokens (as of Dec 2025)
# Format: "provider:model" -> {"input": $, "output": $}
MODEL_PRICING = {
    # Anthropic Claude 4.5
    "anthropic:claude-opus-4-5": {"input": 15.0, "output": 75.0},
    "anthropic:claude-sonnet-4-5": {"input": 3.0, "output": 15.0},
    "anthropic:claude-haiku-4-5": {"input": 1.0, "output": 5.0},
    # OpenAI GPT-5.2 (Dec 2025)
    "openai:gpt-5.2": {"input": 1.75, "output": 14.0},
    "openai:gpt-5.2-pro": {"input": 1.75, "output": 14.0},
    # OpenAI GPT-5.1 (Nov 2025)
    "openai:gpt-5.1": {"input": 1.25, "output": 10.0},
    # OpenAI GPT-5 family (Aug 2025)
    "openai:gpt-5": {"input": 1.25, "output": 10.0},
    "openai:gpt-5-mini": {"input": 0.25, "output": 2.0},
    "openai:gpt-5-nano": {"input": 0.05, "output": 0.40},
    # OpenAI reasoning models
    "openai:o3": {"input": 0.40, "output": 1.60},
    "openai:o4-mini": {"input": 1.10, "output": 4.40},
    # Google Gemini 3 (Dec 2025)
    "google-gla:gemini-3-pro-preview": {"input": 2.0, "output": 12.0},
    "google-gla:gemini-3-flash-preview": {"input": 0.50, "output": 3.0},
    # Google Gemini 2.5
    "google-gla:gemini-2.5-pro": {"input": 1.25, "output": 10.0},
    "google-gla:gemini-2.5-flash": {"input": 0.30, "output": 2.50},
    # Google Gemini 2.0
    "google-gla:gemini-2.0-flash": {"input": 0.10, "output": 0.40},
}


def get_model_cost(
    model: str, input_tokens: int, output_tokens: int
) -> float | None:
    """Calculate cost for given model and token counts."""
    resolved = resolve_model(model)
    if resolved not in MODEL_PRICING:
        return None
    pricing = MODEL_PRICING[resolved]
    return (
        (input_tokens / 1_000_000) * pricing["input"]
        + (output_tokens / 1_000_000) * pricing["output"]
    )


@dataclass
class TranslationStats:
    """Statistics from a translation run."""
    total_strings: int = 0
    translated: int = 0
    skipped_existing: int = 0
    skipped_format_only: int = 0
    errors: int = 0
    input_tokens: int = 0
    output_tokens: int = 0


class TranslationItem(BaseModel):
    """A single translation pair."""
    key: str = Field(description="Original string key (unchanged)")
    value: str = Field(description="Translated string value")


class TranslationResult(BaseModel):
    """Result from AI model for a batch of translations."""
    translations: list[TranslationItem] = Field(description="List of translated strings")

class OutputParseError(RuntimeError):
    """Raised when the model output cannot be parsed as valid JSON."""


def resolve_model(model: str) -> str:
    """
    Resolve a model shorthand or full name to provider:model format.

    Examples:
        "sonnet" -> "anthropic:claude-sonnet-4-5"
        "gpt-4o" -> "openai:gpt-4o"
        "anthropic:claude-sonnet-4-5" -> "anthropic:claude-sonnet-4-5" (unchanged)
    """
    # If already in provider:model format, return as-is
    if ":" in model:
        return model
    # Check aliases
    if model in MODEL_ALIASES:
        return MODEL_ALIASES[model]
    # Unknown model - return as-is (let pydantic-ai handle validation)
    return model


class XCStringsTranslator:
    """Translator for xcstrings files using AI models."""

    def __init__(
        self,
        model: str = "sonnet",
        batch_size: int = 25,
        concurrency: int = 32,
        app_context: str | None = None,
    ):
        """
        Initialize the translator.

        Args:
            model: Model to use. Can be:
                - Shorthand: opus, sonnet, haiku, gpt-4o, gemini-2.0-flash
                - Full format: anthropic:claude-sonnet-4-5, openai:gpt-4o
            batch_size: Number of strings to translate per API call
            concurrency: Max parallel API requests
            app_context: Optional context about the app for better translations
        """
        self.model = resolve_model(model)
        self.model_shorthand = model  # Keep original for display
        self.batch_size = batch_size
        self.concurrency = max(1, int(concurrency))
        self.app_context = app_context or "A mobile app. Tone: friendly, clear."
        self.stats = TranslationStats()
        self._stats_lock = threading.Lock()
    
    def translate_file(
        self,
        xcstrings: XCStringsFile,
        target_languages: list[str],
        overwrite: bool = False,
        progress_callback: callable | None = None,
    ) -> XCStringsFile:
        """
        Translate an xcstrings file to the target languages.
        
        Args:
            xcstrings: The xcstrings file to translate
            target_languages: List of language codes to translate to
            overwrite: If True, overwrite existing translations
            progress_callback: Optional callback for progress updates
            
        Returns:
            The updated xcstrings file with new translations
        """
        # Get translatable strings
        translatable = xcstrings.get_translatable_strings()
        self.stats.total_strings = len(translatable) * len(target_languages)

        context_by_key: dict[str, TranslationContext] = {}
        for key, entry in translatable:
            context_by_key[key] = self._build_context(key, entry)

        # Pre-build work items (batch translations) across all languages, so we can
        # run many API calls in parallel (useful for higher-rate-limit accounts).
        work_items: list[tuple[str, list[tuple[str, StringEntry, TranslationContext]]]] = []
        total_by_lang: dict[str, int] = {}
        completed_by_lang: dict[str, int] = {}

        for lang in target_languages:
            if lang not in SUPPORTED_LANGUAGES:
                raise ValueError(f"Unsupported language: {lang}. Supported: {list(SUPPORTED_LANGUAGES.keys())}")

            strings_to_translate: list[tuple[str, StringEntry, TranslationContext]] = []
            for key, entry in translatable:
                if not overwrite and lang in entry.localizations:
                    loc = entry.localizations[lang]
                    if loc.stringUnit and loc.stringUnit.value:
                        with self._stats_lock:
                            self.stats.skipped_existing += 1
                        continue

                context = context_by_key[key]
                strings_to_translate.append((key, entry, context))

            total_by_lang[lang] = len(strings_to_translate)
            completed_by_lang[lang] = 0

            for i in range(0, len(strings_to_translate), self.batch_size):
                batch = strings_to_translate[i : i + self.batch_size]
                if batch:
                    work_items.append((lang, batch))

        def translate_one(lang: str, batch: list[tuple[str, StringEntry, TranslationContext]]):
            translations = self._translate_batch(batch, lang)
            return lang, translations, len(batch)

        pending: deque[tuple[str, list[tuple[str, StringEntry, TranslationContext]]]] = deque(work_items)

        def split_batch(
            lang: str, batch: list[tuple[str, StringEntry, TranslationContext]]
        ) -> tuple[list[tuple[str, StringEntry, TranslationContext]], list[tuple[str, StringEntry, TranslationContext]]]:
            mid = max(1, len(batch) // 2)
            return batch[:mid], batch[mid:]

        # Submit work gradually so we only keep ~concurrency requests in-flight.
        # Parse failures are handled by splitting and re-queueing, which keeps the
        # UI responsive (progress continues moving) and avoids long "stuck" batches.
        with ThreadPoolExecutor(max_workers=self.concurrency) as executor:
            in_flight: dict[Any, tuple[str, list[tuple[str, StringEntry, TranslationContext]]]] = {}

            def submit_next() -> None:
                if not pending:
                    return
                lang, batch = pending.popleft()
                in_flight[executor.submit(translate_one, lang, batch)] = (lang, batch)

            for _ in range(min(self.concurrency, len(work_items))):
                submit_next()

            while in_flight:
                done, _pending = wait(in_flight.keys(), return_when=FIRST_COMPLETED)
                for future in done:
                    lang, batch = in_flight.pop(future)
                    try:
                        _lang, translations, batch_size = future.result()
                    except OutputParseError:
                        # Split and retry with smaller batches to reduce output size.
                        if len(batch) > 1:
                            left, right = split_batch(lang, batch)
                            if right:
                                pending.appendleft((lang, right))
                            if left:
                                pending.appendleft((lang, left))
                        else:
                            # Single item still unparseable: mark as an error and move on.
                            with self._stats_lock:
                                self.stats.errors += 1
                            completed_by_lang[lang] += 1
                            if progress_callback:
                                progress_callback(
                                    lang=lang,
                                    current=min(completed_by_lang[lang], total_by_lang[lang]),
                                    total=total_by_lang[lang] or 1,
                                    batch_size=1,
                                )
                        submit_next()
                        continue
                    except Exception as e:
                        # Cancel not-yet-started work; already-running requests can't be interrupted.
                        for f in in_flight.keys():
                            f.cancel()
                        keys = [k for k, _entry, _ctx in batch]
                        raise RuntimeError(
                            f"Translation failed for {lang} (batch size {len(batch)}). "
                            f"First key: {keys[0] if keys else '<empty>'}. Error: {e}"
                        ) from e

                    # Apply translations to xcstrings in the main thread.
                    requested_keys = {k for k, _entry, _ctx in batch}
                    for key, translation in translations.items():
                        if key in xcstrings.strings:
                            entry = xcstrings.strings[key]
                            entry.localizations[lang] = Localization(
                                stringUnit=StringUnit(state="translated", value=translation)
                            )
                            with self._stats_lock:
                                self.stats.translated += 1
                    missing = requested_keys.difference(translations.keys())
                    if missing:
                        with self._stats_lock:
                            self.stats.errors += len(missing)

                    completed_by_lang[lang] += batch_size
                    if progress_callback:
                        progress_callback(
                            lang=lang,
                            current=min(completed_by_lang[lang], total_by_lang[lang]),
                            total=total_by_lang[lang] or 1,
                            batch_size=batch_size,
                        )

                    submit_next()

        return xcstrings
    
    def _build_context(self, key: str, entry: StringEntry) -> TranslationContext:
        """Build translation context from existing translations."""
        existing = {}
        for lang, loc in entry.localizations.items():
            if loc.stringUnit and loc.stringUnit.value:
                existing[lang] = loc.stringUnit.value
        
        # Detect format specifiers
        format_specs = re.findall(r'%[\d$]*[@dlfse]|%lld|%%', key)
        
        return TranslationContext(
            key=key,
            comment=entry.comment,
            existing_translations=existing,
            has_format_specifiers=len(format_specs) > 0,
            format_specifiers=format_specs,
        )
    
    def _translate_batch(
        self,
        batch: list[tuple[str, StringEntry, TranslationContext]],
        target_lang: str,
    ) -> dict[str, str]:
        """Translate a batch of strings using AI model via pydantic-ai."""

        # Build the translation request
        strings_data = []
        for key, entry, context in batch:
            string_info = {
                "key": key,
                "existing": context.existing_translations,
            }
            if context.comment:
                string_info["comment"] = context.comment
            if context.format_specifiers:
                string_info["format_specifiers"] = context.format_specifiers
            strings_data.append(string_info)

        target_lang_name = SUPPORTED_LANGUAGES.get(target_lang, target_lang)

        system_prompt = f"""You are an expert iOS app translator. Your task is to translate UI strings for an iOS app.

APP CONTEXT:
{self.app_context}

TARGET LANGUAGE: {target_lang_name} ({target_lang})

CRITICAL RULES:
1. PRESERVE ALL FORMAT SPECIFIERS EXACTLY: %@, %lld, %d, %f, %1$@, %2$lld, etc.
   - These are placeholders that will be replaced with values at runtime
   - The order and format MUST match the original

2. USE NATURAL, NATIVE-SOUNDING TRANSLATIONS:
   - Don't translate word-for-word
   - Use natural phrasing for the target language
   - Match the tone: casual/friendly for user-facing text, technical for settings

3. MAINTAIN CONSISTENCY:
   - Look at existing translations (en/de) to understand meaning and tone
   - If a German translation exists, it shows the intended meaning
   - Keep UI element names consistent throughout

4. HANDLE SPECIAL CASES:
   - Keep brand names unchanged (NeatPass, Apple Wallet, etc.)
   - Keep technical terms if commonly used untranslated
   - Adapt date/time formats to locale conventions
   - Keep placeholder examples appropriate for locale

5. FOR PLURAL-SENSITIVE LANGUAGES:
   - Consider grammatical number agreement
   - Format specifiers like %lld indicate numbers"""

        user_message = f"""Translate these iOS app strings to {target_lang_name}:

```json
{json.dumps(strings_data, ensure_ascii=False, indent=2)}
```

Return the translations. Each item must have "key" (the original key unchanged) and "value" (the translation)."""

        try:
            # Create agent with structured output
            agent: Agent[None, TranslationResult] = Agent(
                self.model,
                output_type=TranslationResult,
                instructions=system_prompt,
            )

            # Run synchronously
            result = agent.run_sync(user_message)

            # Track token usage
            usage = result.usage()
            with self._stats_lock:
                self.stats.input_tokens += usage.request_tokens or 0
                self.stats.output_tokens += usage.response_tokens or 0

            # Extract translations from structured output
            translations: dict[str, str] = {}
            for item in result.output.translations:
                if item.key and item.value:
                    translations[item.key] = item.value

            return translations

        except Exception as e:
            error_str = str(e).lower()
            parse_related = (
                "parse" in error_str
                or "json" in error_str
                or "validation" in error_str
                or "invalid" in error_str
            )
            if parse_related:
                raise OutputParseError(str(e))

            with self._stats_lock:
                self.stats.errors += len(batch)
            raise RuntimeError(f"Translation failed: {e}")
    
    def estimate_cost(
        self,
        xcstrings: XCStringsFile,
        target_languages: list[str],
    ) -> dict[str, Any]:
        """
        Estimate the cost of translating the file.
        
        Returns dict with estimated tokens and cost.
        """
        translatable = xcstrings.get_translatable_strings()
        existing_langs = xcstrings.get_existing_languages()
        
        # Estimate strings per language
        strings_per_lang = {}
        for lang in target_languages:
            count = 0
            for key, entry in translatable:
                if lang not in entry.localizations:
                    count += 1
                elif not entry.localizations[lang].stringUnit:
                    count += 1
            strings_per_lang[lang] = count
        
        total_strings = sum(strings_per_lang.values())
        
        # Estimate tokens (rough: ~50 tokens per string including context)
        estimated_input_tokens = total_strings * 100  # Input with context
        estimated_output_tokens = total_strings * 30   # Output

        # Calculate cost based on model
        estimated_cost = get_model_cost(
            self.model, estimated_input_tokens, estimated_output_tokens
        )

        return {
            "total_strings": len(translatable),
            "strings_per_language": strings_per_lang,
            "total_to_translate": total_strings,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_cost_usd": estimated_cost,
            "model": self.model,
        }
