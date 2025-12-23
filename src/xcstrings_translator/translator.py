"""
Claude-powered translator for xcstrings files.

Uses Claude to translate strings intelligently by:
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
import time
import random
from json import JSONDecodeError
from collections import deque

from anthropic import Anthropic
from pydantic import BaseModel, Field

from .models import (
    XCStringsFile,
    StringEntry,
    Localization,
    StringUnit,
    TranslationContext,
    SUPPORTED_LANGUAGES,
)


# Model configurations
MODEL_CONFIGS = {
    # Snapshot IDs from Anthropic docs (Models overview) as of 2025-12-13.
    # Aliases (without the trailing date) automatically migrate to newer snapshots.
    "opus": "claude-opus-4-5-20251101",
    "sonnet": "claude-sonnet-4-5-20250929",
    "haiku": "claude-haiku-4-5-20251001",
}

MODEL_ALIASES = {
    "opus": "claude-opus-4-5",
    "sonnet": "claude-sonnet-4-5",
    "haiku": "claude-haiku-4-5",
}


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


class TranslationResult(BaseModel):
    """Result from Claude for a batch of translations."""
    translations: list[dict[str, str]]

class OutputParseError(RuntimeError):
    """Raised when the model output cannot be parsed as valid JSON."""


class XCStringsTranslator:
    """Translator for xcstrings files using Claude."""
    
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
            model: Claude model to use (opus, sonnet, haiku)
            batch_size: Number of strings to translate per API call
            app_context: Optional context about the app for better translations
        """
        self.client = Anthropic()
        self.model = MODEL_CONFIGS.get(model, model)
        self.model_alias = MODEL_ALIASES.get(model, self._snapshot_to_alias(self.model))
        self.batch_size = batch_size
        self.concurrency = max(1, int(concurrency))
        self.app_context = app_context or (
            "An iOS app that converts PDF tickets and documents into Apple Wallet passes. "
            "Tone: friendly, clear, reassuring about privacy."
        )
        self.stats = TranslationStats()
        self._stats_lock = threading.Lock()
        self._model_lock = threading.Lock()

    @staticmethod
    def _snapshot_to_alias(model_id: str) -> str | None:
        """
        Convert a snapshot model ID like 'claude-sonnet-4-5-20250929' to its alias
        'claude-sonnet-4-5'. Returns None if it doesn't look like a snapshot ID.
        """
        match = re.match(r"^(claude-(?:opus|sonnet|haiku)-4-5)-\d{8}$", model_id)
        return match.group(1) if match else None
    
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
            # Use a per-call client to avoid thread-safety issues.
            client = Anthropic()
            translations = self._translate_batch(batch, lang, client=client)
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
        client: Anthropic | None = None,
    ) -> dict[str, str]:
        """Translate a batch of strings using Claude."""
        
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
        
        system_prompt = f"""You are an expert iOS app translator. Your task is to translate UI strings for an iOS app called NeatPass.

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
   - Format specifiers like %lld indicate numbers

OUTPUT FORMAT:
Return a JSON object with "translations" array containing objects with "key" and "value" fields.
ONLY return the JSON, no explanation or markdown.
The JSON MUST be strictly valid: escape quotes/newlines inside strings."""

        user_message = f"""Translate these iOS app strings to {target_lang_name}:

```json
{json.dumps(strings_data, ensure_ascii=False, indent=2)}
```

        Return JSON with translations array. Each item must have "key" (the original key unchanged) and "value" (the translation)."""

        client = client or self.client
        model_id = self.model
        model_alias = self.model_alias

        def _create_message(model: str):
            kwargs = dict(
                model=model,
                max_tokens=4096,
                messages=[{"role": "user", "content": user_message}],
                system=system_prompt,
            )
            # Try JSON mode when supported by the SDK.
            try:
                return client.messages.create(**kwargs, response_format={"type": "json_object"})
            except TypeError:
                return client.messages.create(**kwargs)

        def _extract_json(text: str) -> str:
            stripped = (text or "").strip()
            if stripped.startswith("```json"):
                stripped = stripped[7:]
            if stripped.startswith("```"):
                stripped = stripped[3:]
            if stripped.endswith("```"):
                stripped = stripped[:-3]
            stripped = stripped.strip()
            if stripped.startswith("{") and stripped.endswith("}"):
                return stripped
            start = stripped.find("{")
            end = stripped.rfind("}")
            if start != -1 and end != -1 and end > start:
                return stripped[start : end + 1].strip()
            return stripped

        def _repair_json(invalid_json: str, error_text: str) -> str:
            repair_system = "You fix invalid JSON into strict JSON. Output ONLY the corrected JSON object."
            repair_user = (
                "The following was supposed to be a JSON object with a top-level key "
                "\"translations\" containing an array of objects with string fields \"key\" and \"value\".\n\n"
                f"JSON parse error: {error_text}\n\n"
                "Fix the JSON to be strictly valid. Do not change any keys; preserve all values as-is "
                "(only add escaping/quotes/commas/brackets as needed).\n\n"
                f"INVALID JSON:\n{invalid_json}"
            )
            kwargs = dict(
                model=model_id,
                max_tokens=4096,
                messages=[{"role": "user", "content": repair_user}],
                system=repair_system,
            )
            try:
                response = client.messages.create(**kwargs, response_format={"type": "json_object"})
            except TypeError:
                response = client.messages.create(**kwargs)

            usage = getattr(response, "usage", None)
            if usage is not None:
                with self._stats_lock:
                    self.stats.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                    self.stats.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

            blocks = getattr(response, "content", None) or []
            text_parts: list[str] = []
            for block in blocks:
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str):
                    text_parts.append(block_text)
            return _extract_json("\n".join(text_parts).strip())

        try:
            try:
                response = _create_message(model_id)
            except Exception as e:
                # If a pinned snapshot is unavailable (404), retry once with the alias.
                # This avoids breaking when Anthropic retires or migrates snapshots.
                error_text = str(e)
                if (
                    model_alias
                    and model_alias != model_id
                    and ("not_found_error" in error_text or "404" in error_text)
                ):
                    response = _create_message(model_alias)
                    with self._model_lock:
                        self.model = model_alias
                else:
                    raise

            # Track token usage (guarded for older/alternate SDK responses).
            usage = getattr(response, "usage", None)
            if usage is not None:
                with self._stats_lock:
                    self.stats.input_tokens += int(getattr(usage, "input_tokens", 0) or 0)
                    self.stats.output_tokens += int(getattr(usage, "output_tokens", 0) or 0)

            # Extract text content (some SDKs return multiple blocks).
            blocks = getattr(response, "content", None) or []
            text_parts: list[str] = []
            for block in blocks:
                block_text = getattr(block, "text", None)
                if isinstance(block_text, str):
                    text_parts.append(block_text)
            content = _extract_json("\n".join(text_parts).strip())
            if not content:
                raise ValueError("Model returned empty content")

            parse_attempts = 0
            last_parse_error: Exception | None = None
            result: dict[str, Any] | None = None
            while parse_attempts < 2:
                try:
                    result = json.loads(content)
                    break
                except Exception as pe:
                    last_parse_error = pe
                    parse_attempts += 1
                    # Attempt a lightweight repair via the model, then re-parse.
                    content = _repair_json(content, str(pe))
                    # Tiny backoff to avoid hammering in tight loops during parallel runs.
                    time.sleep(0.1 + random.random() * 0.2)

            if result is None:
                raise OutputParseError(str(last_parse_error or "Failed to parse JSON"))

            raw_translations = result.get("translations")
            if raw_translations is None:
                raw_translations = []
            if not isinstance(raw_translations, list):
                raise ValueError("Invalid response: 'translations' must be a list")

            translations: dict[str, str] = {}
            for item in raw_translations:
                if not isinstance(item, dict):
                    continue
                key = item.get("key")
                value = item.get("value")
                if isinstance(key, str) and isinstance(value, str) and key and value:
                    translations[key] = value

            return translations
        except Exception as e:
            if isinstance(e, OutputParseError):
                raise

            parse_related = isinstance(e, (JSONDecodeError, ValueError)) and (
                "parse" in str(e).lower()
                or "json" in str(e).lower()
                or "delimiter" in str(e).lower()
                or "expecting" in str(e).lower()
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
        
        # Cost per 1M tokens (as of 2025-12-13, Claude 4.5 pricing in docs)
        costs = {
            "opus": {"input": 5.0, "output": 25.0},
            "sonnet": {"input": 3.0, "output": 15.0},
            "haiku": {"input": 1.0, "output": 5.0},
        }
        
        model_name = None
        for name, model_id in MODEL_CONFIGS.items():
            if model_id == self.model:
                model_name = name
                break
        
        if model_name and model_name in costs:
            cost = costs[model_name]
            estimated_cost = (
                (estimated_input_tokens / 1_000_000) * cost["input"] +
                (estimated_output_tokens / 1_000_000) * cost["output"]
            )
        else:
            estimated_cost = None
        
        return {
            "total_strings": len(translatable),
            "strings_per_language": strings_per_lang,
            "total_to_translate": total_strings,
            "estimated_input_tokens": estimated_input_tokens,
            "estimated_output_tokens": estimated_output_tokens,
            "estimated_cost_usd": estimated_cost,
            "model": self.model,
        }
