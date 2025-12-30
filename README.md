# XCStrings Translator

AI-powered translation tool for Apple's `Localizable.xcstrings` files. Supports Claude, GPT, and Gemini.

## What is xcstrings?

**Localizable.xcstrings** is Apple's modern localization format introduced in Xcode 15 (WWDC 2023). It replaces the traditional `.strings` and `.stringsdict` files with a single JSON-based file that supports:

- ✅ Multiple languages in one file
- ✅ Pluralization (one, zero, two, few, many, other)
- ✅ Device-specific variations (iPhone, iPad, Mac, Apple Watch)
- ✅ Translation state tracking (new, translated, needs_review)
- ✅ Comments for translator context
- ✅ Automatic extraction from SwiftUI code

### File Structure

```json
{
  "sourceLanguage": "en",
  "version": "1.0",
  "strings": {
    "Welcome to %@": {
      "comment": "Welcome message shown on launch",
      "localizations": {
        "en": {
          "stringUnit": {
            "state": "translated",
            "value": "Welcome to %@"
          }
        },
        "de": {
          "stringUnit": {
            "state": "translated",
            "value": "Willkommen bei %@"
          }
        }
      }
    }
  }
}
```

## Features

- 🤖 **Multi-provider AI** - Claude (Anthropic), GPT (OpenAI), or Gemini (Google)
- 🎯 **Context-aware** - Uses existing translations (EN/DE) to understand meaning
- 📝 **Format specifier preservation** - Correctly handles `%@`, `%lld`, `%1$@`, etc.
- 🌍 **35+ languages supported** - All major App Store languages
- 💰 **Cost estimation** - Know costs before translating
- ⚡ **Batch processing** - Efficient API usage with configurable batch sizes
- ✅ **Validation** - Check for translation issues and format mismatches

## Installation

```bash
pip install xcstrings-translator
```

Or install from source:

```bash
git clone https://github.com/yourusername/xcstrings-translator
cd xcstrings-translator
pip install -e .
```

## Setup

Set your API key for the provider you want to use:

```bash
# For Claude (default)
export ANTHROPIC_API_KEY=your-key-here

# For OpenAI
export OPENAI_API_KEY=your-key-here

# For Google Gemini
export GOOGLE_API_KEY=your-key-here
```

## Usage

### Basic Translation

```bash
# Translate to French, Spanish, and Italian
xcstrings translate Localizable.xcstrings -l fr,es,it

# Faster: run multiple batch requests in parallel (higher API tiers can go higher)
xcstrings translate Localizable.xcstrings -l fr,es,it --concurrency 32

# Output to a different file
xcstrings translate Localizable.xcstrings -l fr,es -o Translated.xcstrings
```

### Model Selection

```bash
# Claude (Anthropic) - default
xcstrings translate Localizable.xcstrings -l fr -m sonnet
xcstrings translate Localizable.xcstrings -l fr -m haiku   # faster/cheaper
xcstrings translate Localizable.xcstrings -l fr -m opus    # highest quality

# OpenAI
xcstrings translate Localizable.xcstrings -l fr -m gpt-5
xcstrings translate Localizable.xcstrings -l fr -m gpt-5-mini

# Google Gemini
xcstrings translate Localizable.xcstrings -l fr -m gemini-2.5-flash
xcstrings translate Localizable.xcstrings -l fr -m gemini-2.5-pro
```

### Cost Estimation

```bash
# Dry run - see what would be translated
xcstrings translate Localizable.xcstrings -l fr,es,it,ja,ko --dry-run

# Or use the estimate command
xcstrings estimate Localizable.xcstrings -l fr,es,it,ja,ko
```

### File Information

```bash
# Show file details and language coverage
xcstrings info Localizable.xcstrings
```

### Validation

```bash
# Check for issues (missing translations, format mismatches)
xcstrings validate Localizable.xcstrings
```

### List Supported Languages

```bash
xcstrings languages
```

## Commands

| Command | Description |
|---------|-------------|
| `translate` | Translate xcstrings file to new languages |
| `info` | Show file information and coverage |
| `estimate` | Estimate translation cost |
| `validate` | Check for common issues |
| `languages` | List supported languages |

## Options

### translate

| Option | Description | Default |
|--------|-------------|---------|
| `-l, --languages` | Comma-separated language codes | Required |
| `-o, --output` | Output file path | Overwrites input |
| `-m, --model` | AI model (see Model Selection) | sonnet |
| `-b, --batch-size` | Strings per API call | 25 |
| `-c, --concurrency` | Max parallel API requests | 32 |
| `--overwrite` | Overwrite existing translations | False |
| `--dry-run` | Show estimate without translating | False |
| `--context` | Custom app description | context.md or generic |

## Supported Languages

### European
- English (en)
- German (de)
- French (fr)
- Spanish (es)
- Italian (it)
- Portuguese (pt, pt-BR)
- Dutch (nl)
- Polish (pl)
- Swedish (sv)
- Danish (da)
- Norwegian Bokmål (nb)
- Finnish (fi)
- Czech (cs)
- Slovak (sk)
- Hungarian (hu)
- Romanian (ro)
- Bulgarian (bg)
- Greek (el)
- Albanian (sq, sq-AL)
- Ukrainian (uk)
- Russian (ru)
- Turkish (tr)

### Asian
- Japanese (ja)
- Korean (ko)
- Simplified Chinese (zh-Hans)
- Traditional Chinese (zh-Hant)
- Thai (th)
- Vietnamese (vi)
- Indonesian (id)
- Malay (ms)
- Hindi (hi)

### Middle Eastern
- Arabic (ar)
- Hebrew (he)

### Other
- Catalan (ca)
- Basque (eu)

## Cost Reference (Dec 2025)

Per 1M tokens:

| Provider | Model | Input | Output |
|----------|-------|-------|--------|
| Anthropic | haiku | $1.00 | $5.00 |
| Anthropic | sonnet | $3.00 | $15.00 |
| Anthropic | opus | $15.00 | $75.00 |
| OpenAI | gpt-5-nano | $0.05 | $0.40 |
| OpenAI | gpt-5-mini | $0.25 | $2.00 |
| OpenAI | gpt-5 | $1.25 | $10.00 |
| Google | gemini-2.0-flash | $0.10 | $0.40 |
| Google | gemini-2.5-flash | $0.30 | $2.50 |
| Google | gemini-2.5-pro | $1.25 | $10.00 |

Typical cost for ~200 strings to 5 languages:
- gpt-5-nano: ~$0.01
- gemini-2.0-flash: ~$0.02
- haiku: ~$0.06
- sonnet: ~$0.20

## Python API

```python
from xcstrings_translator import XCStringsFile, XCStringsTranslator

# Load file
xcstrings = XCStringsFile.from_file("Localizable.xcstrings")

# Check existing languages
print(xcstrings.get_existing_languages())  # {'en', 'de'}

# Initialize translator
translator = XCStringsTranslator(
    model="sonnet",
    batch_size=25,
    app_context="My app description for better translations"
)

# Estimate cost
estimate = translator.estimate_cost(xcstrings, ["fr", "es", "it"])
print(f"Estimated cost: ${estimate['estimated_cost_usd']:.2f}")

# Translate
xcstrings = translator.translate_file(
    xcstrings,
    target_languages=["fr", "es", "it"],
    overwrite=False,  # Skip existing translations
)

# Save
xcstrings.to_file("Localizable.xcstrings")

# Check stats
print(f"Translated: {translator.stats.translated}")
print(f"Tokens used: {translator.stats.input_tokens + translator.stats.output_tokens}")
```

## Best Practices

### 1. Always estimate first

```bash
xcstrings translate input.xcstrings -l fr,es,it,ja,ko --dry-run
```

### 2. Start with Haiku for testing

```bash
xcstrings translate input.xcstrings -l fr -m haiku
# Verify quality, then use Sonnet for production
```

### 3. Translate similar languages together

Languages in the same family often share context:
```bash
# Romance languages
xcstrings translate input.xcstrings -l fr,es,it,pt

# Germanic languages  
xcstrings translate input.xcstrings -l nl,sv,da,no

# CJK languages
xcstrings translate input.xcstrings -l ja,ko,zh-Hans,zh-Hant
```

### 4. Provide app context

Create a `context.md` file next to your xcstrings file:

```markdown
# My Fitness App

A fitness tracking app for athletes.

Tone: Motivational, energetic, supportive.
Use informal "you" forms (du/tu/etc).
```

Or pass context directly:

```bash
xcstrings translate input.xcstrings -l fr \
  --context "A fitness tracking app with motivational tone"
```

Priority: `--context` flag > `context.md` file > generic default

### 5. Validate after translation

```bash
xcstrings validate Localizable.xcstrings
```

## Integration with Xcode

### Adding new languages

1. In Xcode, select your project
2. Go to **Info** → **Localizations**
3. Click **+** to add new languages
4. Xcode will update your xcstrings file

### After translation

1. Replace your `Localizable.xcstrings` with the translated file
2. Build your project - Xcode will validate the file
3. Test in Simulator with different language settings

### Testing localizations

```bash
# Run app in French
xcrun simctl boot "iPhone 15"
xcrun simctl spawn booted defaults write com.apple.Preferences AppleLanguages -array fr
```

## Troubleshooting

### "Unsupported language" error

Check the language code matches Apple's format:
- Use `zh-Hans` not `zh-CN`
- Use `pt-BR` not `pt_BR`
- Use `sq` (or `sq-AL`) not `al`
- Common country codes like `BR`, `CN`, `DK`, `NO`, `SE` normalize automatically

### Format specifier mismatch

Run validation to find issues:
```bash
xcstrings validate Localizable.xcstrings
```

### API errors

- Check the correct API key is set for your model:
  - Claude: `ANTHROPIC_API_KEY`
  - OpenAI: `OPENAI_API_KEY`
  - Gemini: `GOOGLE_API_KEY`
- Verify you have API credits
- Try reducing batch size: `-b 10`

## File Format Reference

### String states

- `new` - Needs translation
- `translated` - Has been translated
- `needs_review` - Translation may be outdated
- `stale` - Source string no longer in code

### Extraction states

- `manual` - Manually added
- `extracted_with_value` - Extracted from code with value
- `stale` - No longer found in code

## License

MIT License - Use freely for your projects.

## Contributing

Issues and PRs welcome! Please test with real xcstrings files before submitting.
