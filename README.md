# XCStrings Translator

Translate Apple's `Localizable.xcstrings` files using AI. Supports **Claude**, **GPT**, and **Gemini**.

```bash
xcstrings translate Localizable.xcstrings -l fr,es,it,ja,ko
```

## Features

- **Multi-provider** - Choose Claude, GPT, or Gemini based on cost/quality needs
- **Context-aware** - Uses existing translations to understand meaning and tone
- **Format-safe** - Preserves `%@`, `%lld`, `%1$@` and other specifiers
- **35+ languages** - All major App Store locales
- **Cost estimation** - Preview costs before translating
- **Validation** - Detect missing translations and format mismatches

## Quick Start

```bash
# Install
pip install xcstrings-translator

# Set API key (pick one)
export ANTHROPIC_API_KEY=sk-...   # Claude
export OPENAI_API_KEY=sk-...      # GPT
export GOOGLE_API_KEY=...         # Gemini

# Translate
xcstrings translate Localizable.xcstrings -l fr,de,ja
```

## Models

| Model | Provider | Cost/1M tokens | Best for |
|-------|----------|----------------|----------|
| `haiku` | Anthropic | $1 in / $5 out | Fast iteration |
| `sonnet` | Anthropic | $3 in / $15 out | **Recommended** |
| `opus` | Anthropic | $15 in / $75 out | Highest quality |
| `gpt-5-nano` | OpenAI | $0.05 in / $0.40 out | Cheapest |
| `gpt-5-mini` | OpenAI | $0.25 in / $2 out | Budget |
| `gpt-5` | OpenAI | $1.25 in / $10 out | Quality |
| `gemini-2.0-flash` | Google | $0.10 in / $0.40 out | Fast & cheap |
| `gemini-3-flash` | Google | $0.50 in / $3 out | Balanced |
| `gemini-3-pro` | Google | $2 in / $12 out | Quality |

```bash
xcstrings translate input.xcstrings -l fr -m haiku      # fast/cheap
xcstrings translate input.xcstrings -l fr -m sonnet     # balanced (default)
xcstrings translate input.xcstrings -l fr -m gpt-5-nano # cheapest
```

## Commands

| Command | Description |
|---------|-------------|
| `translate` | Translate to specified languages |
| `estimate` | Preview cost without translating |
| `info` | Show file stats and coverage |
| `validate` | Check for issues |
| `languages` | List all 35+ supported languages |

## Usage Examples

```bash
# Translate to multiple languages
xcstrings translate Localizable.xcstrings -l fr,es,it,de,ja

# Estimate cost first
xcstrings translate Localizable.xcstrings -l fr,es,it --dry-run

# Use faster model
xcstrings translate Localizable.xcstrings -l fr -m haiku

# Parallel requests (faster for large files)
xcstrings translate Localizable.xcstrings -l fr,es -c 32

# Save to different file
xcstrings translate input.xcstrings -l fr -o translated.xcstrings

# Fill missing translations only
xcstrings translate Localizable.xcstrings --fill-missing

# Check translation coverage
xcstrings info Localizable.xcstrings

# Validate format specifiers
xcstrings validate Localizable.xcstrings
```

## App Context

Provide context for better translations. Create `context.md` next to your xcstrings file:

```markdown
# My Fitness App

A workout tracking app for athletes.

Tone: Motivational, energetic.
Use informal "you" (du/tu).
```

Or pass directly:

```bash
xcstrings translate input.xcstrings -l fr --context "A fitness app with motivational tone"
```

## Python API

```python
from src import XCStringsFile, XCStringsTranslator

# Load
xcstrings = XCStringsFile.from_file("Localizable.xcstrings")

# Translate
translator = XCStringsTranslator(model="sonnet")
xcstrings = translator.translate_file(xcstrings, ["fr", "es", "it"])

# Save
xcstrings.to_file("Localizable.xcstrings")

# Stats
print(f"Translated: {translator.stats.translated}")
print(f"Cost: ${translator.stats.input_tokens * 3 / 1e6 + translator.stats.output_tokens * 15 / 1e6:.2f}")
```

## Supported Languages

**European:** en, de, fr, es, it, pt, pt-BR, nl, pl, sv, da, nb, fi, cs, sk, hu, ro, bg, el, sq, uk, ru, tr

**Asian:** ja, ko, zh-Hans, zh-Hant, th, vi, id, ms, hi

**Middle Eastern:** ar, he

**Other:** ca, eu

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `-l, --languages` | Target language codes | Required |
| `-m, --model` | AI model | sonnet |
| `-o, --output` | Output file | Overwrites input |
| `-b, --batch-size` | Strings per API call | 25 |
| `-c, --concurrency` | Parallel requests | 32 |
| `--context` | App description | context.md |
| `--overwrite` | Replace existing | false |
| `--dry-run` | Estimate only | false |
| `-f, --fill-missing` | Auto-detect languages | false |

## Troubleshooting

**"Unsupported language"** - Use Apple codes: `zh-Hans` not `zh-CN`, `pt-BR` not `pt_BR`

**Format mismatch** - Run `xcstrings validate` to find issues

**API errors** - Check correct key is set for your model

## License

MIT

---

<p align="center"><i>Made with ❤️ by <a href="https://lanfermann.dev">Justin Lanfermann</a></i></p>
