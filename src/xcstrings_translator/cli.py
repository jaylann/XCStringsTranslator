"""
CLI interface for xcstrings translator.

Usage:
    xcstrings translate input.xcstrings -l fr,es,it -o output.xcstrings
    xcstrings translate input.xcstrings -l fr --model opus
    xcstrings info input.xcstrings
    xcstrings languages
    xcstrings estimate input.xcstrings -l fr,es,it,ja,ko
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Annotated

import readchar
import typer
from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.prompt import IntPrompt, Prompt
from rich.table import Table

from .models import SUPPORTED_LANGUAGES, XCStringsFile
from .translator import (
    MODEL_ALIASES,
    MODEL_PRICING,
    XCStringsTranslator,
    get_model_cost,
    resolve_model,
)

# Load .env from the current working directory (searches upward) so that keys
# saved by the interactive setup are picked up on subsequent runs.
load_dotenv()

app = typer.Typer(
    name="xcstrings",
    help="Translate Apple Localizable.xcstrings files using AI (Anthropic/OpenAI/Gemini)",
    add_completion=False,
)
console = Console()

# Provider metadata used by the interactive API-key setup. Ordered for the menu.
PROVIDERS = [
    {
        "key": "anthropic",
        "label": "Anthropic (Claude)",
        "env": "ANTHROPIC_API_KEY",
        "default_model": "sonnet",
        "url": "https://console.anthropic.com/",
    },
    {
        "key": "openai",
        "label": "OpenAI (GPT)",
        "env": "OPENAI_API_KEY",
        "default_model": "gpt-5.4",
        "url": "https://platform.openai.com/api-keys",
    },
    {
        "key": "google-gla",
        "label": "Google (Gemini)",
        "env": "GEMINI_API_KEY",
        "default_model": "gemini-2.5-flash",
        "url": "https://aistudio.google.com/apikey",
    },
    {
        "key": "openrouter",
        "label": "OpenRouter",
        "env": "OPENROUTER_API_KEY",
        "default_model": "or-sonnet",
        "url": "https://openrouter.ai/keys",
    },
]
PROVIDER_BY_KEY = {p["key"]: p for p in PROVIDERS}


def _provider_for_model(resolved: str) -> str:
    """Provider key (e.g. 'anthropic') for a resolved 'provider:model' string."""
    return resolved.split(":", 1)[0]


def _save_env_key(env_var: str, value: str) -> None:
    """Persist a key to ./.env and export it for the current process."""
    os.environ[env_var] = value
    env_path = Path.cwd() / ".env"
    set_key(str(env_path), env_var, value)


def _provider_menu_panel(selected: int) -> Panel:
    """Render the provider picker with the current row highlighted."""
    table = Table(show_header=False, box=None, pad_edge=False)
    table.add_column(width=2)
    table.add_column("Provider", no_wrap=True)
    table.add_column("Default model")
    for i, p in enumerate(PROVIDERS):
        if i == selected:
            table.add_row(
                "[cyan]❯[/cyan]",
                f"[bold cyan]{p['label']}[/bold cyan]",
                f"[cyan]{p['default_model']}[/cyan]",
            )
        else:
            table.add_row("", p["label"], f"[green]{p['default_model']}[/green]")
    return Panel(
        table,
        title="[bold]No API key found — choose a provider[/bold]",
        subtitle="[dim]↑/↓ move · enter select[/dim]",
        border_style="cyan",
        padding=(1, 2),
    )


def _render_provider_menu_numbered() -> dict:
    """Fallback numbered picker for non-TTY environments (e.g. tests, pipes)."""
    table = Table(show_header=True, header_style="bold cyan", box=None, pad_edge=False)
    table.add_column("#", style="cyan", justify="right")
    table.add_column("Provider", style="bold")
    table.add_column("Default model", style="green")
    for i, p in enumerate(PROVIDERS, start=1):
        table.add_row(str(i), p["label"], p["default_model"])
    console.print(
        Panel(
            table,
            title="[bold]No API key found — choose a provider[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    choice = IntPrompt.ask(
        "[cyan]Select provider[/cyan]",
        choices=[str(i) for i in range(1, len(PROVIDERS) + 1)],
        default=1,
    )
    return PROVIDERS[choice - 1]


def _render_provider_menu() -> dict:
    """
    Arrow-key provider picker. Use ↑/↓ (or j/k) to move, Enter to select; the
    number keys jump straight to a row. Falls back to a numbered prompt when
    stdin is not an interactive TTY.
    """
    if not sys.stdin.isatty():
        return _render_provider_menu_numbered()

    selected = 0
    with Live(
        _provider_menu_panel(selected),
        console=console,
        auto_refresh=False,
        transient=False,
    ) as live:
        while True:
            key = readchar.readkey()
            if key in (readchar.key.UP, "k"):
                selected = (selected - 1) % len(PROVIDERS)
            elif key in (readchar.key.DOWN, "j"):
                selected = (selected + 1) % len(PROVIDERS)
            elif key in (readchar.key.ENTER, "\r", "\n"):
                break
            elif key in (readchar.key.CTRL_C, "\x03"):
                raise KeyboardInterrupt
            elif key.isdigit() and 1 <= int(key) <= len(PROVIDERS):
                selected = int(key) - 1
                live.update(_provider_menu_panel(selected), refresh=True)
                break
            live.update(_provider_menu_panel(selected), refresh=True)
    console.print(f"[green]✓[/green] {PROVIDERS[selected]['label']}\n")
    return PROVIDERS[selected]


def _prompt_api_key(provider: dict) -> None:
    """Show a clean input box for the provider's API key and save it."""
    body = (
        f"[bold]{provider['label']}[/bold] needs an API key.\n\n"
        f"Environment variable: [cyan]{provider['env']}[/cyan]\n"
        f"Get a key at: [blue underline]{provider['url']}[/blue underline]\n\n"
        f"It will be saved to [cyan].env[/cyan] for future runs."
    )
    console.print(
        Panel(
            body,
            title="[bold]API key required[/bold]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    key = ""
    while not key.strip():
        key = Prompt.ask(
            f"[cyan]Enter your {provider['label']} API key[/cyan]", password=True
        )
        if not key.strip():
            console.print("[yellow]Key cannot be empty.[/yellow]")
    _save_env_key(provider["env"], key.strip())
    console.print("[green]✓[/green] Saved to .env\n")


def _ensure_provider_and_key(
    model: str | None, *, require_key: bool = True
) -> tuple[str, str]:
    """
    Resolve the model and make sure the matching provider key is available,
    prompting interactively when running in a TTY.

    Returns the concrete ``(model, resolved)`` to use for translation.
    """
    interactive = require_key and sys.stdin.isatty()

    # No model specified: use the default provider if its key is present,
    # otherwise offer the provider menu (interactive only).
    if model is None:
        if os.environ.get("ANTHROPIC_API_KEY") or not interactive:
            return "sonnet", resolve_model("sonnet")
        provider = _render_provider_menu()
        if not os.environ.get(provider["env"]):
            _prompt_api_key(provider)
        model = provider["default_model"]
        return model, resolve_model(model)

    # Model specified: derive the provider from it and prompt for that key.
    resolved = resolve_model(model)
    provider_key = _provider_for_model(resolved)
    provider = PROVIDER_BY_KEY.get(provider_key)
    if provider is None:
        # Unknown provider: let pydantic-ai handle validation downstream.
        return model, resolved
    if not os.environ.get(provider["env"]) and interactive:
        _prompt_api_key(provider)
    return model, resolved


def _canonicalize_bcp47_tag(tag: str) -> str:
    """
    Canonicalize common BCP 47-ish tags to the casing Apple typically uses.

    Examples:
      - pt_br -> pt-BR
      - zh-hans -> zh-Hans
      - sq_al -> sq-AL
    """
    tag = (tag or "").strip().replace("_", "-")
    if not tag:
        return tag

    parts = [p for p in tag.split("-") if p]
    if not parts:
        return tag

    out: list[str] = []
    for i, part in enumerate(parts):
        if i == 0:
            out.append(part.lower())
            continue

        if len(part) == 4 and part.isalpha():
            out.append(part.title())
            continue

        if (len(part) == 2 and part.isalpha()) or (len(part) == 3 and part.isdigit()):
            out.append(part.upper())
            continue

        out.append(part)

    return "-".join(out)


def _normalize_language_tag(raw: str) -> str:
    """
    Normalize user input to a supported Apple locale code.

    This prevents accidentally writing non-standard locale keys into the xcstrings file.
    """
    raw_tag = (raw or "").strip()
    if len(raw_tag) == 2 and raw_tag.isalpha() and raw_tag.isupper():
        # If the user passes an ISO-3166 country code (common mistake), map to a
        # sensible default language code for that region.
        country_code_aliases = {
            "AL": "sq",
            "BR": "pt-BR",
            "CN": "zh-Hans",
            "DK": "da",
            "NO": "nb",
            "SE": "sv",
        }
        if raw_tag in country_code_aliases:
            return country_code_aliases[raw_tag]

    tag = _canonicalize_bcp47_tag(raw)

    # Common Apple/BCP-47 aliases users try.
    aliases = {
        # Users often (incorrectly) think this tool wants country codes.
        "al": "sq",
        "no": "nb",
        # Common locale aliases (also documented in README).
        "zh-CN": "zh-Hans",
        "zh-SG": "zh-Hans",
        "zh-TW": "zh-Hant",
        "zh-HK": "zh-Hant",
        "zh-MO": "zh-Hant",
    }

    if tag in aliases:
        return aliases[tag]

    # Case-insensitive fallback for 2-letter "country code" inputs like AL.
    if len(tag) == 2 and tag.isalpha():
        lower = tag.lower()
        if lower in aliases:
            return aliases[lower]

    return tag


def _parse_target_languages(languages: str) -> list[str]:
    raw_langs = [lang.strip() for lang in (languages or "").split(",") if lang.strip()]
    return [_normalize_language_tag(lang) for lang in raw_langs]


# Directories that should never be searched when discovering nested files
# (worktrees, build artifacts, dependencies, VCS metadata).
_EXCLUDE_DIRS = {
    ".git",
    ".claude",
    "build",
    "DerivedData",
    "Pods",
    ".build",
    "node_modules",
}


def _discover_xcstrings(root: Path) -> list[Path]:
    """Recursively find .xcstrings files under root, skipping hidden/build dirs."""
    return sorted(
        p
        for p in root.rglob("*.xcstrings")
        if not any(
            part in _EXCLUDE_DIRS or part.startswith(".")
            for part in p.relative_to(root).parts[:-1]
        )
    )


def _translate_one_file(
    input_file: Path,
    *,
    languages: list[str] | None,
    model: str,
    resolved: str,
    batch_size: int,
    concurrency: int,
    overwrite: bool,
    dry_run: bool,
    app_context: str | None,
    fill_missing: bool,
    fetch_live_pricing: bool = True,
) -> bool:
    """
    Translate a single .xcstrings file. Returns True on success (or skip), False on
    a handled failure. ``languages`` is the validated target list for explicit mode,
    or None when ``fill_missing`` auto-detects per file.
    """
    console.print(f"\n[cyan]Loading:[/cyan] {input_file}")
    try:
        xcstrings = XCStringsFile.from_file(str(input_file))
    except Exception as e:
        console.print(f"[red]Error loading file:[/red] {e}")
        return False

    if fill_missing:
        target_langs = sorted(xcstrings.get_languages_with_localizations())
        if not target_langs:
            console.print(
                "[yellow]Skipped:[/yellow] no existing translations to fill "
                "(only source language present)"
            )
            return True
    else:
        target_langs = languages or []

    # Show file info
    existing_langs = xcstrings.get_existing_languages()
    translatable = xcstrings.get_translatable_strings()

    console.print(f"[green]✓[/green] Loaded {len(xcstrings.strings)} total strings")
    console.print(f"[green]✓[/green] {len(translatable)} translatable strings")
    console.print(
        f"[green]✓[/green] Existing languages: {', '.join(sorted(existing_langs)) or 'none'}"
    )
    console.print(f"[green]✓[/green] Target languages: {', '.join(target_langs)}")
    if fill_missing:
        console.print("[green]✓[/green] Mode: Fill missing translations only")
    console.print(f"[green]✓[/green] Model: {resolved}")

    # Load app context: --context flag > context.md file > neutral default
    file_context = app_context
    if not file_context:
        context_file = input_file.parent / "context.md"
        if context_file.exists():
            try:
                file_context = context_file.read_text().strip()
                console.print(
                    f"[green]✓[/green] Loaded context from {context_file.name}"
                )
            except OSError as e:
                console.print(
                    f"[yellow]![/yellow] Could not read {context_file.name}: {e}"
                )
        if not file_context:
            file_context = "A mobile app. Tone: friendly, clear."

    # Initialize translator
    translator = XCStringsTranslator(
        model=model,
        batch_size=batch_size,
        concurrency=concurrency,
        app_context=file_context,
        fetch_live_pricing=fetch_live_pricing,
    )

    # Dry run - just show estimate
    if dry_run:
        estimate = translator.estimate_cost(xcstrings, target_langs)

        console.print("\n[yellow]Dry run - no changes will be made[/yellow]\n")

        table = Table(title="Translation Estimate")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total translatable strings", str(estimate["total_strings"]))
        table.add_row("Strings to translate", str(estimate["total_to_translate"]))
        table.add_row(
            "Estimated input tokens", f"{estimate['estimated_input_tokens']:,}"
        )
        table.add_row(
            "Estimated output tokens", f"{estimate['estimated_output_tokens']:,}"
        )
        if estimate["estimated_cost_usd"]:
            table.add_row("Estimated cost", f"${estimate['estimated_cost_usd']:.2f}")

        console.print(table)

        if estimate["strings_per_language"]:
            console.print("\n[cyan]Strings per language:[/cyan]")
            for lang, count in estimate["strings_per_language"].items():
                lang_name = SUPPORTED_LANGUAGES.get(lang, lang)
                console.print(f"  {lang}: {count} strings ({lang_name})")

        return True

    console.print(f"\n[cyan]Translating to {len(target_langs)} language(s)...[/cyan]\n")

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        task_ids: dict[str, int] = {}

        def progress_callback(lang: str, current: int, total: int, batch_size: int):
            task_id = task_ids.get(lang)
            if task_id is None:
                task_id = progress.add_task(f"[cyan]{lang}[/cyan]", total=total)
                task_ids[lang] = task_id
            progress.update(
                task_id,
                total=total,
                completed=current,
                description=f"[cyan]{lang}[/cyan] ({current}/{total})",
            )

        # Add initial task
        for lang in target_langs:
            strings_for_lang = sum(
                1
                for key, entry in translatable
                if lang not in entry.localizations
                or not entry.localizations[lang].stringUnit
            )
            task_ids[lang] = progress.add_task(
                f"[cyan]{lang}[/cyan]", total=strings_for_lang or 1
            )

        try:
            xcstrings = translator.translate_file(
                xcstrings,
                target_langs,
                overwrite=overwrite,
                progress_callback=progress_callback,
            )
        except Exception as e:
            console.print(f"\n[red]Translation error:[/red] {e}")
            return False

    # Save the result (always in place; directory mode forbids -o)
    console.print(f"\n[cyan]Saving:[/cyan] {input_file}")
    try:
        xcstrings.to_file(str(input_file))
    except Exception as e:
        console.print(f"[red]Error saving file:[/red] {e}")
        return False

    # Show stats
    stats = translator.stats

    console.print("\n[green]✓ Translation complete![/green]\n")

    table = Table(title="Translation Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Strings translated", str(stats.translated))
    table.add_row("Skipped (existing)", str(stats.skipped_existing))
    table.add_row("Errors", str(stats.errors))
    table.add_row("Input tokens", f"{stats.input_tokens:,}")
    table.add_row("Output tokens", f"{stats.output_tokens:,}")

    total_cost = get_model_cost(
        translator.model,
        stats.input_tokens,
        stats.output_tokens,
        fetch_live=fetch_live_pricing,
    )
    if total_cost is not None:
        table.add_row("Estimated cost", f"${total_cost:.3f}")

    console.print(table)
    console.print(f"\n[green]Output saved to:[/green] {input_file}")
    return True


@app.command()
def translate(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Input .xcstrings file or a directory to search recursively"
        ),
    ],
    languages: Annotated[
        str,
        typer.Option(
            "-l", "--languages", help="Comma-separated language codes (e.g., fr,es,it)"
        ),
    ] = None,
    output_file: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            help="Output file (default: overwrites input; not allowed for directories)",
        ),
    ] = None,
    model: Annotated[
        str | None,
        typer.Option(
            "-m",
            "--model",
            help="Model: sonnet, gpt-5, gemini-2.5-flash, openrouter:vendor/model (or provider:model). Default: sonnet",
        ),
    ] = None,
    batch_size: Annotated[
        int, typer.Option("-b", "--batch-size", help="Strings per API call")
    ] = 25,
    concurrency: Annotated[
        int, typer.Option("-c", "--concurrency", help="Max parallel API requests")
    ] = 32,
    overwrite: Annotated[
        bool, typer.Option("--overwrite", help="Overwrite existing translations")
    ] = False,
    dry_run: Annotated[
        bool,
        typer.Option(
            "--dry-run", help="Show what would be translated without doing it"
        ),
    ] = False,
    app_context: Annotated[
        str | None,
        typer.Option(
            "--context",
            help="App description for better translations (or use context.md file)",
        ),
    ] = None,
    fill_missing: Annotated[
        bool,
        typer.Option(
            "--fill-missing",
            "-f",
            help="Auto-detect languages and only fill missing translations",
        ),
    ] = False,
    yes: Annotated[
        bool,
        typer.Option(
            "--yes",
            "-y",
            help="Skip confirmation when translating a directory of files",
        ),
    ] = False,
    no_fetch: Annotated[
        bool,
        typer.Option(
            "--no-fetch",
            help="Don't fetch live OpenRouter prices; use cached/static prices only",
        ),
    ] = False,
):
    """
    Translate an xcstrings file (or every .xcstrings file in a directory) using AI.

    Examples:
        xcstrings translate Localizable.xcstrings -l fr,es,it
        xcstrings translate ./MyApp -l fr,es --fill-missing   # recurse a project
        xcstrings translate input.xcstrings -l ja,ko --model gpt-5
        xcstrings translate input.xcstrings -m openrouter:anthropic/claude-sonnet-4.5 -l fr
        xcstrings translate input.xcstrings -l fr --dry-run
    """
    # Validate input path
    if not input_file.exists():
        console.print(f"[red]Error:[/red] Input not found: {input_file}")
        raise typer.Exit(1)

    is_dir = input_file.is_dir()

    # Validate flag combinations
    if fill_missing and languages:
        console.print(
            "[red]Error:[/red] --fill-missing and -l/--languages are mutually exclusive"
        )
        raise typer.Exit(1)
    if fill_missing and overwrite:
        console.print(
            "[red]Error:[/red] --fill-missing and --overwrite are mutually exclusive"
        )
        raise typer.Exit(1)
    if is_dir and output_file:
        console.print(
            "[red]Error:[/red] -o/--output is not allowed when the input is a directory"
        )
        raise typer.Exit(1)

    # Parse + validate target languages once (per-file in fill_missing mode)
    target_langs: list[str] | None = None
    if not fill_missing:
        if not languages:
            console.print(
                "[red]Error:[/red] No languages specified. Use -l/--languages (e.g., -l fr,es,it)"
            )
            raise typer.Exit(1)
        target_langs = _parse_target_languages(languages)
        invalid_langs = [
            lang for lang in target_langs if lang not in SUPPORTED_LANGUAGES
        ]
        if invalid_langs:
            console.print(
                f"[red]Error:[/red] Unsupported languages: {', '.join(invalid_langs)}"
            )
            console.print(
                "Run [cyan]xcstrings languages[/cyan] to see supported languages."
            )
            raise typer.Exit(1)

    # Validate model (accept aliases or provider:model format). Every known alias
    # and every provider:model string resolves to a value containing ":"; anything
    # without one is an unrecognised shorthand (likely a typo). Only validate when
    # the user explicitly passed a model.
    if model is not None:
        resolved = resolve_model(model)
        if resolved not in MODEL_PRICING and ":" not in resolved:
            console.print(f"[red]Error:[/red] Unknown model: {model}")
            console.print(f"Shortcuts: {', '.join(MODEL_ALIASES.keys())}")
            console.print("Or use provider:model format (e.g., openai:gpt-4o)")
            raise typer.Exit(1)

    # Resolve the provider/model and ensure the matching API key is available,
    # prompting interactively when one is missing. Dry runs need no key.
    model, resolved = _ensure_provider_and_key(model, require_key=not dry_run)

    common = {
        "languages": target_langs,
        "model": model,
        "resolved": resolved,
        "batch_size": batch_size,
        "concurrency": concurrency,
        "overwrite": overwrite,
        "dry_run": dry_run,
        "app_context": app_context,
        "fill_missing": fill_missing,
        "fetch_live_pricing": not no_fetch,
    }

    # Single file: preserve original behavior (exit 1 on failure).
    if not is_dir:
        if not _translate_one_file(input_file, **common):
            raise typer.Exit(1)
        return

    # Directory: discover, confirm, then translate each in place.
    files = _discover_xcstrings(input_file)
    if not files:
        console.print(f"[red]Error:[/red] No .xcstrings files found under {input_file}")
        raise typer.Exit(1)

    console.print(
        f"\n[cyan]Found {len(files)} .xcstrings file(s) under[/cyan] {input_file}:"
    )
    for f in files:
        console.print(f"  • {f.relative_to(input_file)}")

    if (
        not yes
        and not dry_run
        and not typer.confirm(f"\nTranslate all {len(files)} file(s)?")
    ):
        console.print("[yellow]Aborted.[/yellow]")
        raise typer.Exit(0)

    failures: list[Path] = []
    for f in files:
        console.rule(f"[bold]{f.relative_to(input_file)}")
        if not _translate_one_file(f, **common):
            failures.append(f)

    console.print()
    if failures:
        console.print(f"[red]Completed with {len(failures)} failure(s):[/red]")
        for f in failures:
            console.print(f"  • {f.relative_to(input_file)}")
        raise typer.Exit(1)
    console.print(f"[green]✓ Processed {len(files)} file(s).[/green]")


@app.command()
def info(
    input_file: Annotated[Path, typer.Argument(help="Input .xcstrings file")],
):
    """
    Show information about an xcstrings file.
    """
    if not input_file.exists():
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)

    try:
        xcstrings = XCStringsFile.from_file(str(input_file))
    except Exception as e:
        console.print(f"[red]Error loading file:[/red] {e}")
        raise typer.Exit(1) from e

    existing_langs = xcstrings.get_existing_languages()
    translatable = xcstrings.get_translatable_strings()

    console.print(Panel(f"[cyan]{input_file}[/cyan]", title="XCStrings File Info"))

    table = Table()
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Source language", xcstrings.sourceLanguage)
    table.add_row("Version", xcstrings.version)
    table.add_row("Total strings", str(len(xcstrings.strings)))
    table.add_row("Translatable strings", str(len(translatable)))
    table.add_row("Languages", ", ".join(sorted(existing_langs)) or "none")

    console.print(table)

    # Show language coverage
    if existing_langs:
        console.print("\n[cyan]Language Coverage:[/cyan]")

        coverage_table = Table()
        coverage_table.add_column("Language", style="cyan")
        coverage_table.add_column("Translated", style="green")
        coverage_table.add_column("Coverage", style="yellow")

        for lang in sorted(existing_langs):
            count = sum(
                1
                for key, entry in translatable
                if lang in entry.localizations and entry.localizations[lang].stringUnit
            )
            percentage = (count / len(translatable) * 100) if translatable else 0
            lang_name = SUPPORTED_LANGUAGES.get(lang, lang)
            coverage_table.add_row(
                f"{lang} ({lang_name})", str(count), f"{percentage:.1f}%"
            )

        console.print(coverage_table)


@app.command()
def languages():
    """
    List all supported languages.
    """
    table = Table(title="Supported Languages")
    table.add_column("Code", style="cyan")
    table.add_column("Language", style="green")

    for code, name in sorted(SUPPORTED_LANGUAGES.items()):
        table.add_row(code, name)

    console.print(table)
    console.print(f"\n[dim]Total: {len(SUPPORTED_LANGUAGES)} languages[/dim]")


@app.command()
def estimate(
    input_file: Annotated[Path, typer.Argument(help="Input .xcstrings file")],
    languages: Annotated[
        str, typer.Option("-l", "--languages", help="Comma-separated language codes")
    ] = None,
    model: Annotated[
        str, typer.Option("-m", "--model", help="Model (sonnet, gpt-4o, etc.)")
    ] = "sonnet",
    no_fetch: Annotated[
        bool,
        typer.Option(
            "--no-fetch",
            help="Don't fetch live OpenRouter prices; use cached/static prices only",
        ),
    ] = False,
):
    """
    Estimate the cost of translating an xcstrings file.
    """
    if not input_file.exists():
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)

    if not languages:
        console.print("[red]Error:[/red] No languages specified. Use -l/--languages")
        raise typer.Exit(1)

    target_langs = _parse_target_languages(languages)
    invalid_langs = [lang for lang in target_langs if lang not in SUPPORTED_LANGUAGES]
    if invalid_langs:
        console.print(
            f"[red]Error:[/red] Unsupported languages: {', '.join(invalid_langs)}"
        )
        console.print(
            "Run [cyan]xcstrings languages[/cyan] to see supported languages."
        )
        raise typer.Exit(1)

    try:
        xcstrings = XCStringsFile.from_file(str(input_file))
    except Exception as e:
        console.print(f"[red]Error loading file:[/red] {e}")
        raise typer.Exit(1) from e

    translator = XCStringsTranslator(model=model, fetch_live_pricing=not no_fetch)
    estimate = translator.estimate_cost(xcstrings, target_langs)

    table = Table(title=f"Cost Estimate ({model})")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Translatable strings", str(estimate["total_strings"]))
    table.add_row("Strings to translate", str(estimate["total_to_translate"]))
    table.add_row("Est. input tokens", f"{estimate['estimated_input_tokens']:,}")
    table.add_row("Est. output tokens", f"{estimate['estimated_output_tokens']:,}")

    if estimate["estimated_cost_usd"]:
        table.add_row("Estimated cost", f"${estimate['estimated_cost_usd']:.2f}")

    console.print(table)

    # Show all models comparison
    console.print("\n[cyan]Cost by model:[/cyan]")

    comparison = Table()
    comparison.add_column("Model", style="cyan")
    comparison.add_column("Est. Cost", style="green")
    comparison.add_column("Notes", style="yellow")

    model_notes = {
        "gpt-5-nano": "Cheapest",
        "gemini-2.0-flash": "Very cheap",
        "gpt-5-mini": "Cheap, fast",
        "gemini-2.5-flash": "Cheap, fast",
        "o3": "Reasoning, cheap",
        "gemini-3-flash": "Fast, quality",
        "haiku": "Fast, balanced",
        "gpt-5.1": "Quality",
        "gemini-2.5-pro": "Quality",
        "sonnet": "Balanced (default)",
        "gpt-5.2": "Latest OpenAI",
        "gemini-3-pro": "Latest Gemini",
        "opus": "Best quality",
        "or-gpt-5.4-nano": "OpenRouter, cheapest",
        "or-gpt-5.4-mini": "OpenRouter, cheap",
        "or-gemini-3.5-flash": "OpenRouter, fast",
        "or-sonnet": "OpenRouter, balanced",
    }

    for model_name in model_notes:
        est_cost = get_model_cost(
            model_name,
            estimate["estimated_input_tokens"],
            estimate["estimated_output_tokens"],
            fetch_live=not no_fetch,
        )
        if est_cost is not None:
            comparison.add_row(model_name, f"${est_cost:.2f}", model_notes[model_name])

    console.print(comparison)


@app.command()
def validate(
    input_file: Annotated[Path, typer.Argument(help="Input .xcstrings file")],
):
    """
    Validate an xcstrings file for common issues.
    """
    if not input_file.exists():
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1)

    try:
        xcstrings = XCStringsFile.from_file(str(input_file))
    except Exception as e:
        console.print(f"[red]Error loading file:[/red] {e}")
        raise typer.Exit(1) from e

    issues = []
    warnings = []

    # Check for missing translations
    existing_langs = xcstrings.get_existing_languages()
    translatable = xcstrings.get_translatable_strings()

    for lang in existing_langs:
        missing = []
        for key, entry in translatable:
            if (
                lang not in entry.localizations
                or not entry.localizations[lang].stringUnit
            ):
                missing.append(key)

        if missing:
            warnings.append(f"{lang}: {len(missing)} missing translations")

    # Check for format specifier mismatches
    for key, entry in translatable:
        source_specs = re.findall(r"%[\d$]*[@dlfse]|%lld|%%", key)

        for lang, loc in entry.localizations.items():
            if loc.stringUnit and loc.stringUnit.value:
                trans_specs = re.findall(
                    r"%[\d$]*[@dlfse]|%lld|%%", loc.stringUnit.value
                )
                if sorted(source_specs) != sorted(trans_specs):
                    issues.append(f"Format mismatch in {lang} for: {key[:50]}...")

    # Print results
    console.print(Panel(f"[cyan]{input_file}[/cyan]", title="Validation Results"))

    if issues:
        console.print(f"\n[red]Errors ({len(issues)}):[/red]")
        for issue in issues[:10]:  # Show first 10
            console.print(f"  • {issue}")
        if len(issues) > 10:
            console.print(f"  ... and {len(issues) - 10} more")

    if warnings:
        console.print(f"\n[yellow]Warnings ({len(warnings)}):[/yellow]")
        for warning in warnings:
            console.print(f"  • {warning}")

    if not issues and not warnings:
        console.print("[green]✓ No issues found![/green]")

    raise typer.Exit(1 if issues else 0)


if __name__ == "__main__":
    app()
