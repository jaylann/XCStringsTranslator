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
import json
from pathlib import Path
from typing import Annotated, Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.panel import Panel
from rich import print as rprint

from .models import XCStringsFile, SUPPORTED_LANGUAGES
from .translator import XCStringsTranslator, MODEL_CONFIGS, MODEL_ALIASES

app = typer.Typer(
    name="xcstrings",
    help="Translate Apple Localizable.xcstrings files using Claude AI",
    add_completion=False,
)
console = Console()


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
    raw_langs = [l.strip() for l in (languages or "").split(",") if l.strip()]
    return [_normalize_language_tag(l) for l in raw_langs]


@app.command()
def translate(
    input_file: Annotated[Path, typer.Argument(help="Input .xcstrings file")],
    languages: Annotated[str, typer.Option("-l", "--languages", help="Comma-separated language codes (e.g., fr,es,it)")] = None,
    output_file: Annotated[Optional[Path], typer.Option("-o", "--output", help="Output file (default: overwrites input)")] = None,
    model: Annotated[str, typer.Option("-m", "--model", help="Claude model: opus, sonnet, haiku")] = "sonnet",
    batch_size: Annotated[int, typer.Option("-b", "--batch-size", help="Strings per API call")] = 25,
    concurrency: Annotated[int, typer.Option("-c", "--concurrency", help="Max parallel API requests")] = 32,
    overwrite: Annotated[bool, typer.Option("--overwrite", help="Overwrite existing translations")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run", help="Show what would be translated without doing it")] = False,
    app_context: Annotated[Optional[str], typer.Option("--context", help="App description for better translations")] = None,
    fill_missing: Annotated[bool, typer.Option("--fill-missing", "-f", help="Auto-detect languages and only fill missing translations")] = False,
):
    """
    Translate an xcstrings file to new languages using Claude AI.
    
    Examples:
        xcstrings translate Localizable.xcstrings -l fr,es,it
        xcstrings translate input.xcstrings -l ja,ko -o output.xcstrings --model opus
        xcstrings translate input.xcstrings -l fr --dry-run
    """
    # Validate input file
    if not input_file.exists():
        console.print(f"[red]Error:[/red] Input file not found: {input_file}")
        raise typer.Exit(1)

    # Validate flag combinations
    if fill_missing and languages:
        console.print("[red]Error:[/red] --fill-missing and -l/--languages are mutually exclusive")
        raise typer.Exit(1)

    if fill_missing and overwrite:
        console.print("[red]Error:[/red] --fill-missing and --overwrite are mutually exclusive")
        raise typer.Exit(1)

    # Load the xcstrings file early if using fill_missing
    if fill_missing:
        console.print(f"\n[cyan]Loading:[/cyan] {input_file}")
        try:
            xcstrings = XCStringsFile.from_file(str(input_file))
        except Exception as e:
            console.print(f"[red]Error loading file:[/red] {e}")
            raise typer.Exit(1)

        # Auto-detect target languages
        target_langs = sorted(xcstrings.get_languages_with_localizations())

        if not target_langs:
            console.print("[yellow]Warning:[/yellow] No languages with existing translations found (only source language present)")
            console.print("Use -l/--languages to specify target languages instead")
            raise typer.Exit(0)
    else:
        if not languages:
            console.print("[red]Error:[/red] No languages specified. Use -l/--languages (e.g., -l fr,es,it)")
            raise typer.Exit(1)

        # Parse languages
        target_langs = _parse_target_languages(languages)
    
    # Validate languages
    invalid_langs = [l for l in target_langs if l not in SUPPORTED_LANGUAGES]
    if invalid_langs:
        console.print(f"[red]Error:[/red] Unsupported languages: {', '.join(invalid_langs)}")
        console.print("Run [cyan]xcstrings languages[/cyan] to see supported languages.")
        raise typer.Exit(1)
    
    # Validate model
    if model not in MODEL_CONFIGS:
        console.print(f"[red]Error:[/red] Invalid model: {model}. Choose from: {', '.join(MODEL_CONFIGS.keys())}")
        raise typer.Exit(1)

    # Load the xcstrings file (if not already loaded for fill_missing)
    if not fill_missing:
        console.print(f"\n[cyan]Loading:[/cyan] {input_file}")
        try:
            xcstrings = XCStringsFile.from_file(str(input_file))
        except Exception as e:
            console.print(f"[red]Error loading file:[/red] {e}")
            raise typer.Exit(1)
    
    # Show file info
    existing_langs = xcstrings.get_existing_languages()
    translatable = xcstrings.get_translatable_strings()

    console.print(f"[green]✓[/green] Loaded {len(xcstrings.strings)} total strings")
    console.print(f"[green]✓[/green] {len(translatable)} translatable strings")
    console.print(f"[green]✓[/green] Existing languages: {', '.join(sorted(existing_langs)) or 'none'}")
    console.print(f"[green]✓[/green] Target languages: {', '.join(target_langs)}")
    if fill_missing:
        console.print(f"[green]✓[/green] Mode: Fill missing translations only")
    console.print(f"[green]✓[/green] Model: {MODEL_CONFIGS[model]}")
    
    # Default app context for NeatPass
    if not app_context:
        app_context = """NeatPass - A privacy-focused iOS app that turns PDFs (tickets/documents) into Apple Wallet passes. Key features:
- 100% on-device processing (no cloud uploads)
- One-time purchase (no subscription)
- Automatic barcode detection using ML
- Privacy-first approach

Tone: Friendly, casual, reassuring about privacy. Use informal "you" forms where applicable (du/tu/etc)."""
    
    # Initialize translator
    translator = XCStringsTranslator(
        model=model,
        batch_size=batch_size,
        concurrency=concurrency,
        app_context=app_context,
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
        table.add_row("Estimated input tokens", f"{estimate['estimated_input_tokens']:,}")
        table.add_row("Estimated output tokens", f"{estimate['estimated_output_tokens']:,}")
        if estimate["estimated_cost_usd"]:
            table.add_row("Estimated cost", f"${estimate['estimated_cost_usd']:.2f}")
        
        console.print(table)
        
        if estimate["strings_per_language"]:
            console.print("\n[cyan]Strings per language:[/cyan]")
            for lang, count in estimate["strings_per_language"].items():
                lang_name = SUPPORTED_LANGUAGES.get(lang, lang)
                console.print(f"  {lang}: {count} strings ({lang_name})")
        
        raise typer.Exit(0)
    
    # Translate with progress
    output_path = output_file or input_file
    
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
                1 for key, entry in translatable 
                if lang not in entry.localizations or not entry.localizations[lang].stringUnit
            )
            task_ids[lang] = progress.add_task(f"[cyan]{lang}[/cyan]", total=strings_for_lang or 1)
        
        try:
            xcstrings = translator.translate_file(
                xcstrings,
                target_langs,
                overwrite=overwrite,
                progress_callback=progress_callback,
            )
        except Exception as e:
            console.print(f"\n[red]Translation error:[/red] {e}")
            raise typer.Exit(1)
    
    # Save the result
    console.print(f"\n[cyan]Saving:[/cyan] {output_path}")
    try:
        xcstrings.to_file(str(output_path))
    except Exception as e:
        console.print(f"[red]Error saving file:[/red] {e}")
        raise typer.Exit(1)
    
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
    
    # Calculate cost
    costs = {
        # Claude 4.5 pricing (per 1M tokens), as of 2025-12-13 in Anthropic docs.
        MODEL_CONFIGS["opus"]: {"input": 5.0, "output": 25.0},
        MODEL_CONFIGS["sonnet"]: {"input": 3.0, "output": 15.0},
        MODEL_CONFIGS["haiku"]: {"input": 1.0, "output": 5.0},
        # Also accept aliases in case a run migrated to an alias fallback.
        MODEL_ALIASES["opus"]: {"input": 5.0, "output": 25.0},
        MODEL_ALIASES["sonnet"]: {"input": 3.0, "output": 15.0},
        MODEL_ALIASES["haiku"]: {"input": 1.0, "output": 5.0},
    }
    if translator.model in costs:
        cost = costs[translator.model]
        total_cost = (
            (stats.input_tokens / 1_000_000) * cost["input"] +
            (stats.output_tokens / 1_000_000) * cost["output"]
        )
        table.add_row("Estimated cost", f"${total_cost:.3f}")
    
    console.print(table)
    console.print(f"\n[green]Output saved to:[/green] {output_path}")


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
        raise typer.Exit(1)
    
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
                1 for key, entry in translatable
                if lang in entry.localizations and entry.localizations[lang].stringUnit
            )
            percentage = (count / len(translatable) * 100) if translatable else 0
            lang_name = SUPPORTED_LANGUAGES.get(lang, lang)
            coverage_table.add_row(
                f"{lang} ({lang_name})",
                str(count),
                f"{percentage:.1f}%"
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
    languages: Annotated[str, typer.Option("-l", "--languages", help="Comma-separated language codes")] = None,
    model: Annotated[str, typer.Option("-m", "--model", help="Claude model")] = "sonnet",
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
    invalid_langs = [l for l in target_langs if l not in SUPPORTED_LANGUAGES]
    if invalid_langs:
        console.print(f"[red]Error:[/red] Unsupported languages: {', '.join(invalid_langs)}")
        console.print("Run [cyan]xcstrings languages[/cyan] to see supported languages.")
        raise typer.Exit(1)
    
    try:
        xcstrings = XCStringsFile.from_file(str(input_file))
    except Exception as e:
        console.print(f"[red]Error loading file:[/red] {e}")
        raise typer.Exit(1)
    
    translator = XCStringsTranslator(model=model)
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
    comparison.add_column("Speed", style="yellow")
    
    costs = {
        "haiku": {"input": 1.0, "output": 5.0, "speed": "Fastest"},
        "sonnet": {"input": 3.0, "output": 15.0, "speed": "Balanced"},
        "opus": {"input": 5.0, "output": 25.0, "speed": "Best quality"},
    }
    
    for model_name, cost_info in costs.items():
        est_cost = (
            (estimate["estimated_input_tokens"] / 1_000_000) * cost_info["input"] +
            (estimate["estimated_output_tokens"] / 1_000_000) * cost_info["output"]
        )
        comparison.add_row(model_name, f"${est_cost:.2f}", cost_info["speed"])
    
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
        raise typer.Exit(1)
    
    issues = []
    warnings = []
    
    # Check for missing translations
    existing_langs = xcstrings.get_existing_languages()
    translatable = xcstrings.get_translatable_strings()
    
    for lang in existing_langs:
        missing = []
        for key, entry in translatable:
            if lang not in entry.localizations:
                missing.append(key)
            elif not entry.localizations[lang].stringUnit:
                missing.append(key)
        
        if missing:
            warnings.append(f"{lang}: {len(missing)} missing translations")
    
    # Check for format specifier mismatches
    import re
    for key, entry in translatable:
        source_specs = re.findall(r'%[\d$]*[@dlfse]|%lld|%%', key)
        
        for lang, loc in entry.localizations.items():
            if loc.stringUnit and loc.stringUnit.value:
                trans_specs = re.findall(r'%[\d$]*[@dlfse]|%lld|%%', loc.stringUnit.value)
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
