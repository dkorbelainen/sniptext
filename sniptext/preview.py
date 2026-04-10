"""Interactive text preview and confirmation UI."""

from typing import Tuple

from loguru import logger


class TextPreview:
    """Interactive text preview using rich/textual."""

    def __init__(self):
        """Initialize text preview UI."""
        self._use_rich = self._check_rich_available()

    @staticmethod
    def _check_rich_available() -> bool:
        """Check if rich library is available."""
        try:
            import rich  # noqa: F401

            return True
        except ImportError:
            return False

    def show_preview(self, text: str, allow_edit: bool = False) -> Tuple[str, bool]:
        """Show interactive preview of recognized text.

        Args:
            text: Recognized text to preview
            allow_edit: Allow user to edit text (requires interactive terminal)

        Returns:
            tuple of (modified_text, should_copy)
            - modified_text: User's final text (original or edited)
            - should_copy: True if user confirmed, False if cancelled
        """
        if not self._use_rich:
            return self._show_fallback_preview(text)

        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.prompt import Prompt
            from rich.text import Text

            console = Console()

            # Show text in a nice panel
            styled_text = Text(text, style="bold cyan")
            panel = Panel(
                styled_text,
                title="[bold green]✓ OCR Result[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
            console.print(panel)

            # Prompt for action
            console.print("[yellow]Options:[/yellow]")
            console.print("  [green]C[/green]opy text (Enter)")
            if allow_edit:
                console.print("  [yellow]E[/yellow]dit text")
            console.print("  [red]Cancel[/red] (Esc or Q)")
            console.print()

            # Get user input
            while True:
                try:
                    choice = (
                        Prompt.ask(
                            "[bold]Action[/bold]",
                            choices=["c", "q", "e"] if allow_edit else ["c", "q"],
                            default="c",
                        )
                        .strip()
                        .lower()
                    )

                    if choice in ("", "c"):
                        return (text, True)
                    elif choice == "q":
                        console.print("[red]Cancelled[/red]")
                        return (text, False)
                    elif choice == "e" and allow_edit:
                        # Edit using Prompt.ask with default
                        edited = Prompt.ask("[bold]Edit text[/bold]", default=text)
                        if edited:
                            console.print(f"[green]✓ Updated[/green] ({len(edited)} chars)")
                            return (edited, True)
                    else:
                        console.print("[red]Invalid option[/red]")
                except KeyboardInterrupt:
                    console.print("\n[red]Cancelled[/red]")
                    return (text, False)

        except Exception as e:
            logger.warning(f"Rich preview failed: {e}, using fallback")
            return self._show_fallback_preview(text)

    @staticmethod
    def _show_fallback_preview(text: str) -> Tuple[str, bool]:
        """Fallback preview without rich library."""
        print("\n" + "=" * 60)
        print("OCR RESULT:")
        print("=" * 60)
        print(text)
        print("=" * 60)

        while True:
            try:
                choice = input("\nCopy text? (y/n): ").strip().lower()

                if choice in ("y", "yes", ""):
                    return (text, True)
                elif choice in ("n", "no", "q"):
                    print("Cancelled")
                    return (text, False)
                else:
                    print("Invalid choice. Enter 'y' or 'n'")
            except (KeyboardInterrupt, EOFError):
                print("\nCancelled")
                return (text, False)

    def is_available(self) -> bool:
        """Check if rich library is available for full interactive features."""
        return self._use_rich
