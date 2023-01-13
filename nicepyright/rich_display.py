from rich.style import Style
from rich.syntax import Syntax

from .utils import Range


def get_diagnostic_syntax(
    file_text: str,
    display_range_start: Range,
    display_range_end: Range,
    n_context_lines: int = 3,
    # error_style: Style = Style(color="red", bold=True, underline=True),
    error_style: Style = Style(bold=True, underline=True, reverse=True),
    dimmed_style: Style = Style(dim=True),
) -> Syntax:
    """Return a rich Syntax instance with context lines and highlighting around an error.

    Args:
        file_text: The full code contained within a file.
        display_range_start: The start of the range to highlight.
        display_range_end: The end of the range to highlight.
        n_context_lines: The number of lines of context to include before and after the error.
        error_style: The style to use for the error segment.
        dimmed_style: The style to use for the context lines.

    Returns:
        A rich Syntax instance for the code that caused the error.
    """
    code_lines = file_text.splitlines()
    n_lines = len(code_lines)

    snippet_line_start = display_range_start.line - n_context_lines + 1
    snippet_line_start = max(snippet_line_start, 1)
    while code_lines[snippet_line_start - 1].strip() == "":
        # Skip blank lines at the start of the snippet.
        snippet_line_start += 1

    snippet_line_end = display_range_end.line + n_context_lines + 1
    snippet_line_end = min(snippet_line_end, n_lines)
    while code_lines[snippet_line_end - 1].strip() == "":
        # Skip blank lines at the end of the snippet.
        snippet_line_end -= 1

    error_line_start = display_range_start.line + 1
    error_line_end = display_range_end.line + 2

    syntax = Syntax(
        file_text,
        "python",
        line_numbers=True,
        line_range=(
            snippet_line_start,
            snippet_line_end,
        ),
        highlight_lines=set(range(error_line_start, error_line_end)),
        background_color="default",
    )

    # Apply styling to the error segment
    syntax.stylize_range(
        error_style,
        (display_range_start.line + 1, display_range_start.character),
        (display_range_end.line + 1, display_range_end.character),
    )

    # Apply dimming to the context lines
    syntax.stylize_range(
        dimmed_style,
        start=(snippet_line_start, 0),
        end=(snippet_line_end + 1, 0),
    )

    # Remove the dimming from the error lines
    syntax.stylize_range(
        Style(dim=False),
        start=(error_line_start, 0),
        end=(error_line_end, 0),
    )
    return syntax


__all__ = ("get_diagnostic_syntax",)
