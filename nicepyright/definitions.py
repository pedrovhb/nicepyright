from __future__ import annotations

import asyncio
import json
import textwrap
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from functools import lru_cache
from importlib import resources
from itertools import chain, cycle
from pathlib import Path
from typing import AsyncIterable, Dict, List, NamedTuple, Tuple, Union

import parse
from rich.console import Console, ConsoleOptions, ConsoleRenderable, Group, RenderResult
from rich.padding import Padding
from rich.panel import Panel
from rich.text import Span, Text
from rich.tree import Tree

from . import con
from .rich_display import get_diagnostic_syntax
from .utils import camel_case_to_capitalized_text, Range


class DiagnosticRule(str, Enum):
    """The diagnostic rule that caused the error.

    See Also:
        https://github.com/microsoft/pyright/
        File: packages/pyright-internal/src/common/diagnosticRules.ts
    """

    StrictListInference = "strictListInference"
    StrictSetInference = "strictSetInference"
    StrictDictionaryInference = "strictDictionaryInference"
    AnalyzeUnannotatedFunctions = "analyzeUnannotatedFunctions"
    StrictParameterNoneValue = "strictParameterNoneValue"
    EnableTypeIgnoreComments = "enableTypeIgnoreComments"

    GeneralTypeIssues = "reportGeneralTypeIssues"
    PropertyTypeMismatch = "reportPropertyTypeMismatch"
    FunctionMemberAccess = "reportFunctionMemberAccess"
    MissingImports = "reportMissingImports"
    MissingModuleSource = "reportMissingModuleSource"
    MissingTypeStubs = "reportMissingTypeStubs"
    ImportCycles = "reportImportCycles"
    UnusedImport = "reportUnusedImport"
    UnusedClass = "reportUnusedClass"
    UnusedFunction = "reportUnusedFunction"
    UnusedVariable = "reportUnusedVariable"
    DuplicateImport = "reportDuplicateImport"
    WildcardImportFromLibrary = "reportWildcardImportFromLibrary"
    OptionalSubscript = "reportOptionalSubscript"
    OptionalMemberAccess = "reportOptionalMemberAccess"
    OptionalCall = "reportOptionalCall"
    OptionalIterable = "reportOptionalIterable"
    OptionalContextManager = "reportOptionalContextManager"
    OptionalOperand = "reportOptionalOperand"
    TypedDictNotRequiredAccess = "reportTypedDictNotRequiredAccess"
    UntypedFunctionDecorator = "reportUntypedFunctionDecorator"
    UntypedClassDecorator = "reportUntypedClassDecorator"
    UntypedBaseClass = "reportUntypedBaseClass"
    UntypedNamedTuple = "reportUntypedNamedTuple"
    PrivateUsage = "reportPrivateUsage"
    TypeCommentUsage = "reportTypeCommentUsage"
    PrivateImportUsage = "reportPrivateImportUsage"
    ConstantRedefinition = "reportConstantRedefinition"
    IncompatibleMethodOverride = "reportIncompatibleMethodOverride"
    IncompatibleVariableOverride = "reportIncompatibleVariableOverride"
    InconsistentConstructor = "reportInconsistentConstructor"
    OverlappingOverload = "reportOverlappingOverload"
    MissingSuperCall = "reportMissingSuperCall"
    UninitializedInstanceVariable = "reportUninitializedInstanceVariable"
    InvalidStringEscapeSequence = "reportInvalidStringEscapeSequence"
    UnknownParameterType = "reportUnknownParameterType"
    UnknownArgumentType = "reportUnknownArgumentType"
    UnknownLambdaType = "reportUnknownLambdaType"
    UnknownVariableType = "reportUnknownVariableType"
    UnknownMemberType = "reportUnknownMemberType"
    MissingParameterType = "reportMissingParameterType"
    MissingTypeArgument = "reportMissingTypeArgument"
    InvalidTypeVarUse = "reportInvalidTypeVarUse"
    CallInDefaultInitializer = "reportCallInDefaultInitializer"
    UnnecessaryIsInstance = "reportUnnecessaryIsInstance"
    UnnecessaryCast = "reportUnnecessaryCast"
    UnnecessaryComparison = "reportUnnecessaryComparison"
    UnnecessaryContains = "reportUnnecessaryContains"
    AssertAlwaysTrue = "reportAssertAlwaysTrue"
    SelfClsParameterName = "reportSelfClsParameterName"
    ImplicitStringConcatenation = "reportImplicitStringConcatenation"
    UndefinedVariable = "reportUndefinedVariable"
    UnboundVariable = "reportUnboundVariable"
    InvalidStubStatement = "reportInvalidStubStatement"
    IncompleteStub = "reportIncompleteStub"
    UnsupportedDunderAll = "reportUnsupportedDunderAll"
    UnusedCallResult = "reportUnusedCallResult"
    UnusedCoroutine = "reportUnusedCoroutine"
    UnusedExpression = "reportUnusedExpression"
    UnnecessaryTypeIgnoreComment = "reportUnnecessaryTypeIgnoreComment"
    MatchNotExhaustive = "reportMatchNotExhaustive"
    ShadowedImports = "reportShadowedImports"


class SeverityLevel(str, Enum):
    Error = "error"
    Warning = "warning"
    Information = "information"

    def __rich_console__(self, console, options):
        if self == SeverityLevel.Error:
            yield f"[bold red]{self.value.capitalize()}[/bold red]"
        elif self == SeverityLevel.Warning:
            yield f"[bold yellow]{self.value.capitalize()}[/bold yellow]"
        elif self == SeverityLevel.Information:
            yield f"[bold blue]{self.value.capitalize()}[/bold blue]"
        else:
            raise ValueError(f"Unknown severity level: {self}")


class ParsedDiagnosticAddendum(NamedTuple):
    """A parsed diagnostic addendum.

    This is a message that complements information from a diagnostic message.

    An example would be a diagnostic message such as the following:

        "overrideParamName": "Parameter {index} name mismatch: base parameter is
            named \"{baseName}\", override parameter is named \"{overrideName}\""

    In this case, the class would be have the following attributes:

    category: "overrideParamName"

    message: 'Parameter foo name mismatch: base parameter is named "bar", override
        parameter is named "baz"'

    values: {"index": "foo", "baseName": "bar", "overrideName": "baz"}

    category_text: "Override param name"
    """

    message: str
    category: str
    category_text: str
    values: Dict[str, str]
    result: parse.Result

    @classmethod
    @lru_cache(maxsize=512)
    def from_message(cls, message: str) -> ParsedDiagnosticAddendum | None:
        """Parse a diagnostic addendum message."""

        for category, parser in parsed_messages["DiagnosticAddendum"].items():
            if parsed := parser.parse(message):
                break
        else:
            con.log(f"Failed to parse diagnostic addendum: {message}")
            return None
            # raise ValueError(f"Could not parse diagnostic addendum: {message!r}")

        category_text = camel_case_to_capitalized_text(category)
        return cls(
            message=message,
            category=category,
            category_text=category_text,
            values={},
            result=parsed,
        )


class ParsedDiagnostic(NamedTuple):
    """A parsed diagnostic message.

    This is a message that is returned by pyright to indicate an error or warning
    in a particular piece of code. It is parsed from the JSON output of pyright,
    and contains the following attributes:

    category: The diagnostic category, e.g. "paramSpecArgsMissing".

    message: The diagnostic message, e.g. "Arguments for ParamSpec "MyClass" are missing".

    addendums: A list of diagnostic addendums, which are additional messages that complement
        the diagnostic message.

    values: A dictionary of values that are used to format the diagnostic message. For example,
        the diagnostic message "Arguments for ParamSpec {type} are missing" would look like
        {"type": "MyClass"}.
    """

    category: str
    message: str
    result: parse.Result
    addendums: List[ParsedDiagnosticAddendum]

    @property
    def category_text(self) -> str:
        return camel_case_to_capitalized_text(self.category)

    @property
    def values(self) -> Dict[str, str]:
        return self.result.named

    def __rich_repr__(self) -> str:
        cat_str = camel_case_to_capitalized_text(self.category)
        results = [
            f"{camel_case_to_capitalized_text(k)}: {v}" for k, v in self.result.named.items()
        ]
        results_str = textwrap.indent("\n".join(results), "  ")
        return f"{cat_str}:\n{results_str}"


@dataclass(frozen=True)
class PyrightDiagnostic:
    """Base class for pyright diagnostics information."""

    file: Path
    severity: SeverityLevel
    message: str
    range_start: Range
    range_end: Range
    rule: Union[DiagnosticRule, None] = None

    @classmethod
    def from_dict(cls, d: dict):
        """Convert a dict, as per `pyright --outputjson`, to a PyrightOutput instance.

        Args:
            d: A dict, as per `pyright --outputjson`.

        Returns:
            A PyrightOutput instance.
        """
        d["file"] = Path(d["file"])
        d["severity"] = SeverityLevel[d["severity"].capitalize()]
        rng = d.pop("range")
        d["range_start"] = Range(rng["start"]["line"], rng["start"]["character"])
        d["range_end"] = Range(rng["end"]["line"], rng["end"]["character"])
        d["rule"] = DiagnosticRule[d["rule"].replace("report", "")] if "rule" in d else None
        return cls(**d)

    @property
    def code_fragment(self):
        """Return the code that caused the error.

        This is the code that is between the start and end of the error range.

        Returns:
            The code that caused the error.
        """
        file_contents = self.file.read_text()
        lines = file_contents.splitlines()
        out_lines = lines[self.range_start.line : self.range_end.line + 1]
        out_lines[0] = out_lines[0][self.range_start.character :]
        out_lines[-1] = out_lines[-1][: self.range_end.character]
        return "\n".join(out_lines)

    def get_surrounding_code(self, n_lines: int = 3) -> Tuple[str, int, int]:
        """Return the code that caused the error, with surrounding context.

        Args:
            n_lines: The number of lines of context to include before and after the error.

        Returns:
            The code that caused the error, with surrounding context.
        """
        file_contents = self.file.read_text()
        lines = file_contents.splitlines()
        start_line = max(0, self.range_start.line - n_lines)
        end_line = min(len(lines), self.range_end.line + n_lines + 1)
        out_lines = lines[start_line:end_line]
        return "\n".join(out_lines), start_line + 1, end_line

    def get_message(self) -> RenderResult:
        # yield f"Getting message for {self}"
        parsed = parse_message(self.message)
        diagnostic_text = Text(parsed.message)

        # Gather all unique special values in the message, and assign colors to each.
        unique_tags = set(
            chain(
                parsed.result.named.values(),
                *[ad.result.named.values() for ad in parsed.addendums],
            )
        )
        palette = (
            "#f08080",
            "#90ee90",
            "#87cefa",
            "#ffa07a",
            "#ba55d3",
            "#40e0d0",
            "#ff69b4",
            "#7b68ee",
            "#00bfff",
            "#ffd700",
            "#daa520",
            "#8b0000",
        )
        tag_colors = dict(zip(unique_tags, cycle(palette)))

        # Go through diagnostic  and addendums and colorize the special values.
        for result_tag, result_span in parsed.result.spans.items():
            result_text = parsed.result.named[result_tag]
            diagnostic_text.spans.append(
                Span(*result_span, style=f"bold {tag_colors[result_text]}")
            )

        addendum_text_list = []
        for addendum in parsed.addendums:
            addendum_text = Text(addendum.message)
            for result_tag, result_span in addendum.result.spans.items():
                result_text = addendum.result.named[result_tag]
                addendum_text.spans.append(
                    Span(*result_span, style=f"bold {tag_colors[result_text]}")
                )
            addendum_text_list.append(addendum_text)

        # Create a tree of the diagnostic and addendums.
        # todo - allow for disabling colors, and for disabling addendums.
        tree = Tree(diagnostic_text)
        for addendum_text in addendum_text_list:
            tree.add(addendum_text)

        yield tree

    def __rich_console__(self, console, options) -> ConsoleRenderable:
        """Display the diagnostic as rich terminal output.

        Args:
            console: The rich console instance.
            options: The rich console options.
        """

        err_class_style = {
            SeverityLevel.Error: "[bold red]",
            SeverityLevel.Warning: "[bold yellow]",
            SeverityLevel.Information: "[bold blue]",
        }[self.severity]

        if self.rule is not None:
            err_description = f" ({camel_case_to_capitalized_text(self.rule.name)})"
        else:
            err_description = ""
        err_title = f"{err_class_style}{self.severity.name.capitalize()}[/]{err_description}"

        f_path, f_name = str(self.file.relative_to(Path.cwd()).parent), self.file.name
        file_display = f"[dim]{f_path}[/]/[bold #FFFFFF]{f_name}[/]"
        err_subtitle = (
            f"{file_display} ─── "
            f"[blue]{self.range_start.line + 1}[/]:"
            f"[blue]{self.range_start.character + 1}[/]"
        )

        panel = Panel(
            Group(
                get_diagnostic_syntax(self.file.read_text(), self.range_start, self.range_end),
                Padding(Group(*self.get_message()), (1, 2, 0, 2)),
            ),
            title=err_title,
            title_align="left",
            subtitle=err_subtitle,
            subtitle_align="left",
            padding=(1, 1),
        )
        yield Padding(panel, (1, 0))


@dataclass(frozen=True)
class PyrightOutput:
    version: str
    diagnostics: List[PyrightDiagnostic]
    time: datetime
    files_analyzed: int
    error_count: int
    warning_count: int
    information_count: int
    time_in_sec: float

    @classmethod
    def from_dict(cls, d: dict):
        """Convert a dict, as per `pyright --outputjson`, to a PyrightOutput instance.

        Args:
            d: A dict, as per `pyright --outputjson`.

        Returns:
            A PyrightOutput instance.
        """
        summary = d.pop("summary")
        d["diagnostics"] = [
            PyrightDiagnostic.from_dict(diag) for diag in d.pop("generalDiagnostics")
        ]
        d["time"] = datetime.fromtimestamp(float(d["time"]) / 1000)
        d["files_analyzed"] = int(summary["filesAnalyzed"])
        d["error_count"] = int(summary["errorCount"])
        d["warning_count"] = int(summary["warningCount"])
        d["information_count"] = int(summary["informationCount"])
        d["time_in_sec"] = float(summary["timeInSec"])
        return cls(**d)

    def grouped_by_file(self) -> Dict[Path, Tuple[PyrightDiagnostic]]:
        """Return a dict with files as keys and a tuple of diagnostics pertaining to each file as
        values.
        """
        out = defaultdict(list)
        for diag in self.diagnostics:
            out[diag.file].append(diag)

        out_tup: Dict[Path, Tuple[PyrightDiagnostic]] = {}
        for file, diags in out.items():
            out_tup[file] = tuple(diags)
        return out_tup

    @classmethod
    async def pyright_watch(cls) -> AsyncIterable[PyrightOutput]:
        """Watch pyright for changes.

        Yields:
            A dict, as per `pyright --outputjson`.
        """
        proc = await asyncio.create_subprocess_exec(
            "pyright",
            "--watch",
            "--outputjson",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        assert proc.stdout is not None
        lns = []
        async for line in proc.stdout:
            assert isinstance(line, bytes)
            lns.append(line)
            if line == b"}\n":
                yield cls.from_dict(json.loads(b"".join(lns)))
                lns = []

    def get_per_file_info(self) -> RenderResult:
        for file, diags in self.grouped_by_file().items():
            yield Text.from_markup(
                f"[green]--------- [bold]{file}[/bold] ---------[/]",
                justify="center",
            )
            for diag in diags:
                yield diag

    def get_output_stats_panel(self) -> RenderResult:
        """Return a renderable for the stats panel."""

        s_error = "error" if self.error_count == 1 else "errors"
        s_warning = "warning" if self.warning_count == 1 else "warnings"
        s_info = "message" if self.information_count == 1 else "messages"

        count_error = self.error_count if self.error_count > 0 else "No"
        count_warning = self.warning_count if self.warning_count > 0 else "No"
        count_info = self.information_count if self.information_count > 0 else "No"

        yield Panel(
            Text("\t", justify="center", end="").join(
                [
                    Text.from_markup(
                        f"[bold]{self.files_analyzed}[/] files analyzed in "
                        f"[bold]{self.time_in_sec:.3f}[/bold] seconds",
                        end="",
                    ),
                    Text.from_markup(f"[bold red]{count_error} {s_error}", end=""),
                    Text.from_markup(f"[bold yellow]{count_warning} {s_warning}", end=""),
                    Text.from_markup(f"[bold blue]{count_info} information {s_info}", end=""),
                ]
            ),
        )

    def __rich_console__(self, console: Console, options: ConsoleOptions) -> RenderResult:
        """Render the pyright output as a rich console."""
        yield f"pyright {self.version}"

        yield Padding(Group(*self.get_per_file_info()), (1, 2))
        yield from self.get_output_stats_panel()


def _get_message_parsers() -> Dict[str, Dict[str, parse.Parser]]:
    """Gets the parsers for the diagnostic messages.

    Returns:
        A dictionary mapping the categories Diagnostic and DiagnosticAddendum to their parsers.

    Examples:
        >>> parsers = _get_message_parsers()
        >>> one_parser = parsers["Diagnostic"]["constructorNoArgs"]
        >>> one_parser
        <Parser 'Expected no argum...'>
        >>> one_parser.parse('Expected no arguments to \"{MyClass}\" constructor')
        <Result () {'type': '{MyClass}'}>

    """
    base_dir = resources.files("nicepyright") / "data"
    messages_file = Path(str(base_dir.joinpath("messages.json")))
    raw_messages = json.loads(messages_file.read_text())
    msg_parsers: Dict[str, Dict[str, parse.Parser]] = {}
    for category, messages in raw_messages.items():
        cat = msg_parsers[category] = {}
        for name, message in messages.items():
            cat[name] = parse.compile(message, case_sensitive=True)

    return msg_parsers


def parse_message(message: str) -> ParsedDiagnostic:
    diagnostic, *addendums = map(str.strip, message.splitlines())
    parsed_addendums = [
        parsed_addendum
        for addendum in addendums
        if (parsed_addendum := ParsedDiagnosticAddendum.from_message(addendum))
    ]

    parsed_diagnostics = [
        ParsedDiagnostic(cat, diagnostic, parsed, parsed_addendums)
        for cat, p in parsed_messages["Diagnostic"].items()
        if (parsed := p.parse(diagnostic))
    ]
    if len(parsed_diagnostics) == 0:
        # con.log(f"Could not parse diagnostic: {diagnostic}", log_locals=True)
        return ParsedDiagnostic(
            "unknown", diagnostic, parse.parse("result: {mask}", "result: idk"), []
        )  # todo
    elif len(parsed_diagnostics) > 1:
        # con.log(f"Could not parse diagnostic: {diagnostic}", log_locals=True)
        pass

    return parsed_diagnostics[-1]


parsed_messages = _get_message_parsers()

__all__ = (
    "DiagnosticRule",
    "parse_message",
    "parsed_messages",
    "ParsedDiagnostic",
    "ParsedDiagnosticAddendum",
    "PyrightDiagnostic",
    "PyrightOutput",
    "SeverityLevel",
)
