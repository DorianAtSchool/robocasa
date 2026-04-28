#!/usr/bin/env python3
"""Report the largest Python files and definitions in the repository.

This is meant as a pre-PR audit tool for identifying low-risk refactor
targets. It uses AST line spans so the numbers reflect actual definition size
rather than grep heuristics.
"""

from __future__ import annotations

import argparse
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class DefinitionSpan:
    kind: str
    qualname: str
    start_line: int
    end_line: int

    @property
    def line_count(self) -> int:
        return self.end_line - self.start_line + 1


def _iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if any(part.startswith(".") for part in path.parts):
            continue
        yield path


def _collect_definitions(tree: ast.AST) -> list[DefinitionSpan]:
    definitions: list[DefinitionSpan] = []

    def visit(node: ast.AST, parents: list[str]) -> None:
        for child in ast.iter_child_nodes(node):
            if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                qualname = ".".join([*parents, child.name])
                definitions.append(
                    DefinitionSpan(
                        kind="method" if parents else "function",
                        qualname=qualname,
                        start_line=child.lineno,
                        end_line=child.end_lineno or child.lineno,
                    )
                )
                visit(child, [*parents, child.name])
            elif isinstance(child, ast.ClassDef):
                qualname = ".".join([*parents, child.name])
                definitions.append(
                    DefinitionSpan(
                        kind="class",
                        qualname=qualname,
                        start_line=child.lineno,
                        end_line=child.end_lineno or child.lineno,
                    )
                )
                visit(child, [*parents, child.name])
            else:
                visit(child, parents)

    visit(tree, [])
    return definitions


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("."),
        help="Repository root to scan.",
    )
    parser.add_argument(
        "--top-files",
        type=int,
        default=15,
        help="Number of largest files to print.",
    )
    parser.add_argument(
        "--top-defs",
        type=int,
        default=20,
        help="Number of largest definitions to print.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Path fragment to exclude. May be repeated.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    root = args.root.resolve()
    excluded = tuple(fragment for fragment in args.exclude if fragment)

    file_rows: list[tuple[int, str]] = []
    definition_rows: list[tuple[int, str, str, str, int, int]] = []
    parse_failures: list[str] = []

    for path in _iter_python_files(root):
        relative_path = path.relative_to(root)
        relative_str = relative_path.as_posix()
        if excluded and any(fragment in relative_str for fragment in excluded):
            continue
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=str(relative_path))
        except (OSError, UnicodeDecodeError, SyntaxError) as exc:
            parse_failures.append(f"{relative_str}: {exc}")
            continue

        line_count = source.count("\n") + (0 if not source else 1)
        file_rows.append((line_count, relative_str))
        for definition in _collect_definitions(tree):
            definition_rows.append(
                (
                    definition.line_count,
                    definition.kind,
                    definition.qualname,
                    relative_str,
                    definition.start_line,
                    definition.end_line,
                )
            )

    file_rows.sort(key=lambda item: (-item[0], item[1]))
    definition_rows.sort(key=lambda item: (-item[0], item[3], item[4], item[1], item[2]))

    print("Largest Python files")
    for line_count, relative_str in file_rows[: args.top_files]:
        print(f"{line_count:6d}  {relative_str}")

    print("\nLargest classes/functions/methods")
    for line_count, kind, qualname, relative_str, start_line, end_line in definition_rows[
        : args.top_defs
    ]:
        print(
            f"{line_count:6d}  {kind:8s}  {qualname:48s}  {relative_str}:{start_line}-{end_line}"
        )

    if parse_failures:
        print("\nSkipped files")
        for failure in parse_failures:
            print(failure)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
