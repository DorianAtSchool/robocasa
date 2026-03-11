from __future__ import annotations

import ast
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
ATOMIC_TASK_DIR = (
    REPO_ROOT / "robocasa" / "environments" / "kitchen" / "atomic"
)

# These are shared helpers or abstract bases, not promptable leaf tasks.
HELPER_CLASS_NAMES = {
    "CloseDoor",
    "CloseDropDownDoor",
    "ManipulateDoor",
    "ManipulateDrawer",
    "ManipulateLowerDoor",
    "ManipulateSinkFaucet",
    "ManipulateStoveKnob",
    "MicrowavePressButton",
    "OpenDoor",
    "OpenDropDownDoor",
    "PickPlace",
    "PickPlaceCoffee",
}


@dataclass(frozen=True)
class ConstructorArg:
    name: str
    default: str | None
    required: bool
    kind: str = "positional_or_keyword"


@dataclass(frozen=True)
class AtomicToolSpec:
    name: str
    description: str
    constructor_args: tuple[ConstructorArg, ...]
    source_path: str
    signature_text: str

    def to_prompt_block(self) -> str:
        if self.constructor_args:
            arg_text = ", ".join(
                format_constructor_arg(arg) for arg in self.constructor_args
            )
        else:
            arg_text = "no explicit constructor inputs"
        return f"- {self.name}: {self.description} Inputs: {arg_text}."


@dataclass(frozen=True)
class ParsedAtomicClass:
    name: str
    base_names: tuple[str, ...]
    docstring: str
    constructor_args: tuple[ConstructorArg, ...]
    source_path: str
    signature_text: str
    is_abstract: bool


def format_constructor_arg(arg: ConstructorArg) -> str:
    if arg.required:
        return arg.name
    return f"{arg.name}={arg.default}"


def _expr_to_text(node: ast.AST | None) -> str | None:
    if node is None:
        return None
    return ast.unparse(node)


def _base_name(base: ast.expr) -> str:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return ast.unparse(base)


def _contains_not_implemented(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Raise):
            continue
        exc = child.exc
        if exc is None:
            continue
        if isinstance(exc, ast.Call):
            exc = exc.func
        if isinstance(exc, ast.Name) and exc.id == "NotImplementedError":
            return True
    return False


def _parse_constructor_args(class_node: ast.ClassDef) -> tuple[ConstructorArg, ...]:
    init_node = next(
        (
            node
            for node in class_node.body
            if isinstance(node, ast.FunctionDef) and node.name == "__init__"
        ),
        None,
    )
    if init_node is None:
        return ()

    args: list[ConstructorArg] = []
    init_args = init_node.args
    positional = list(init_args.posonlyargs) + list(init_args.args)
    # Only surface task-facing constructor inputs in the prompt catalog.
    if positional and positional[0].arg == "self":
        positional = positional[1:]

    positional_defaults = [None] * (len(positional) - len(init_args.defaults)) + list(
        init_args.defaults
    )
    for arg_node, default_node in zip(positional, positional_defaults):
        if arg_node.arg in {"args", "kwargs"}:
            continue
        args.append(
            ConstructorArg(
                name=arg_node.arg,
                default=_expr_to_text(default_node),
                required=default_node is None,
            )
        )

    for arg_node, default_node in zip(init_args.kwonlyargs, init_args.kw_defaults):
        if arg_node.arg in {"args", "kwargs"}:
            continue
        args.append(
            ConstructorArg(
                name=arg_node.arg,
                default=_expr_to_text(default_node),
                required=default_node is None,
                kind="keyword_only",
            )
        )

    return tuple(args)


def _parse_atomic_source(path: Path) -> list[ParsedAtomicClass]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parsed_classes: list[ParsedAtomicClass] = []

    for node in module.body:
        if not isinstance(node, ast.ClassDef):
            continue
        constructor_args = _parse_constructor_args(node)
        signature_args = ", ".join(
            format_constructor_arg(arg) for arg in constructor_args
        )
        signature_text = f"{node.name}({signature_args})" if signature_args else f"{node.name}()"
        docstring = ast.get_docstring(node) or ""
        parsed_classes.append(
            ParsedAtomicClass(
                name=node.name,
                base_names=tuple(_base_name(base) for base in node.bases),
                docstring=" ".join(docstring.strip().split()),
                constructor_args=constructor_args,
                source_path=str(path.relative_to(REPO_ROOT)),
                signature_text=signature_text,
                is_abstract=_contains_not_implemented(node),
            )
        )

    return parsed_classes


def _is_concrete_tool(
    parsed_class: ParsedAtomicClass,
    subclass_index: dict[str, set[str]],
) -> bool:
    # Keep only concrete leaf classes so the catalog matches executable tasks.
    if parsed_class.name.startswith("_"):
        return False
    if parsed_class.name in HELPER_CLASS_NAMES:
        return False
    if parsed_class.is_abstract:
        return False
    if subclass_index.get(parsed_class.name):
        return False
    return True


def _normalize_description(parsed_class: ParsedAtomicClass) -> str:
    if parsed_class.docstring:
        return parsed_class.docstring.split(". ")[0].rstrip(".") + "."
    return f"Atomic RoboCasa task {parsed_class.name}."


@lru_cache(maxsize=1)
def discover_atomic_tools() -> tuple[AtomicToolSpec, ...]:
    # Cache the parsed catalog because prompt construction hits this repeatedly.
    parsed_classes: list[ParsedAtomicClass] = []
    for path in sorted(ATOMIC_TASK_DIR.glob("*.py")):
        if path.name == "__init__.py":
            continue
        parsed_classes.extend(_parse_atomic_source(path))

    subclass_index: dict[str, set[str]] = {}
    for parsed_class in parsed_classes:
        for base_name in parsed_class.base_names:
            subclass_index.setdefault(base_name, set()).add(parsed_class.name)

    concrete_tools = [
        AtomicToolSpec(
            name=parsed_class.name,
            description=_normalize_description(parsed_class),
            constructor_args=parsed_class.constructor_args,
            source_path=parsed_class.source_path,
            signature_text=parsed_class.signature_text,
        )
        for parsed_class in parsed_classes
        if _is_concrete_tool(parsed_class, subclass_index)
    ]
    return tuple(sorted(concrete_tools, key=lambda tool: tool.name))


def render_atomic_tool_catalog(tool_specs: tuple[AtomicToolSpec, ...] | None = None) -> str:
    if tool_specs is None:
        tool_specs = discover_atomic_tools()
    return "\n".join(tool.to_prompt_block() for tool in tool_specs)


def render_synthetic_communication_tool() -> str:
    return (
        "- communicate: Send a short coordination message to the other agent. "
        "Inputs: to_agent_id, message."
    )
