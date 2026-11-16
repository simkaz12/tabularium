"""Readability metrics for a Python source directory.

Usage::

    uv run python scripts/code_metrics.py src/tabularium

The directory is resolved against the repository root. The first output line is
the headline: file, line and function counts.
"""

from __future__ import annotations

import ast
import collections
import hashlib
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

NEST = (
    ast.If,
    ast.For,
    ast.While,
    ast.With,
    ast.Try,
    ast.AsyncFor,
    ast.AsyncWith,
    ast.Match,
)
FUNC = (ast.FunctionDef, ast.AsyncFunctionDef)
SCOPES = (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda, ast.ClassDef)
CLOSED = re.compile(
    r"(kind|status|lifecycle|mode|outcome|verdict|action|state|tier|phase|role|level"
    r"|severity|reason|type|category|stage|verb|op|direction|source)$"
)
LOOSE_DICT = re.compile(r"(dict|Dict|Mapping|MutableMapping)\[str, (Any|object)\]")
LOOSE_RETURN = re.compile(r"(dict|Dict|Mapping)\[str, (Any|object)\]")
TYPE_IGNORE = re.compile(r"type: ignore(?:\[([^\]]*)\])?")
LOOSE_KEYS = [
    "Any",
    "cast",
    "type_ignore",
    "object_annot",
    "dict_str_Any_or_object",
    "TypedDict_def",
    "Literal",
    "str_param_closed_set",
    "strkey_idx",
    "dict_get_strkey",
]


@dataclass(frozen=True, slots=True)
class FuncInfo:
    module: str
    name: str
    line: int
    length: int
    depth: int
    params: int


@dataclass(slots=True)
class Collected:
    funcs: list[FuncInfo] = field(default_factory=list)
    modules: dict[str, tuple[int, int, list[tuple[int, str]]]] = field(default_factory=dict)
    loose: dict[str, collections.Counter[str]] = field(
        default_factory=lambda: collections.defaultdict(collections.Counter)
    )
    sites: dict[str, list[tuple[str, int, str]]] = field(
        default_factory=lambda: collections.defaultdict(list)
    )
    ignore_codes: list[str] = field(default_factory=list)
    shapes: collections.Counter[str] = field(default_factory=collections.Counter)
    excepts: dict[str, list[tuple[int, str, str]]] = field(default_factory=dict)
    duplicates: dict[str, list[tuple[str, str, int, int]]] = field(default_factory=dict)
    chains: list[tuple[str, int, int]] = field(default_factory=list)

    def mark(self, module: str, key: str, line: int, text: str) -> None:
        self.loose[module][key] += 1
        self.sites[key].append((module, line, text.strip()[:110]))


def nesting_depth(node: ast.AST, depth: int = 0) -> int:
    deepest = depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, SCOPES):
            continue
        child_depth = depth + 1 if isinstance(child, NEST) else depth
        deepest = max(deepest, nesting_depth(child, child_depth))
    return deepest


def unparse(node: ast.AST | None) -> str:
    return ast.unparse(node) if node is not None else ""


def strip_docstring(body: list[ast.stmt]) -> list[ast.stmt]:
    if body and isinstance(body[0], ast.Expr):
        value = body[0].value
        if isinstance(value, ast.Constant) and isinstance(value.value, str):
            return body[1:]
    return body


def is_ellipsis_body(body: list[ast.stmt]) -> bool:
    if len(body) != 1 or not isinstance(body[0], ast.Expr):
        return False
    value = body[0].value
    return isinstance(value, ast.Constant) and value.value is Ellipsis


def param_names(args: ast.arguments) -> list[str]:
    names = [a.arg for a in args.posonlyargs + args.args + args.kwonlyargs]
    if args.vararg:
        names.append(args.vararg.arg)
    if args.kwarg:
        names.append(args.kwarg.arg)
    return [n for n in names if n not in ("self", "cls")]


def collect_function(
    out: Collected,
    module: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> None:
    length = (node.end_lineno or node.lineno) - node.lineno + 1
    out.funcs.append(
        FuncInfo(
            module=module,
            name=node.name,
            line=node.lineno,
            length=length,
            depth=nesting_depth(node),
            params=len(param_names(node.args)),
        )
    )
    out.modules[module][2].append((length, node.name))
    if length > 10:
        body = strip_docstring(node.body)
        if is_ellipsis_body(body):
            return
        dump = ast.dump(ast.Module(body=body, type_ignores=[]), annotate_fields=False)
        digest = hashlib.sha256(dump.encode()).hexdigest()
        out.duplicates.setdefault(digest, []).append((module, node.name, node.lineno, length))
    collect_signature(out, module, node)


def collect_signature(
    out: Collected,
    module: str,
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> None:
    args = node.args
    for arg in args.posonlyargs + args.args + args.kwonlyargs:
        annotation = unparse(arg.annotation)
        if arg.annotation is not None and annotation == "str" and CLOSED.search(arg.arg):
            out.mark(module, "str_param_closed_set", node.lineno, f"{node.name}({arg.arg}: str)")
        if annotation == "object":
            out.mark(module, "object_annot", arg.lineno, f"{node.name} param {arg.arg}: object")
    returns = unparse(node.returns)
    if returns == "object":
        out.mark(module, "object_annot", node.lineno, f"{node.name} -> object")
    if LOOSE_RETURN.match(returns) or returns in ("dict", "dict[str, Any]"):
        out.shapes["returns_dict_annot"] += 1
    if returns == "str" and CLOSED.search(node.name.split("_")[-1]):
        out.mark(module, "str_return_closed_name", node.lineno, node.name)


def collect_class(out: Collected, module: str, node: ast.ClassDef) -> None:
    bases = [unparse(b) for b in node.bases]
    decorators = [unparse(d) for d in node.decorator_list]
    if any("dataclass" in d for d in decorators):
        frozen = any("frozen=True" in d for d in decorators)
        out.shapes["dataclass_frozen" if frozen else "dataclass_mutable"] += 1
    if any("NamedTuple" in b for b in bases):
        out.shapes["NamedTuple"] += 1
    if any("TypedDict" in b for b in bases):
        out.shapes["TypedDict"] += 1
        out.mark(module, "TypedDict_def", node.lineno, node.name)
    if any(b.endswith(("Enum", "StrEnum", "IntEnum")) for b in bases):
        out.shapes["Enum"] += 1
    if any("BaseModel" in b for b in bases):
        out.shapes["pydantic"] += 1
    if any("Protocol" in b for b in bases):
        out.shapes["Protocol"] += 1


def classify_handler(handler: ast.ExceptHandler) -> tuple[str, str] | None:
    caught = unparse(handler.type) if handler.type else "bare"
    if handler.type is not None and caught not in ("Exception", "BaseException"):
        return None
    body = handler.body
    first = body[0]
    trivial = isinstance(first, ast.Pass | ast.Continue) or (
        isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant)
    )
    returns = any(isinstance(n, ast.Return) for n in body)
    walked = ast.walk(ast.Module(body=body, type_ignores=[]))
    reraises = any(isinstance(n, ast.Raise) for n in walked)
    if len(body) == 1 and trivial:
        category = "pass"
    elif returns and not reraises:
        category = "return_default"
    elif reraises:
        category = "reraise"
    else:
        category = "other(log/handle)"
    return caught, category


def isinstance_chain(node: ast.If) -> int:
    count = 0
    current = node
    while "isinstance" in ast.unparse(current.test):
        count += 1
        if len(current.orelse) == 1 and isinstance(current.orelse[0], ast.If):
            current = current.orelse[0]
        else:
            break
    return count


def collect_node(out: Collected, module: str, lines: list[str], node: ast.AST) -> None:
    if isinstance(node, FUNC):
        collect_function(out, module, node)
    if isinstance(node, ast.ClassDef):
        collect_class(out, module, node)
    if isinstance(node, ast.Try):
        for handler in node.handlers:
            found = classify_handler(handler)
            if found is not None:
                out.excepts.setdefault(module, []).append((handler.lineno, found[0], found[1]))
    if isinstance(node, ast.Subscript) and unparse(node.value) in ("Literal", "typing.Literal"):
        out.mark(module, "Literal", node.lineno, ast.unparse(node))
    if isinstance(node, ast.Call) and unparse(node.func) in ("cast", "typing.cast"):
        out.mark(module, "cast", node.lineno, lines[node.lineno - 1])
    if (isinstance(node, ast.Name) and node.id == "Any") or (
        isinstance(node, ast.Attribute) and node.attr == "Any"
    ):
        out.mark(module, "Any", node.lineno, lines[node.lineno - 1])
    if isinstance(node, ast.If):
        chain = isinstance_chain(node)
        if chain >= 3:
            out.chains.append((module, node.lineno, chain))


def collect_dict_edges(out: Collected, module: str, lines: list[str], tree: ast.AST) -> None:
    for node in ast.walk(tree):
        annotations: list[ast.expr] = []
        if isinstance(node, ast.arg) and node.annotation:
            annotations.append(node.annotation)
        if isinstance(node, ast.AnnAssign):
            annotations.append(node.annotation)
        if isinstance(node, FUNC) and node.returns:
            annotations.append(node.returns)
        for annotation in annotations:
            if LOOSE_DICT.search(ast.unparse(annotation)):
                out.mark(
                    module,
                    "dict_str_Any_or_object",
                    annotation.lineno,
                    lines[annotation.lineno - 1],
                )
        if isinstance(node, ast.Return) and isinstance(node.value, ast.Dict):
            out.shapes["return_dict_literal"] += 1
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Constant)
            and isinstance(node.slice.value, str)
            and isinstance(node.ctx, ast.Load)
        ):
            out.shapes["str_key_subscript_load"] += 1
            out.loose[module]["strkey_idx"] += 1
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "get"
            and node.args
            and isinstance(node.args[0], ast.Constant)
            and isinstance(node.args[0].value, str)
        ):
            out.loose[module]["dict_get_strkey"] += 1


def collect_file(out: Collected, path: Path) -> None:
    source = path.read_text()
    module = str(path.relative_to(ROOT))
    tree = ast.parse(source)
    lines = source.splitlines()
    tops = [n for n in tree.body if isinstance(n, (*FUNC, ast.ClassDef))]
    out.modules[module] = (len(lines), len(tops), [])
    for node in ast.walk(tree):
        collect_node(out, module, lines, node)
    collect_dict_edges(out, module, lines, tree)
    for number, line in enumerate(lines, 1):
        if "# type: ignore" in line:
            found = TYPE_IGNORE.findall(line)
            out.mark(module, "type_ignore", number, line)
            out.ignore_codes.append((found[0] or "(bare)") if found else "(bare)")


def print_headline(target: str, out: Collected) -> None:
    total = sum(v[0] for v in out.modules.values())
    print(f"=== {target}: {len(out.modules)} files, {total} lines, {len(out.funcs)} funcs")


def print_modules(out: Collected) -> None:
    print("\n## 1 per module (lines, top-level defs, longest fn)")
    for module, (count, tops, funcs) in sorted(out.modules.items(), key=lambda x: -x[1][0]):
        longest = max(funcs, default=(0, "-"))
        print(f"{module}|{count}|{tops}|{longest[1]} ({longest[0]})")


def print_distribution(out: Collected) -> None:
    print("\n## 2 distribution")
    for threshold in (40, 80, 150):
        print(f">{threshold}: {sum(1 for f in out.funcs if f.length > threshold)}")
    print("top25 longest")
    for f in sorted(out.funcs, key=lambda f: -f.length)[:25]:
        print(f"{f.module}:{f.line}|{f.name}|{f.length}")


def print_nesting_and_params(out: Collected) -> None:
    print("\n## 3 nesting")
    histogram = collections.Counter(f.depth for f in out.funcs)
    print("depth histogram", sorted(histogram.items()))
    for f in sorted(out.funcs, key=lambda f: (-f.depth, -f.length))[:25]:
        print(f"{f.module}:{f.line}|{f.name}|depth {f.depth}|{f.length}L")
    print("\n## 4 params>6")
    wide = [f for f in out.funcs if f.params > 6]
    print("count", len(wide))
    for f in sorted(wide, key=lambda f: -f.params)[:20]:
        print(f"{f.module}:{f.line}|{f.name}|{f.params}")


def print_looseness(out: Collected) -> None:
    print("\n## 5 looseness totals")
    totals: collections.Counter[str] = collections.Counter()
    for counter in out.loose.values():
        totals.update(counter)
    print(dict(totals))
    print("per module (top by total):")
    print("module|" + "|".join(LOOSE_KEYS))
    ranked = sorted(out.loose.items(), key=lambda x: -sum(x[1][k] for k in LOOSE_KEYS[:9]))
    for module, counter in ranked:
        if sum(counter[k] for k in LOOSE_KEYS) == 0:
            continue
        print(module + "|" + "|".join(str(counter[k]) for k in LOOSE_KEYS))
    print("type ignore codes", collections.Counter(out.ignore_codes).most_common())
    for key in ("type_ignore", "cast", "object_annot", "str_param_closed_set", "Literal"):
        print(f"\n-- {key} lines (up to 25 of {len(out.sites[key])})")
        for module, line, text in out.sites[key][:25]:
            print(f"{module}:{line}|{text}")
    print(f"\n-- TypedDict_def lines (up to 25 of {len(out.sites['TypedDict_def'])})")
    for module, line, text in out.sites["TypedDict_def"][:25]:
        print(f"{module}:{line}|{text}")
    print("\n-- Any lines sample (first 15 of", len(out.sites["Any"]), ")")
    for module, line, text in out.sites["Any"][:15]:
        print(f"{module}:{line}|{text}")
    loose_dicts = out.sites["dict_str_Any_or_object"]
    print("-- dict[str,Any/object] sample (first 20 of", len(loose_dicts), ")")
    for module, line, text in loose_dicts[:20]:
        print(f"{module}:{line}|{text}")
    print("-- isinstance chains >=3:", len(out.chains))
    for chain in out.chains:
        print(chain)


def print_shapes_and_excepts(out: Collected) -> None:
    print("\n## 6 shapes", dict(out.shapes))
    sets = collections.Counter(text for _, _, text in out.sites["Literal"])
    print("distinct Literal sets", len(sets))
    print(sets.most_common(15))
    print("\n## 7 broad excepts")
    categories: collections.Counter[tuple[str, str]] = collections.Counter()
    for handlers in out.excepts.values():
        for _, caught, category in handlers:
            categories[(caught, category)] += 1
    print(dict(categories))
    per_module = {m: len(h) for m, h in sorted(out.excepts.items(), key=lambda x: -len(x[1]))}
    print("per module", per_module)
    print("pass/return_default sites:")
    for module, handlers in out.excepts.items():
        for line, caught, category in handlers:
            if category in ("pass", "return_default"):
                print(f"{module}:{line}|{caught}|{category}")


def print_duplicates(out: Collected) -> None:
    print("\n## 9 duplicates (identical body AST, >10 lines)")
    for sites in out.duplicates.values():
        if len(sites) > 1:
            print(sites)


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: code_metrics.py <directory>", file=sys.stderr)
        return 2
    target = argv[1]
    root = Path(target)
    directory = root if root.is_absolute() else ROOT / root
    out = Collected()
    for path in sorted(directory.rglob("*.py")):
        collect_file(out, path)
    print_headline(target, out)
    print_modules(out)
    print_distribution(out)
    print_nesting_and_params(out)
    print_looseness(out)
    print_shapes_and_excepts(out)
    print_duplicates(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
