"""Sandboxed execution of VLM-generated ReKep constraint functions.

The reference plugin executes constraint text with ``utils.exec_safe`` which only
bans the substrings ``import`` and ``__`` and then calls ``exec``. That is not an
adequate security boundary. This module replaces it with a strict AST allow-list
that is *safe by construction*: loops, comprehensions, imports, lambdas,
``if``-statements, dunder access, and calls other than ``np.*`` and a small set
of pure builtins / the provided grasping-cost function are all rejected.

Contract with the VLM prompt (see ``reference/.../vlm_query/prompt_template.txt``):
    def stageN_subgoal_constraintM(end_effector, keypoints): ...; return cost
    def stageN_path_constraintM(end_effector, keypoints): ...; return cost
Constraints return a numeric cost; satisfied when ``cost <= tolerance``.
"""

from __future__ import annotations

import ast
import math
from typing import Any, Callable, Iterable

import numpy as np


class ConstraintError(ValueError):
    """Base class for rejected constraint programs."""


class ConstraintSyntaxError(ConstraintError):
    """The constraint source is not valid Python."""


class ConstraintSecurityError(ConstraintError):
    """The constraint program uses forbidden syntax or names."""


# Statements / expressions that are always rejected.
_FORBIDDEN_NODES: tuple[type[ast.AST], ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.If,
    ast.While,
    ast.For,
    ast.AsyncFor,
    ast.With,
    ast.AsyncWith,
    ast.AsyncFunctionDef,
    ast.Lambda,
    ast.Global,
    ast.Nonlocal,
    ast.ClassDef,
    ast.Try,
    ast.Raise,
    ast.Assert,
    ast.Delete,
    ast.Yield,
    ast.YieldFrom,
    ast.Await,
    ast.ListComp,
    ast.SetComp,
    ast.DictComp,
    ast.GeneratorExp,
    ast.NamedExpr,  # walrus
    ast.Match,
)

# Names that must never be *called* directly.
_DISALLOWED_CALL_NAMES = {
    "eval",
    "exec",
    "open",
    "compile",
    "__import__",
    "input",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "delattr",
    "memoryview",
    "breakpoint",
    "help",
    "exit",
    "quit",
    "print",
}

# Pure builtins the constraints may use. Deliberately excludes anything that can
# touch the environment or the interpreter.
_SAFE_BUILTINS: dict[str, Any] = {
    "abs": abs,
    "min": min,
    "max": max,
    "sum": sum,
    "len": len,
    "float": float,
    "int": int,
    "round": round,
    "bool": bool,
    "pow": pow,
    "sorted": sorted,
    "math": math,
    "True": True,
    "False": False,
    "None": None,
}

# Functions the sandbox exposes to the program.
_GRASP_FN_NAME = "get_grasping_cost_by_keypoint_idx"
_ALLOWED_BARE_CALLS = set(_SAFE_BUILTINS) | {_GRASP_FN_NAME, "np"}

# Attribute names that are rejected on any object.
_FORBIDDEN_ATTRS = {
    "f_locals",
    "f_globals",
    "f_builtins",
    "func_globals",
    "gi_frame",
    "cr_frame",
    "__globals__",
    "__builtins__",
}


def _reject(node: ast.AST, reason: str) -> None:
    raise ConstraintSecurityError(
        f"forbidden {type(node).__name__}: {reason} (line {getattr(node, 'lineno', '?')})"
    )


def _validate(tree: ast.AST) -> list[str]:
    """Validate the AST; return the names of defined functions."""
    defined: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            _reject(node, "syntax not allowed in constraints")
        if isinstance(node, ast.FunctionDef):
            defined.append(node.name)
            if node.decorator_list:
                _reject(node, "decorators are not allowed")
            args = node.args
            if args.vararg or args.kwarg or args.kwonlyargs or args.posonlyargs or args.defaults:
                _reject(node, "only plain positional parameters are allowed")
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__") or node.attr in _FORBIDDEN_ATTRS:
                _reject(node, f"attribute {node.attr!r} is forbidden")
        if isinstance(node, ast.Name):
            if node.id.startswith("__"):
                _reject(node, f"name {node.id!r} is forbidden")
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id in _DISALLOWED_CALL_NAMES or func.id not in _ALLOWED_BARE_CALLS:
                    _reject(node, f"call to {func.id!r} is not allowed")
            elif isinstance(func, ast.Attribute):
                root = func
                while isinstance(root, ast.Attribute):
                    root = root.value
                if not (isinstance(root, ast.Name) and root.id in {"np", "math"}):
                    _reject(node, "only np.* / math.* calls are allowed")
            else:
                _reject(node, "unsupported call target")
    if not defined:
        raise ConstraintSecurityError("constraint program defines no functions")
    return defined


def _grasping_cost_fn(grasped_keypoints: Iterable[int] | None):
    grasped = {int(x) for x in (grasped_keypoints or [])}

    def get_grasping_cost_by_keypoint_idx(keypoint_idx):  # noqa: N802 (VLM prompt name)
        return 0 if int(keypoint_idx) in grasped else 1

    return get_grasping_cost_by_keypoint_idx


def load_functions(
    source: str,
    *,
    grasped_keypoints: Iterable[int] | None = None,
    timeout_s: float | None = 2.0,
) -> list[Callable[[np.ndarray, np.ndarray], float]]:
    """Parse, validate, and exec a constraint program in a restricted namespace.

    Returns the list of defined functions in definition order. Raises
    ``ConstraintSyntaxError`` / ``ConstraintSecurityError`` on rejection.
    """
    if not isinstance(source, str) or not source.strip():
        raise ConstraintSyntaxError("constraint source is empty")
    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError as exc:  # pragma: no cover - message text is version dependent
        raise ConstraintSyntaxError(f"invalid Python: {exc}") from exc

    _validate(tree)

    globals_ns: dict[str, Any] = {
        "__builtins__": dict(_SAFE_BUILTINS),
        "np": np,
        "math": math,
        _GRASP_FN_NAME: _grasping_cost_fn(grasped_keypoints),
    }
    locals_ns: dict[str, Any] = {}
    code = compile(tree, "<rekep_constraints>", "exec")

    use_signal = False
    if timeout_s and timeout_s > 0:
        try:
            import signal
            import threading

            use_signal = threading.current_thread() is threading.main_thread()
        except Exception:  # pragma: no cover
            use_signal = False

    if use_signal:
        import signal

        def _handler(signum, frame):  # pragma: no cover - timing dependent
            raise ConstraintError(f"constraint execution timed out after {timeout_s}s")

        previous = signal.signal(signal.SIGALRM, _handler)
        signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
        try:
            exec(code, globals_ns, locals_ns)
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, previous)
    else:
        exec(code, globals_ns, locals_ns)

    functions: list[Callable[[np.ndarray, np.ndarray], float]] = []
    for name, value in locals_ns.items():
        if name.startswith("__"):
            continue
        if callable(value):
            functions.append(value)
    if not functions:
        raise ConstraintSecurityError("constraint program produced no callable")
    return functions


def evaluate(
    functions: Iterable[Callable[[np.ndarray, np.ndarray], float]],
    end_effector: np.ndarray,
    keypoints: np.ndarray,
    *,
    tolerance: float = 1e-6,
) -> tuple[list[float], list[str]]:
    """Evaluate constraint functions; return (costs, reasons)."""
    costs: list[float] = []
    reasons: list[str] = []
    for idx, fn in enumerate(functions, start=1):
        try:
            cost = float(fn(end_effector, keypoints))
        except Exception as exc:  # pragma: no cover - defensive
            costs.append(float("inf"))
            reasons.append(f"constraint {idx} raised: {exc}")
            continue
        costs.append(cost)
        if cost > tolerance:
            reasons.append(f"constraint {idx} violated: {cost:.6f}")
    return costs, reasons
