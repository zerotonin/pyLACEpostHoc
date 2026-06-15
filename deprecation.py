# ╔══════════════════════════════════════════════════════════════════╗
# ║  pyLACEpostHoc — deprecation                                     ║
# ║  « snake_case rename without breaking old callers »              ║
# ╠══════════════════════════════════════════════════════════════════╣
# ║  deprecated_alias() wraps a renamed callable so the old          ║
# ║  camelCase name keeps working while emitting a                   ║
# ║  DeprecationWarning pointing at the new name.                    ║
# ╚══════════════════════════════════════════════════════════════════╝
"""Backward-compatible aliasing for the camelCase → snake_case rename."""
from __future__ import annotations

import functools
import warnings
from typing import Callable, TypeVar

F = TypeVar("F", bound=Callable[..., object])


def deprecated_alias(new_callable: F, old_name: str | None = None) -> F:
    """Wrap ``new_callable`` so calls warn and forward to the new name.

    Use at class or module scope to keep a renamed identifier alive::

        class Foo:
            def compute_thing(self): ...
            computeThing = deprecated_alias(compute_thing)

    Args:
        new_callable: The renamed function/method to forward to.
        old_name:     Deprecated name for the message; defaults to the
                      new callable's ``__name__``.

    Returns:
        A wrapper that emits ``DeprecationWarning`` then delegates.
    """
    new_name = getattr(new_callable, "__name__", repr(new_callable))
    shown_old = old_name or new_name

    @functools.wraps(new_callable)
    def wrapper(*args, **kwargs):
        warnings.warn(
            f"{shown_old!r} is deprecated; use {new_name!r} instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return new_callable(*args, **kwargs)

    wrapper.__doc__ = f"Deprecated alias for :func:`{new_name}`."
    return wrapper  # type: ignore[return-value]
