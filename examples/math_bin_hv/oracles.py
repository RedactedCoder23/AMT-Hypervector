"""Utility oracles for the math/bin demo."""

from __future__ import annotations

import math

__all__ = ["is_valid_math", "checksum_mod7"]


def is_valid_math(expr: str) -> bool:
    """Return True if ``expr`` like ``a+b=c`` evaluates correctly."""
    try:
        left, right = expr.split("=")
        a_str, b_str = left.split("+")
        a = int(a_str)
        b = int(b_str)
        c = int(right)
    except ValueError:
        return False
    return math.isclose(a + b, c)


def checksum_mod7(hexstr: str) -> bool:
    """Return True if byte sum of ``hexstr`` is divisible by 7."""
    try:
        data = bytes.fromhex(hexstr)
    except ValueError:
        return False
    return sum(data) % 7 == 0
