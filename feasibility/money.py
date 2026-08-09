"""Integer-cent money helpers.

The spec mandates round-half-up (ties away from zero). Python's built-in
``round`` is round-half-to-even, so every ``round(...)`` in the spec is
implemented here explicitly instead.

Percentages arrive as JSON floats (``0.125``, ``0.65``). Multiplying a float by
an int and then rounding is doubly unsafe: the float may not be exactly the
decimal the author wrote, and a product landing on ``x.5`` may land just below
it. Both are avoided by going through ``Decimal(str(pct))``, which recovers the
decimal literal as written.
"""

from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal


def round_half_up(value: Decimal | float | int) -> int:
    """Round to the nearest integer, with ``.5`` always going away from zero."""
    if not isinstance(value, Decimal):
        value = Decimal(str(value))
    return int(value.quantize(Decimal(1), rounding=ROUND_HALF_UP))


def pct_of_cents(pct: float, cents: int) -> int:
    """``round_half_up(pct * cents)`` without float drift."""
    return round_half_up(Decimal(str(pct)) * Decimal(int(cents)))
