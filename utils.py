"""
Validation helpers and formatting for the dashboard.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class ValidationResult:
    ok: bool
    error_message: Optional[str] = None


def validate_optimization_inputs(
    learning_rate: float,
    max_iterations: int,
    tolerance: float,
) -> ValidationResult:
    if learning_rate <= 0:
        return ValidationResult(
            ok=False,
            error_message="Learning rate must be a **positive** number. "
            "Try a small value (e.g. 0.05–0.2) for stable descent.",
        )
    if max_iterations <= 0:
        return ValidationResult(
            ok=False,
            error_message="Maximum iterations must be **greater than zero**.",
        )
    if tolerance <= 0:
        return ValidationResult(
            ok=False,
            error_message="Tolerance must be **positive** (e.g. 1e-6). "
            "It controls how small the gradient norm must be before stopping.",
        )
    return ValidationResult(ok=True)


def parse_float(value: str, default: float) -> Tuple[float, Optional[str]]:
    try:
        return float(value), None
    except (TypeError, ValueError):
        return default, "Invalid numeric input."


def parse_int(value: str, default: int) -> Tuple[int, Optional[str]]:
    try:
        return int(float(value)), None
    except (TypeError, ValueError):
        return default, "Invalid integer input."
