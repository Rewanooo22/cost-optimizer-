"""
Steepest descent optimization for the production cost surface f(x, y).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd


def cost_function(x: float, y: float) -> float:
    
    return float(x * x + y * y - 4.0 * x - 6.0 * y + 13.0)


def gradient(x: float, y: float) -> Tuple[float, float]:
    """∇f = (2x - 4, 2y - 6)"""
    return float(2.0 * x - 4.0), float(2.0 * y - 6.0)


@dataclass
class OptimizationResult:
    converged: bool
    iterations: int
    x_opt: float
    y_opt: float
    cost_opt: float
    history: pd.DataFrame
    message: str


def steepest_descent(
    x0: float,
    y0: float,
    learning_rate: float,
    max_iterations: int,
    tolerance: float,
) -> OptimizationResult:
    """
    Minimize f using steepest descent:
    (x, y)_{k+1} = (x, y)_k - η * ∇f(x_k, y_k)
    Stops when ||∇f|| < tolerance or max_iterations reached.
    """
    x, y = float(x0), float(y0)
    rows: List[dict] = []

    for k in range(max_iterations):
        gx, gy = gradient(x, y)
        grad_norm = float(np.sqrt(gx * gx + gy * gy))
        fx = cost_function(x, y)
        rows.append(
            {
                "iteration": k,
                "x": x,
                "y": y,
                "cost": fx,
                "grad_x": gx,
                "grad_y": gy,
                "grad_norm": grad_norm,
            }
        )
        if grad_norm < tolerance:
            return OptimizationResult(
                converged=True,
                iterations=k + 1,
                x_opt=x,
                y_opt=y,
                cost_opt=fx,
                history=pd.DataFrame(rows),
                message=f"Converged: gradient norm {grad_norm:.2e} < tolerance {tolerance}.",
            )
        x -= learning_rate * gx
        y -= learning_rate * gy

    gx, gy = gradient(x, y)
    grad_norm = float(np.sqrt(gx * gx + gy * gy))
    fx = cost_function(x, y)
    rows.append(
        {
            "iteration": max_iterations,
            "x": x,
            "y": y,
            "cost": fx,
            "grad_x": gx,
            "grad_y": gy,
            "grad_norm": grad_norm,
        }
    )
    return OptimizationResult(
        converged=False,
        iterations=max_iterations,
        x_opt=x,
        y_opt=y,
        cost_opt=fx,
        history=pd.DataFrame(rows),
        message="Maximum iterations reached without meeting the gradient tolerance.",
    )


def analytical_minimum() -> Tuple[float, float, float]:
    """Known optimum: x=2, y=3, f=0."""
    return 2.0, 3.0, 0.0
