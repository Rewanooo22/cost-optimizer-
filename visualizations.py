"""
Plotly (interactive) and Matplotlib figures for the cost surface and optimization path.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from optimizer import analytical_minimum, cost_function


def mesh_grid(x_range: tuple[float, float], y_range: tuple[float, float], n: int = 80):
    xs = np.linspace(x_range[0], x_range[1], n)
    ys = np.linspace(y_range[0], y_range[1], n)
    X, Y = np.meshgrid(xs, ys)
    Z = X**2 + Y**2 - 4 * X - 6 * Y + 13
    return X, Y, Z


def plot_cost_convergence(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["iteration"],
            y=history["cost"],
            mode="lines+markers",
            name="Cost f(x,y)",
            line=dict(color="#22d3ee", width=2),
            marker=dict(size=6, color="#06b6d4"),
        )
    )
    xa, ya, fa = analytical_minimum()
    fig.add_hline(
        y=fa,
        line_dash="dash",
        line_color="#94a3b8",
        annotation_text="Analytical minimum (0)",
        annotation_position="bottom right",
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        margin=dict(l=50, r=20, t=40, b=50),
        title=dict(text="Cost Convergence", font=dict(size=16, color="#e2e8f0")),
        xaxis_title="Iteration",
        yaxis_title="Cost",
        font=dict(color="#cbd5e1"),
        hovermode="x unified",
    )
    return fig


def plot_contour_with_path(
    history: pd.DataFrame,
    x_range: tuple[float, float] = (-0.5, 4.5),
    y_range: tuple[float, float] = (-0.5, 6.5),
) -> go.Figure:
    X, Y, Z = mesh_grid(x_range, y_range, n=90)
    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=X[0, :],
            y=Y[:, 0],
            z=Z,
            colorscale="Blues",
            name="f(x,y)",
            contours=dict(coloring="fill", showlabels=True),
            line=dict(width=0.5, color="#0f172a"),
            showscale=True,
            colorbar=dict(
                title=dict(text="Cost", font=dict(color="#e2e8f0")),
                tickfont=dict(color="#cbd5e1"),
            ),
        )
    )
    if history is not None and not history.empty:
        fig.add_trace(
            go.Scatter(
                x=history["x"],
                y=history["y"],
                mode="lines+markers",
                name="Optimization path",
                line=dict(color="#22d3ee", width=3),
                marker=dict(size=7, color="#67e8f9", line=dict(width=1, color="#0e7490")),
            )
        )
    xa, ya, _ = analytical_minimum()
    fig.add_trace(
        go.Scatter(
            x=[xa],
            y=[ya],
            mode="markers",
            name="True optimum (2, 3)",
            marker=dict(size=14, color="#fbbf24", symbol="star", line=dict(width=1, color="#78350f")),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        margin=dict(l=50, r=20, t=40, b=50),
        title=dict(text="2D Contour & Descent Path", font=dict(size=16, color="#e2e8f0")),
        xaxis_title="Labor hours (x)",
        yaxis_title="Machine hours (y)",
        font=dict(color="#cbd5e1"),
        hovermode="closest",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_surface_3d(
    history: Optional[pd.DataFrame] = None,
    x_range: tuple[float, float] = (-0.5, 4.5),
    y_range: tuple[float, float] = (-0.5, 6.5),
) -> go.Figure:
    X, Y, Z = mesh_grid(x_range, y_range, n=55)
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            x=X,
            y=Y,
            z=Z,
            colorscale="Blues",
            showscale=True,
            lighting=dict(ambient=0.45, diffuse=0.85, specular=0.2),
            colorbar=dict(
                title=dict(text="Cost", font=dict(color="#e2e8f0")),
                tickfont=dict(color="#cbd5e1"),
            ),
        )
    )
    if history is not None and not history.empty:
        zh = history.apply(lambda r: cost_function(r["x"], r["y"]), axis=1).values
        fig.add_trace(
            go.Scatter3d(
                x=history["x"],
                y=history["y"],
                z=zh,
                mode="lines+markers",
                name="Path on surface",
                line=dict(color="#22d3ee", width=8),
                marker=dict(size=4, color="#67e8f9"),
            )
        )
    xa, ya, za = analytical_minimum()
    fig.add_trace(
        go.Scatter3d(
            x=[xa],
            y=[ya],
            z=[za],
            mode="markers",
            name="Optimum",
            marker=dict(size=10, color="#fbbf24", symbol="diamond"),
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
        title=dict(text="3D Cost Surface", font=dict(size=16, color="#e2e8f0")),
        scene=dict(
            xaxis_title="Labor x",
            yaxis_title="Machine y",
            zaxis_title="Cost",
            bgcolor="rgba(15,23,42,0.85)",
            xaxis=dict(gridcolor="#334155", color="#cbd5e1"),
            yaxis=dict(gridcolor="#334155", color="#cbd5e1"),
            zaxis=dict(gridcolor="#334155", color="#cbd5e1"),
        ),
        font=dict(color="#cbd5e1"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    return fig


def plot_before_after(
    x0: float,
    y0: float,
    x_opt: float,
    y_opt: float,
) -> go.Figure:
    c0 = cost_function(x0, y0)
    c1 = cost_function(x_opt, y_opt)
    fig = go.Figure(
        data=[
            go.Bar(
                x=["Initial guess", "After optimization"],
                y=[c0, c1],
                marker_color=["#64748b", "#22d3ee"],
                text=[f"{c0:.4f}", f"{c1:.4f}"],
                textposition="outside",
            )
        ]
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        margin=dict(l=50, r=20, t=40, b=60),
        title=dict(text="Before vs After — Total Cost", font=dict(size=16, color="#e2e8f0")),
        yaxis_title="Cost f(x, y)",
        font=dict(color="#cbd5e1"),
        showlegend=False,
    )
    return fig


def plot_gradient_norm(history: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["iteration"],
            y=history["grad_norm"],
            mode="lines",
            fill="tozeroy",
            line=dict(color="#38bdf8"),
            name="‖∇f‖",
        )
    )
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.6)",
        margin=dict(l=50, r=20, t=40, b=50),
        title=dict(text="Gradient Norm vs Iteration", font=dict(size=16, color="#e2e8f0")),
        xaxis_title="Iteration",
        yaxis_title="‖∇f‖",
        font=dict(color="#cbd5e1"),
    )
    return fig


def matplotlib_cost_surface_static():
    """Optional static figure for environments that prefer Matplotlib."""
    import matplotlib.pyplot as plt

    with plt.style.context("dark_background"):
        X, Y, Z = mesh_grid((-0.5, 4.5), (-0.5, 6.5), n=60)
        fig, ax = plt.subplots(figsize=(6, 5))
        cs = ax.contourf(X, Y, Z, levels=25, cmap="Blues_r")
        ax.set_xlabel("Labor hours (x)")
        ax.set_ylabel("Machine hours (y)")
        ax.set_title("Cost contours (Matplotlib)")
        fig.colorbar(cs, ax=ax, label="Cost")
        fig.tight_layout()
    return fig
