"""
AI-Powered Production Cost Optimizer — Streamlit industrial dashboard.
"""

from __future__ import annotations

import streamlit as st

from optimizer import analytical_minimum, cost_function, steepest_descent
from utils import validate_optimization_inputs
from visualizations import (
    matplotlib_cost_surface_static,
    plot_before_after,
    plot_contour_with_path,
    plot_cost_convergence,
    plot_gradient_norm,
    plot_surface_3d,
)


def inject_theme_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

            html, body, [class*="css"]  {
                font-family: 'DM Sans', system-ui, sans-serif;
            }

            .stApp {
                background: radial-gradient(1200px 600px at 10% -10%, #0c4a6e 0%, transparent 55%),
                            radial-gradient(900px 500px at 100% 0%, #164e63 0%, transparent 50%),
                            linear-gradient(180deg, #020617 0%, #0f172a 35%, #020617 100%);
                color: #e2e8f0;
            }

            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}

            .block-container {
                padding-top: 1.25rem;
                padding-bottom: 3rem;
                max-width: 1200px;
            }

            div[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #0b1220 0%, #0f172a 100%);
                border-right: 1px solid rgba(56, 189, 248, 0.15);
            }
            div[data-testid="stSidebar"] .block-container {
                padding-top: 1.5rem;
            }

            .hero-wrap {
                background: linear-gradient(135deg, rgba(14, 165, 233, 0.12) 0%, rgba(15, 23, 42, 0.9) 45%, rgba(6, 182, 212, 0.08) 100%);
                border: 1px solid rgba(56, 189, 248, 0.25);
                border-radius: 18px;
                padding: 1.75rem 1.75rem 1.5rem 1.75rem;
                margin-bottom: 1.25rem;
                box-shadow: 0 18px 50px rgba(2, 6, 23, 0.55), inset 0 1px 0 rgba(255,255,255,0.04);
            }
            .hero-title {
                font-size: 1.85rem;
                font-weight: 700;
                letter-spacing: -0.02em;
                color: #f8fafc;
                margin: 0 0 0.35rem 0;
            }
            .hero-sub {
                font-size: 1rem;
                color: #94a3b8;
                margin: 0;
                max-width: 820px;
                line-height: 1.55;
            }
            .hero-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                color: #67e8f9;
                border: 1px solid rgba(103, 232, 249, 0.35);
                border-radius: 999px;
                padding: 0.25rem 0.65rem;
                margin-bottom: 0.75rem;
                background: rgba(8, 47, 73, 0.45);
            }

            .card {
                background: rgba(15, 23, 42, 0.72);
                border: 1px solid rgba(148, 163, 184, 0.18);
                border-radius: 16px;
                padding: 1.15rem 1.25rem;
                margin-bottom: 1rem;
                box-shadow: 0 12px 30px rgba(2, 6, 23, 0.45);
            }
            .card h3 {
                margin: 0 0 0.5rem 0;
                font-size: 1.05rem;
                color: #e2e8f0;
            }
            .card p, .card li {
                color: #cbd5e1;
                font-size: 0.95rem;
                line-height: 1.55;
            }

            .metric-grid {
                display: grid;
                grid-template-columns: repeat(4, minmax(0, 1fr));
                gap: 0.75rem;
                margin: 0.5rem 0 1rem 0;
            }
            @media (max-width: 1100px) {
                .metric-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); }
            }
            @media (max-width: 600px) {
                .metric-grid { grid-template-columns: 1fr; }
            }

            .metric-card {
                background: linear-gradient(160deg, rgba(30, 41, 59, 0.95) 0%, rgba(15, 23, 42, 0.95) 100%);
                border: 1px solid rgba(56, 189, 248, 0.22);
                border-radius: 14px;
                padding: 1rem 1rem 0.9rem 1rem;
                box-shadow: 0 10px 28px rgba(2, 6, 23, 0.5);
            }
            .metric-label {
                font-size: 0.78rem;
                text-transform: uppercase;
                letter-spacing: 0.08em;
                color: #94a3b8;
                margin-bottom: 0.35rem;
            }
            .metric-value {
                font-size: 1.45rem;
                font-weight: 700;
                color: #22d3ee;
                font-family: 'JetBrains Mono', ui-monospace, monospace;
            }
            .metric-hint {
                font-size: 0.78rem;
                color: #64748b;
                margin-top: 0.35rem;
            }

            .stTabs [data-baseweb="tab-list"] {
                gap: 8px;
                background-color: rgba(15, 23, 42, 0.55);
                border-radius: 12px;
                padding: 6px;
                border: 1px solid rgba(148, 163, 184, 0.15);
            }
            .stTabs [data-baseweb="tab"] {
                border-radius: 10px;
                color: #94a3b8;
                font-weight: 600;
            }
            .stTabs [aria-selected="true"] {
                background: linear-gradient(135deg, rgba(14, 165, 233, 0.35), rgba(6, 182, 212, 0.15)) !important;
                color: #f1f5f9 !important;
                border: 1px solid rgba(56, 189, 248, 0.35) !important;
            }

            div[data-testid="stExpander"] {
                background: rgba(15, 23, 42, 0.55);
                border: 1px solid rgba(148, 163, 184, 0.15);
                border-radius: 12px;
            }

            .stButton > button {
                background: linear-gradient(135deg, #0284c7 0%, #06b6d4 100%);
                color: #f8fafc;
                border: none;
                border-radius: 12px;
                font-weight: 700;
                padding: 0.65rem 1rem;
                width: 100%;
                box-shadow: 0 10px 25px rgba(8, 145, 178, 0.35);
            }
            .stButton > button:hover {
                filter: brightness(1.06);
                box-shadow: 0 12px 30px rgba(8, 145, 178, 0.45);
            }

            .mono-inline {
                font-family: 'JetBrains Mono', ui-monospace, monospace;
                background: rgba(30, 41, 59, 0.9);
                padding: 0.1rem 0.35rem;
                border-radius: 6px;
                border: 1px solid rgba(148, 163, 184, 0.2);
                color: #e2e8f0;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def metric_cards_html(x_opt: float, y_opt: float, cost: float, iters: int) -> str:
    return f"""
    <div class="metric-grid">
        <div class="metric-card">
            <div class="metric-label">⚙ Optimal labor hours</div>
            <div class="metric-value">{x_opt:.4f}</div>
            <div class="metric-hint">Decision variable x</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">🏭 Optimal machine hours</div>
            <div class="metric-value">{y_opt:.4f}</div>
            <div class="metric-hint">Decision variable y</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">💰 Minimum cost</div>
            <div class="metric-value">{cost:.6f}</div>
            <div class="metric-hint">Objective f(x, y)</div>
        </div>
        <div class="metric-card">
            <div class="metric-label">📈 Iterations</div>
            <div class="metric-value">{iters}</div>
            <div class="metric-hint">Steepest descent steps</div>
        </div>
    </div>
    """


def init_session():
    if "result" not in st.session_state:
        st.session_state.result = None
    if "last_error" not in st.session_state:
        st.session_state.last_error = None


def main():
    st.set_page_config(
        page_title="Production Cost Optimizer",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme_css()
    init_session()

    xa, ya, fa = analytical_minimum()

    with st.sidebar:
        st.markdown("### 🎛 Control panel")
        st.caption("Configure the numerical optimizer and initial plant allocation.")
        x0 = st.number_input("Initial labor hours (x₀)", value=0.0, step=0.1, format="%.4f")
        y0 = st.number_input("Initial machine hours (y₀)", value=0.0, step=0.1, format="%.4f")
        lr = st.number_input("Learning rate η", value=0.12, step=0.01, format="%.6f", help="Must be positive.")
        max_iter = st.number_input(
            "Max iterations",
            min_value=0,
            max_value=50000,
            value=500,
            step=1,
            help="Must be greater than zero to run.",
        )
        tol = st.number_input(
            "Gradient tolerance",
            value=1e-6,
            format="%.2e",
            step=1e-6,
            help="Must be positive; typical values 1e-4 … 1e-8.",
        )
        st.divider()
        run = st.button("▶ Run optimization", type="primary")
        st.markdown(
            """
            <div class="card" style="margin-top:0.5rem;">
                <p style="margin:0;font-size:0.85rem;color:#94a3b8;">
                    Tip: if the path oscillates, reduce η or increase tolerance slightly.
                    The analytical optimum is <span class="mono-inline">(2, 3)</span> with cost <span class="mono-inline">0</span>.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        f"""
        <div class="hero-wrap">
            <div class="hero-badge">Industrial AI · Decision support</div>
            <h1 class="hero-title">AI-Powered Production Cost Optimizer</h1>
            <p class="hero-sub">
                Steepest descent on a quadratic cost model for labor (<strong>x</strong>) and machine (<strong>y</strong>) hours.
                Live KPIs, convergence analytics, and interactive surfaces — built for operations and academic review.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="card">
            <h3>📋 Problem overview</h3>
            <p style="margin:0;">
                Minimize production cost<br/>
                <span class="mono-inline">f(x, y) = x² + y² − 4x − 6y + 13</span>
                &nbsp;with&nbsp;<span class="mono-inline">x</span> = labor hours and <span class="mono-inline">y</span> = machine hours.
                Gradient-driven updates surface an explainable allocation plan for plant managers.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if run:
        v = validate_optimization_inputs(float(lr), int(max_iter), float(tol))
        if not v.ok:
            st.session_state.result = None
            st.session_state.last_error = v.error_message
            st.error(v.error_message or "Invalid inputs.")
        else:
            st.session_state.last_error = None
            with st.spinner("Running steepest descent on the cost surface…"):
                res = steepest_descent(
                    float(x0),
                    float(y0),
                    float(lr),
                    int(max_iter),
                    float(tol),
                )
            st.session_state.result = res
            if res.converged:
                st.success(res.message)
            else:
                st.warning(res.message)

    if st.session_state.last_error and not run:
        st.error(st.session_state.last_error)

    res = st.session_state.result

    tab_ov, tab_res, tab_vis, tab_bus, tab_alg = st.tabs(
        [
            "📌 Overview",
            "🧮 Optimization results",
            "📉 Visualizations",
            "💼 Business interpretation",
            "🎓 Algorithm explanation",
        ]
    )

    with tab_ov:
        col_a, col_b = st.columns((1.1, 1))
        with col_a:
            st.markdown("#### Plant digital twin (conceptual)")
            st.info(
                "This dashboard treats the cost function as a **surrogate model** of combined wage, energy, "
                "maintenance, and throughput penalties. Adjust sidebar controls to stress-test stability vs speed."
            )
        with col_b:
            st.markdown("#### Reference optimum (closed form)")
            st.markdown(
                f"The convex quadratic has a unique minimum at **x = {xa}**, **y = {ya}** with **f = {fa}** "
                "(verifiable by setting ∇f = 0)."
            )

    with tab_res:
        if res is None:
            st.warning("Run optimization from the sidebar to populate KPIs, tables, and charts.")
        else:
            st.markdown(metric_cards_html(res.x_opt, res.y_opt, res.cost_opt, res.iterations), unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                st.plotly_chart(plot_cost_convergence(res.history), use_container_width=True)
            with c2:
                st.plotly_chart(plot_gradient_norm(res.history), use_container_width=True)
            st.markdown("#### Iteration history")
            display_cols = ["iteration", "x", "y", "cost", "grad_norm"]
            hist_show = res.history[display_cols].copy()
            for c in ("x", "y", "cost"):
                hist_show[c] = hist_show[c].map(lambda v: f"{v:.8f}")
            hist_show["grad_norm"] = hist_show["grad_norm"].map(lambda v: f"{v:.6e}")
            st.dataframe(hist_show, use_container_width=True, height=360)

    with tab_vis:
        if res is None:
            st.warning("Visualizations unlock after the first successful run.")
        else:
            st.plotly_chart(plot_contour_with_path(res.history), use_container_width=True)
            st.plotly_chart(plot_surface_3d(res.history), use_container_width=True)
            st.plotly_chart(plot_before_after(float(x0), float(y0), res.x_opt, res.y_opt), use_container_width=True)
            with st.expander("Matplotlib static contour (publication / export friendly)"):
                import matplotlib.pyplot as plt

                mfig = matplotlib_cost_surface_static()
                st.pyplot(mfig, use_container_width=True)
                plt.close(mfig)

    with tab_bus:
        st.markdown(
            """
            <div class="card">
                <h3>Executive narrative</h3>
                <p>
                    Operations teams rarely see gradients — they need **hours and dollars**.
                    This module translates numerical convergence into staffing and capacity guidance.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        if res is not None:
            st.success(
                f"The factory should use approximately **{res.x_opt:.3f}** labor hours and **{res.y_opt:.3f}** "
                f"machine hours to minimize production cost (modeled cost ≈ **{res.cost_opt:.6f}** vs analytical **{fa}**)."
            )
            gap_x = abs(res.x_opt - xa)
            gap_y = abs(res.y_opt - ya)
            if gap_x > 0.05 or gap_y > 0.05:
                st.warning(
                    f"Solution differs from the analytical optimum by Δx ≈ {gap_x:.4f}, Δy ≈ {gap_y:.4f}. "
                    "Consider smaller learning rate, tighter tolerance, or more iterations."
                )
            else:
                st.info("Solution is effectively aligned with the analytical optimum — suitable for reporting.")
        else:
            st.info("Run the optimizer to generate tailored recommendations for your chosen starting point.")

    with tab_alg:
        st.markdown(
            """
            <div class="card">
                <h3>Steepest descent (gradient descent)</h3>
                <p style="margin:0;">
                    Core update: move opposite the gradient with step size <span class="mono-inline">η</span>.
                    The implementation logs every iterate for auditability and teaching.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            r"""
            **Cost function**

            $$
            f(x,y) = x^2 + y^2 - 4x - 6y + 13
            $$

            **Gradient**

            $$
            \frac{\partial f}{\partial x} = 2x - 4,\qquad
            \frac{\partial f}{\partial y} = 2y - 6
            $$

            **Steepest descent update**

            $$
            (x, y) \leftarrow (x, y) - \eta \, \nabla f(x, y)
            $$

            **Stopping rule**

            Stop when $\lVert \nabla f(x,y) \rVert_2 < \text{tolerance}$, or when the iteration budget is reached.
            """
        )
        st.markdown(
            """
            For university presentations, emphasize **convexity** (unique minimum), the trade-off controlled by **η**
            (step size versus stability), and **validation** against the closed-form optimum at **(2, 3)** with cost **0**.
            """
        )


if __name__ == "__main__":
    main()
