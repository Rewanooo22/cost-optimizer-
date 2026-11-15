# Steepest Descent Cost Optimizer

A single-user Streamlit dashboard that minimizes a two-variable quadratic production-cost model with **steepest descent** (fixed-step gradient descent). In the model, **x** is labor hours and **y** is machine hours. The app plots the descent path, logs every iterate, and checks the numerical result against the known closed-form optimum. It is built as an educational and presentation demo, with a dark blue/cyan theme, KPI cards, interactive Plotly charts and tabs that explain the results.

## Problem

The factory's total cost is modelled as

$$
f(x, y) = x^2 + y^2 - 4x - 6y + 13 = (x - 2)^2 + (y - 3)^2
$$

where **x** is labor hours and **y** is machine hours. The function is a convex quadratic with a unique minimum at **x = 2, y = 3**, where **f = 0**. The app treats it as a surrogate for combined wage, energy, maintenance and throughput costs. It is not fitted to real data.

## Key features

- **Sidebar control panel** with these inputs:
  - initial point (x0, y0), default (0, 0)
  - learning rate, default 0.12
  - maximum iterations, default 500, capped at 50,000
  - gradient tolerance, default 1e-6
- **Input validation** (`utils.validate_optimization_inputs`): the learning rate, maximum iterations and tolerance must all be positive, and the app shows an error message otherwise.
- **KPI cards** for optimal x, optimal y, minimum cost and iteration count.
- **Five tabs:**
  - **Overview:** problem framing and the closed-form optimum.
  - **Optimization results:** KPI cards, a cost-convergence chart, a gradient-norm chart and the iteration history table (iteration, x, y, cost, gradient norm).
  - **Visualizations:**
    - a 2D contour plot with the descent path
    - a 3D cost surface with the path
    - a before/after cost bar chart
    - a static Matplotlib contour plot in an expander
  - **Business interpretation:** a plain-language recommendation. It warns when the result is more than 0.05 from the analytical optimum in x or y.
  - **Algorithm explanation:** the cost function, gradient, update rule and stopping rule in LaTeX.

## Algorithm

Implemented in `optimizer.py` (`steepest_descent`):

- **Gradient (analytic):** `grad f = (2x - 4, 2y - 6)`
- **Update:** `(x, y) <- (x, y) - eta * grad f(x, y)`. The step size `eta` is constant and set from the sidebar; there is no line search.
- **Stopping criteria:**
  1. Stop with `converged = True` when the Euclidean gradient norm `||grad f||_2` falls below the tolerance. This is checked at each iterate before stepping.
  2. Otherwise, stop after `max_iterations` updates and report "Maximum iterations reached without meeting the gradient tolerance."
- **Logging:** each iterate's `iteration`, `x`, `y`, `cost`, `grad_x`, `grad_y` and `grad_norm` is stored in a pandas DataFrame and used for the table and charts.
- **Reference:** `analytical_minimum()` returns `(2.0, 3.0, 0.0)` for comparison.

Because the Hessian of `f` is `2I`, each step multiplies the distance to the optimum by `(1 - 2*eta)`. With a fixed step, the iteration therefore converges for `0 < eta < 1`, overshoots and oscillates for `0.5 < eta < 1`, and diverges for `eta >= 1`. The UI accepts any positive learning rate and does not enforce this bound.

## Tech stack

| Library | Role |
|---|---|
| Python | Core language |
| Streamlit | Web UI and dashboard |
| NumPy | Gradient norm, mesh grids |
| pandas | Iteration history |
| Plotly | Interactive charts (convergence, gradient norm, contour, 3D surface, before/after) |
| Matplotlib | Static contour plot in the Visualizations tab |

## Project structure

```text
.
├── app.py              # Streamlit entry point: theme CSS, sidebar, KPI cards, tabs
├── optimizer.py        # Cost function, gradient, steepest descent, analytical minimum
├── visualizations.py   # Plotly figures and a Matplotlib contour helper
├── utils.py            # Input validation (plus unused parse_float / parse_int helpers)
├── requirements.txt    # Python dependencies
├── opt.zip             # Archived copy of an earlier version of the project files
├── __pycache__/.gitignore
├── .gitignore
├── LICENSE
└── README.md
```

## Getting started

### Prerequisites

- Python 3 with `pip`

### Installation

```bash
git clone https://github.com/Rewanooo22/steepest-descent-cost-optimizer.git
cd steepest-descent-cost-optimizer

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### Running

```bash
streamlit run app.py
```

Open the local URL that Streamlit prints (by default `http://localhost:8501`). Set the parameters in the sidebar and click **Run optimization**.

## Example interpretation

After a run, the Business interpretation tab states the result in planning terms, for example:

> "The factory should use approximately **X** labor hours and **Y** machine hours to minimize production cost."

You can check the result against the analytical optimum (2, 3).

## Limitations

- The objective is a fixed, hand-chosen quadratic. The cost model cannot be changed from the UI, and it is not derived from real production data.
- The step size is fixed. There is no line search, and the app does not check whether the chosen learning rate is in the stable range.
- The contour and 3D plots use a fixed grid (x from -0.5 to 4.5, y from -0.5 to 6.5). Paths that start far outside this window are only partly visible.
- There are no automated tests.

## Future improvements (ideas)

- Allow user-defined quadratic coefficients or constraints.
- Add exact or backtracking line search and compare it with the fixed step.
- Scale plot ranges to the descent path.
- Add unit tests for `optimizer.py`.

## License

Released under the [MIT License](LICENSE).
