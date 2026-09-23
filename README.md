# AE46 — Least Squares for GNSS Positioning

A university Final Year Project (FYP) exploring Global Navigation Satellite Systems (GNSS). The first stage builds a general linear least-squares solver and demonstrates it by fitting a straight line. The planned next stage applies that foundation to Single Point Positioning (SPP).

## Current implementation

[least_squares_line.py](least_squares_line.py) uses only Python's standard library:

| Function | Purpose |
| --- | --- |
| `solve_linear(A, b)` | Solve a square linear system using Gaussian elimination with partial pivoting. |
| `lstsq(A, b)` | Build and solve the normal equations for a linear least-squares problem with N unknowns. |
| `demo()` | Generate noisy straight-line measurements, estimate the two line parameters, and check the result. |

The separate `line_fit(xs, ys)` function was removed because the general `lstsq()` solver can do the same job. The three-unknown `demo_lstsq()` example was also removed to keep one introductory demonstration. Removing that example does **not** restrict the solver to two unknowns: the number of columns in `A` determines the number of unknowns.

The second stage is now implemented: [gnss_ls.py](gnss_ls.py) solves the **nonlinear** GNSS positioning problem iteratively (Gauss–Newton), reusing the same least-squares idea for the four unknowns `[x, y, z, b]`.

| File | Purpose |
| --- | --- |
| [least_squares_line.py](least_squares_line.py) | Linear least squares from scratch (standard library only). |
| [gnss_ls.py](gnss_ls.py) | Gauss–Newton GNSS positioning solver (uses NumPy). |

## Run the examples

From the repository directory, run:

```bash
python3 least_squares_line.py
```

The current example prints:

```text
true:    y = 2.0x + 1.0
fit:     y = 2.0109x + 0.9559
sum of squared residuals: 0.7238
checks passed
```

No third-party packages or input files are required for this part.

The GNSS solver needs NumPy. Install it into the same Python environment used to run the code:

```bash
python3 -m pip install numpy
```

```bash
python3 gnss_ls.py        # synthetic worked example with verbose iterations
```

## 1. What are we estimating?

For a straight line, the observation model is

```math
y_i = m x_i + b + \varepsilon_i,
```

where $x_i$ and $y_i$ are measured data, $m$ is the unknown slope, $b$ is the unknown intercept, and $\varepsilon_i$ represents measurement noise. The code calls the slope `a`, so its `a, b` correspond to the mathematical $m,b$ here.

With noisy measurements, one line generally cannot pass through every point. For a candidate line, the residual is the observed value minus the predicted value:

```math
r_i = y_i - (m x_i + b).
```

Least squares chooses the parameters that minimise the sum of squared residuals:

```math
J(m,b) = \sum_{i=1}^{M} r_i^2
       = \sum_{i=1}^{M} \left[y_i-(m x_i+b)\right]^2.
```

Squaring prevents positive and negative residuals from cancelling and gives larger errors a greater contribution to the objective. Here, $M$ is the number of measurements.

## 2. Write the problem as a matrix equation

Use a parameter vector $\boldsymbol{\theta}$ and an observation vector $\mathbf{d}$:

```math
\boldsymbol{\theta}=
\begin{bmatrix}m\\b\end{bmatrix},
\qquad
\mathbf{d}=
\begin{bmatrix}y_1\\y_2\\\vdots\\y_M\end{bmatrix},
\qquad
A=
\begin{bmatrix}
x_1 & 1\\ x_2 & 1\\ \vdots & \vdots\\ x_M & 1
\end{bmatrix}.
```

Each row of $A$ represents one measurement. Each column contains the coefficients multiplying one unknown. In particular,

```math
\begin{bmatrix}x_i & 1\end{bmatrix}
\begin{bmatrix}m\\b\end{bmatrix}
= m x_i+b.
```

The complete model and objective become

```math
\mathbf{d}=A\boldsymbol{\theta}+\boldsymbol{\varepsilon},
\qquad
\hat{\boldsymbol{\theta}}
=\mathop{\mathrm{arg\,min}}_{\boldsymbol{\theta}}
\left\|\mathbf{d}-A\boldsymbol{\theta}\right\|_2^2.
```

This is what the remaining demo constructs:

```python
A = [[x, 1.0] for x in xs]
a, b = lstsq(A, ys)
```

The second argument named `b` in `lstsq(A, b)` is the **whole observation vector** $\mathbf{d}$. It is not the scalar line intercept, even though the code uses the same letter in different scopes.

## 3. Derive the normal equations

Expand the objective:

```math
\begin{aligned}
J(\boldsymbol{\theta})
&=(\mathbf{d}-A\boldsymbol{\theta})^T
  (\mathbf{d}-A\boldsymbol{\theta})\\ &=\mathbf{d}^T\mathbf{d}
  -2\boldsymbol{\theta}^T A^T\mathbf{d}
  +\boldsymbol{\theta}^T A^T A\boldsymbol{\theta}.
\end{aligned}
```

Differentiate with respect to the unknown parameters and set the gradient to zero:

```math
\nabla J=-2A^T\mathbf{d}+2A^T A\boldsymbol{\theta}=\mathbf{0}.
```

Therefore,

```math
\boxed{A^T A\hat{\boldsymbol{\theta}}=A^T\mathbf{d}}.
```

These are the **normal equations**. If $A$ has linearly independent columns, $A^T A$ is positive definite, so this solution is the unique minimum.

For the two-unknown line model, the normal equations are explicitly

```math
\begin{bmatrix}
\sum_i x_i^2 & \sum_i x_i\\ \sum_i x_i & M
\end{bmatrix}
\begin{bmatrix}m\\b\end{bmatrix}
=
\begin{bmatrix}
\sum_i x_i y_i\\ \sum_i y_i
\end{bmatrix}.
```

The removed `line_fit()` function used a formula specialised to this two-by-two system. The general solver builds the same system automatically and also works when there are more columns.

## 4. What the solver does internally

For $M$ measurements and $N$ unknowns, the dimensions are

```math
A\in\mathbb{R}^{M\times N},\qquad
\mathbf{d}\in\mathbb{R}^{M},\qquad
\boldsymbol{\theta}\in\mathbb{R}^{N}.
```

`lstsq()` reads $N$ from `len(A[0])`, then computes `ata` and `atb`:

```math
(A^T A)_{jk}=\sum_{i=1}^{M} A_{ij}A_{ik},
\qquad
(A^T\mathbf{d})_j=\sum_{i=1}^{M} A_{ij}d_i.
```

It passes the resulting $N\times N$ system to `solve_linear()`. That function:

1. Copies the coefficient matrix and appends the right-hand side to form an augmented matrix.
2. Selects the largest available absolute pivot in each column and swaps rows as needed. This is partial pivoting.
3. Eliminates entries below each pivot, leaving an upper-triangular system.
4. Uses back substitution to obtain the unknowns.

For an upper-triangular system $U\boldsymbol{\theta}=\mathbf{v}$, back substitution evaluates the rows from bottom to top:

```math
\theta_j=
\frac{v_j-\sum_{k=j+1}^{N}U_{jk}\theta_k}{U_{jj}}.
```

The code solves the equations directly; it does not explicitly calculate a matrix inverse.

## 5. Two, three, four, or N unknowns

The same solver handles different linear models by changing the columns in $A$:

| Model | Unknown vector | One row of A |
| --- | --- | --- |
| Straight line $y_i=m x_i+b$ | $[m,b]^T$ | $[x_i,1]$ |
| Quadratic $y_i=p x_i^2+q x_i+r$ | $[p,q,r]^T$ | $[x_i^2,x_i,1]$ |
| Cubic $y_i=u x_i^3+v x_i^2+w x_i+t$ | $[u,v,w,t]^T$ | $[x_i^3,x_i^2,x_i,1]$ |

A quadratic or cubic curve is still **linear in its unknown parameters**: the parameters multiply known coefficients and are not squared or multiplied together. That is the meaning of “linear” required by this solver.

For example, a three-unknown fit can still be requested without a dedicated demo function:

```python
from least_squares_line import lstsq

xs = [-2, -1, 0, 1, 2]
ys = [3, 2, 3, 6, 11]  # y = x^2 + 2x + 3
A = [[x * x, x, 1.0] for x in xs]
p, q, r = lstsq(A, ys)  # approximately 1, 2, 3
```

A unique solution requires

```math
M\geq N,\qquad \mathrm{rank}(A)=N.
```

Enough measurements are necessary, but they must also provide independent information. For example, repeating the same $x_i$ for every point cannot determine both slope and intercept.

With $M=N$ and full rank, the square system can fit the observations exactly. With $M\gt N$, least squares finds the best fit to the extra measurements; an exact fit is only possible when those observations are consistent with the model.

## 6. What happens in the current demo?

The demo uses `random.seed(42)` and generates 20 inputs:

```math
x_i=0.5i,\qquad i=0,1,\ldots,19.
```

It creates observations from a known line with Gaussian noise:

```math
y_i=2x_i+1+\varepsilon_i,\qquad
\varepsilon_i\sim\mathcal{N}(0,0.3^2).
```

The noise standard deviation is $0.3$. The known slope and intercept are used to generate and check the data; the solver receives only $A$ and the noisy observations.

After solving, the fitted line is approximately

```math
\hat y=2.0109x+0.9559,
\qquad
\sum_i(y_i-\hat y_i)^2\approx0.7238.
```

The estimates are close to $2$ and $1$, but not identical, because this particular sample contains noise. The residual sum is not zero because the points do not all lie on one line.

The built-in assertions check that:

- The fitted slope is within $0.1$ of the true slope and the intercept is within $0.5$ of the true intercept.
- Changing either fitted parameter by $+0.05$ or $-0.05$, while holding the other fixed, increases the residual sum of squares.

These are small regression checks for this example. The normal-equation derivation establishes the minimum under the full-rank assumption; checking four perturbations alone is not a general proof.

## 7. Connection to GNSS: four unknowns

The GNSS stage estimates receiver position `(x, y, z)` and receiver clock bias. Clock bias can be represented as a distance b in metres or an offset delta t in seconds, related by $b=c\delta t$; these are two representations of the same fourth unknown.

See [GNSS least squares: step-by-step derivation](Gnss_ls.md) for the pseudorange model, Taylor expansion, partial derivatives, geometry matrix, correction solve, iteration, and clock-unit comparison.

## 7b. The implemented positioning solver ([gnss_ls.py](gnss_ls.py))

**Input.** `SAT` is an N×4 matrix (N ≥ 4); row *i* is `[X_i, Y_i, Z_i, P_i]`: satellite ECEF coordinates in metres and the measured pseudorange in metres. Inputs are assumed already corrected for satellite clock error, atmosphere and Earth rotation — this is a simplified educational solver, not a raw-observation processor.

**Unknowns.** `q = [x, y, z, b]`: receiver ECEF position and receiver clock bias in metres (`delta_t = b / 299792458` gives the bias in seconds).

**Algorithm (Gauss–Newton).** At the current estimate `q`:

1. Geometric ranges: `rho_i = sqrt((x-X_i)² + (y-Y_i)² + (z-Z_i)²)`
2. Predicted pseudoranges: `P_pred_i = rho_i + b`
3. Residuals: `v_i = P_i - P_pred_i` (measured minus predicted)
4. Jacobian row *i*: `[(x-X_i)/rho_i, (y-Y_i)/rho_i, (z-Z_i)/rho_i, 1]` — the partial derivatives of the measurement model: the position part is the unit vector from satellite to receiver (the negative of the receiver-to-satellite line of sight), and the bias column is 1 because the bias enters the model directly.
5. Solve `min ||v - H dq||²` for the **correction** `dq` (its normal equations are `HᵀH dq = Hᵀv`; the code uses `numpy.linalg.lstsq` for stability and never forms `HᵀH`).
6. Update all four parameters together: `q ← q + dq`, rebuild `H` at the new `q`, and repeat — that rebuild is what makes the loop Gauss–Newton rather than a one-shot linear fit.

Key distinctions the code makes explicit:

- `q` is the current estimate; `dq` contains **corrections**, not coordinates.
- `v` is evaluated at the *current* estimate, not at the solution.
- A full Gauss–Newton step need not reduce the objective every iteration; no damping, weighting, or line search is added.
- The pseudorange residual RMSE is a **fit** quality measure; it is not the position error. Small fitted residuals do not prove an accurate receiver position — that also requires good satellite geometry.
- Convergence from a distant initial guess (default `[0, 0, 0, 0]` is the Earth's centre) is not guaranteed.

**Output.** `solve_gnss_ls` returns a `GNSSResult` with the receiver ECEF coordinates, clock bias in metres and seconds, convergence status and reason, iteration count, final residual vector and sum of squared residuals, pseudorange RMSE, and a per-iteration history (old estimate, correction, updated estimate, objective before and after). Rank-deficient geometry (`rank(H) < 4`), malformed inputs, nonfinite values and zero geometric range are reported as failures, never as success.

Invalid inputs and zero geometric range raise `ValueError`. Rank deficiency, decomposition failure, overflow, and the iteration limit return `converged=False` with a reason. The result retains the last accepted state; residuals and fit metrics are NaN if the initial model cannot be evaluated. `iterations` counts completed updates and equals the history length. Tolerances must be finite positive scalars; `max_iterations` must be a positive integer.

**Worked example.** `gnss_ls.py` runs a synthetic 5-satellite example. At the first iteration from `q0 = [0, 0, 6370000, 0]`, the hand-checkable values are `v = [9, 21, 16, 10, 10]`, `dq = [10, -5, 20, 30]`, `J_before = 978 m²`, `J_after ≈ 4.000003 m²`. The data are synthetic educational values, not a real GNSS dataset.

## 8. Current limitations and next steps

The original standard-library linear solver has these limitations (the NumPy GNSS solver validates its inputs and solves directly with `numpy.linalg.lstsq`):

- Inputs must be nonempty, rectangular, dimensionally consistent, and finite. The code does not yet validate all these conditions; mismatched observation lengths can be silently truncated by `zip()`.
- The solver requires independent columns. `solve_linear()` rejects a pivot whose absolute value is below `1e-12`; this fixed threshold is sensitive to scale and is not a full conditioning check.
- Forming normal equations can amplify numerical errors. For a full-column-rank matrix, $\kappa_2(A^T A)=\kappa_2(A)^2$. QR or SVD would be preferable for difficult numerical problems.
- All measurements currently have equal weight. Weighted least squares, GNSS data loading, and real measurement corrections are not implemented.

The simulated GNSS example described above is now implemented in [gnss_ls.py](gnss_ls.py): satellite coordinates and pseudoranges generated from a known receiver state, four unknowns estimated iteratively, and the result compared with the known state in local tests (not included in this repository).
