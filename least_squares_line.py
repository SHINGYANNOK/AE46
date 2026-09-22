"""
Part 1: Least squares on a straight line.

Given noisy measurements y_i = a*x_i + b + noise, find (a, b) minimising
the sum of squared residuals  sum_i (y_i - (a*x_i + b))^2.

Use the general N-unknown solver with rows [x_i, 1] and unknowns [a, b].
The same solver supports other linear models by changing the columns of A.
"""

import random


def solve_linear(A, b):
    """Solve A x = b (square, n x n) by Gaussian elimination with partial pivoting."""
    n = len(A)
    # ponytail: copy rows as lists; augment with b so we eliminate in one matrix
    m = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            raise ValueError("singular matrix - columns of A are dependent")
        m[col], m[piv] = m[piv], m[col]
        for r in range(col + 1, n):
            f = m[r][col] / m[col][col]
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    x = [0.0] * n
    for r in range(n - 1, -1, -1):
        x[r] = (m[r][n] - sum(m[r][c] * x[c] for c in range(r + 1, n))) / m[r][r]
    return x


def lstsq(A, b):
    """Least squares for N unknowns: min ||b - A x||^2, where A is m x N (m >= N).

    Forms the normal equations A^T A x = A^T b and solves them.
    A must have linearly independent columns.
    A is a list of rows (one row per measurement), b a list of observations.
    """
    n = len(A[0])  # number of unknowns
    ata = [[sum(row[i] * row[j] for row in A) for j in range(n)] for i in range(n)]
    atb = [sum(row[i] * bi for row, bi in zip(A, b)) for i in range(n)]
    return solve_linear(ata, atb)


def demo():
    random.seed(42)
    a_true, b_true = 2.0, 1.0
    xs = [i * 0.5 for i in range(20)]
    ys = [a_true * x + b_true + random.gauss(0, 0.3) for x in xs]

    A = [[x, 1.0] for x in xs]
    a, b = lstsq(A, ys)
    residuals = [y - (a * x + b) for x, y in zip(xs, ys)]
    ssr = sum(r * r for r in residuals)

    print(f"true:    y = {a_true}x + {b_true}")
    print(f"fit:     y = {a:.4f}x + {b:.4f}")
    print(f"sum of squared residuals: {ssr:.4f}")

    # self-check: fit must land near truth given the noise level, and the
    # fitted slope/intercept must beat every perturbation (that's the "least" part)
    assert abs(a - a_true) < 0.1 and abs(b - b_true) < 0.5
    for da, db in [(0.05, 0), (-0.05, 0), (0, 0.05), (0, -0.05)]:
        ssr_perturbed = sum(
            (y - ((a + da) * x + (b + db))) ** 2 for x, y in zip(xs, ys)
        )
        assert ssr < ssr_perturbed, "fit is not a minimum"
    print("checks passed")


if __name__ == "__main__":
    demo()
