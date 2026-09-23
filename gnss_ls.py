"""
Educational GNSS single-point positioning by Gauss-Newton least squares.

SYNTHETIC EDUCATIONAL DATA ONLY. Inputs are assumed already corrected for
satellite clock error, atmospheric delays and Earth rotation. This solver does
NOT process raw GNSS observations or navigation messages.

Mathematics (all lengths in metres):
    unknowns    q = [x, y, z, b]       receiver ECEF + clock bias (b in metres)
    geometry    rho_i = sqrt((x-X_i)^2 + (y-Y_i)^2 + (z-Z_i)^2)
    model       P_pred_i = rho_i + b
    residual    v_i = P_i - P_pred_i   (measured minus predicted, at current q)
    Jacobian    H row i = [(x-X_i)/rho_i, (y-Y_i)/rho_i, (z-Z_i)/rho_i, 1]

Gauss-Newton linearises the nonlinear model at the current q and solves, for
the CORRECTION dq (not coordinates):

    min || v - H dq ||^2,   normal equations:  H^T H dq = H^T v.

We solve with numpy.linalg.lstsq (SVD-based) instead of forming H^T H, for
numerical stability. Then q <- q + dq and H is rebuilt at the new q; that
rebuild-then-resolve loop is what makes it Gauss-Newton rather than a
one-shot linear fit. A full Gauss-Newton step need NOT reduce the objective
every iteration; there is no damping, weighting or line search here by design.

Caveat: small fitted residuals do not prove an accurate receiver position.
They only say the pseudoranges are consistent with SOME position; accuracy
also needs good satellite geometry.
"""

import numpy as np
from dataclasses import dataclass

C = 299792458.0  # speed of light, m/s; clock bias seconds = metres / C


@dataclass
class GNSSResult:
    position: np.ndarray          # receiver ECEF [x, y, z], metres
    clock_bias_m: float           # receiver clock bias, metres
    clock_bias_s: float           # same bias as seconds = b / C
    converged: bool
    termination_reason: str
    iterations: int
    residuals: np.ndarray         # final nonlinear residuals P - (rho + b)
    sum_squared_residuals: float  # J = sum(residuals^2)
    pseudorange_rmse_m: float     # sqrt(J / N): FIT quality, NOT position error
    history: list                 # one dict per iteration, see solve_gnss_ls


def _validate_sat(SAT):
    SAT = np.asarray(SAT, dtype=float)
    if SAT.ndim != 2 or SAT.shape[1] != 4:
        raise ValueError(f"SAT must be an (N, 4) matrix, got shape {SAT.shape}")
    if SAT.shape[0] < 4:
        raise ValueError(f"need at least 4 satellites, got {SAT.shape[0]}")
    if not np.all(np.isfinite(SAT)):
        raise ValueError("SAT contains nonfinite values")
    return SAT


def solve_gnss_ls(
    SAT,
    q0=None,
    max_iterations=50,
    position_tolerance_m=1e-4,
    clock_tolerance_m=1e-4,
    verbose=False,
):
    """Estimate q = [x, y, z, b] from satellite matrix SAT (N x 4 rows
    [X_i, Y_i, Z_i, P_i]) by Gauss-Newton least squares.

    q0 defaults to [0, 0, 0, 0], an Earth-centre starting guess; convergence
    from a distant initial estimate is not guaranteed.

    Invalid inputs and zero geometric range raise ValueError. Numerical or
    rank failures return converged=False at the last accepted estimate.
    iterations counts completed updates and equals len(history). If even the
    initial residuals cannot be evaluated, residuals and fit metrics are NaN.
    """
    SAT = _validate_sat(SAT)
    n_sat = SAT.shape[0]
    if (isinstance(max_iterations, (bool, np.bool_))
            or not isinstance(max_iterations, (int, np.integer))
            or max_iterations < 1):
        raise ValueError("max_iterations must be >= 1")
    for tolerance in (position_tolerance_m, clock_tolerance_m):
        if (isinstance(tolerance, (bool, np.bool_))
                or not isinstance(tolerance, (int, float, np.integer, np.floating))
                or not np.isfinite(tolerance) or tolerance <= 0):
            raise ValueError("tolerances must be finite positive numbers")
    if q0 is None:
        q = np.zeros(4)
    else:
        q = np.asarray(q0, dtype=float).copy()
        if q.shape != (4,) or not np.all(np.isfinite(q)):
            raise ValueError("q0 must be a finite sequence of 4 numbers")

    sat_xyz = SAT[:, :3]
    P = SAT[:, 3]
    history = []

    @np.errstate(over="raise", invalid="raise", divide="raise")
    def ranges_bias(qv):
        """rho, v, J at estimate qv (steps 2-4 and 8 of the algorithm)."""
        rho = np.linalg.norm(sat_xyz - qv[:3], axis=1)
        if np.any(rho <= 0.0):
            raise ValueError("zero geometric range: receiver estimate coincides with a satellite")
        v = P - (rho + qv[3])
        return rho, v, float(v @ v)

    reason = None
    for iteration in range(1, max_iterations + 1):
        try:
            rho, v, j_before = ranges_bias(q)
        except FloatingPointError:
            reason = "nonfinite calculation at current estimate"
            break

        # H: derivative of P_pred_i wrt [x, y, z] is the unit line-of-sight
        # vector from satellite to receiver (q_xyz - sat_xyz) / rho_i; wrt b it is 1 (steps 5).
        H = np.column_stack([(q[:3] - sat_xyz) / rho[:, None], np.ones(n_sat)])

        # Solve directly and use the rank reported by this same decomposition.
        try:
            dq, _, rank, singular_values = np.linalg.lstsq(H, v, rcond=None)
        except np.linalg.LinAlgError:
            reason = "least-squares decomposition failed"
            break
        if rank < 4:
            reason = (f"rank deficient: rank(H)={rank} < 4, geometry does not "
                      "allow a locally identifiable solution")
            break

        if not (np.all(np.isfinite(dq)) and np.isfinite(j_before)):
            reason = "nonfinite values encountered while solving"
            break

        try:
            with np.errstate(over="raise", invalid="raise"):
                q_new = q + dq
            _, _, j_after = ranges_bias(q_new)  # actual nonlinear objective
        except FloatingPointError:
            reason = "nonfinite calculation after update"
            break
        if not np.all(np.isfinite(q_new)) or not np.isfinite(j_after):
            reason = "nonfinite values encountered after update"
            break

        history.append({
            "q_old": q.copy(), "dq": dq.copy(), "q_new": q_new.copy(),
            "objective_before": j_before, "objective_after": j_after,
        })
        if verbose:
            print(f"iter {iteration:2d}: |dq_xyz|={np.linalg.norm(dq[:3]):12.6f} m, "
                  f"db={dq[3]:10.6f} m, J: {j_before:.6f} -> {j_after:.6f}")

        q = q_new
        if np.linalg.norm(dq[:3]) < position_tolerance_m and abs(dq[3]) < clock_tolerance_m:
            reason = "converged: corrections below tolerances"
            break
    else:
        reason = f"maximum iterations ({max_iterations}) reached without convergence"

    try:
        _, v_final, j_final = ranges_bias(q)
    except FloatingPointError:
        v_final = np.full(n_sat, np.nan)
        j_final = float("nan")
    return GNSSResult(
        position=q[:3].copy(),
        clock_bias_m=q[3],
        clock_bias_s=q[3] / C,
        converged=reason.startswith("converged"),
        termination_reason=reason,
        iterations=len(history),
        residuals=v_final,
        sum_squared_residuals=j_final,
        pseudorange_rmse_m=float(np.sqrt(j_final / n_sat)),
        history=history,
    )


# Worked example: synthetic educational data, NOT a real GNSS dataset.
SAT_WORKED = np.array([
    [ 12000000,         0, 22370000, 20000009],
    [-12000000,         0, 22370000, 20000021],
    [        0,  12000000, 22370000, 20000016],
    [        0, -12000000, 22370000, 20000010],
    [        0,         0, 26370000, 20000010],
], dtype=float)


def worked_example():
    """Run the synthetic worked example with verbose output and check it."""
    result = solve_gnss_ls(SAT_WORKED, q0=[0, 0, 6370000, 0], verbose=True)
    print(f"\nposition:    {result.position} m")
    print(f"clock bias:  {result.clock_bias_m:.6f} m "
          f"({result.clock_bias_s * 1e9:.3f} ns)")
    print(f"converged:   {result.converged} ({result.termination_reason}) "
          f"in {result.iterations} iterations")
    print(f"residuals:   {result.residuals}")
    print(f"J:           {result.sum_squared_residuals:.6f} m^2, "
          f"pseudorange RMSE: {result.pseudorange_rmse_m:.6f} m")

    # first-iteration values from the hand calculation
    h0 = result.history[0]
    assert np.allclose(h0["dq"], [10, -5, 20, 30], atol=1e-6), h0["dq"]
    assert abs(h0["objective_before"] - 978.0) < 1e-6
    assert abs(h0["objective_after"] - 4.000003) < 1e-5
    print("worked-example first-iteration checks passed")


if __name__ == "__main__":
    worked_example()
