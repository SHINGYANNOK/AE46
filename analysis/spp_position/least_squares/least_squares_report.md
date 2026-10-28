# Ordinary least-squares results

Input: sppStruct_user.mat, measurement column **cor_pseudo**. Every satellite has equal weight. The numerical estimators read only satellite X, Y, Z and pseudorange. Time fields are exported as labels.

For each epoch, minimize

    sum_i (P_i - sqrt((X_i-x)^2+(Y_i-y)^2+(Z_i-z)^2) - b)^2.

The four unknowns are receiver ECEF x, y, z and common receiver clock bias b in metres. The existing project function solve_gnss_ls iterates Gauss-Newton: build the range Jacobian H, solve min ||v-H*dq||^2 with numpy.linalg.lstsq, update q, and repeat. Each epoch starts from [0,0,0,0].

## Combined fit assuming a stationary receiver

- Latitude: **22.313126726 degrees N**.
- Longitude: **114.191797004 degrees E**.
- WGS-84 ellipsoidal height: **34.586 m** (not height above sea level).
- ECEF XYZ (m): [-2419185.388948666, 5384993.44321855, 2406538.8714224347].
- All 196,405 observations are fitted together with one shared XYZ and 6,693 independent epoch clock offsets.
- Converged in 5 iterations; pooled pseudorange residual RMS: 11.545 m.

The combined objective is sum over epochs k and satellites i of (P_ki-rho_ki(x)-b_k)^2. At any candidate position, the exact least-squares clock offset is b_k=mean_i(P_ki-rho_ki). Subtracting each epoch mean from both residuals and XYZ Jacobian eliminates the clock offsets, leaving an SVD least-squares solve for the three shared position corrections. This is an ordinary least-squares fit of all measurements; it is not an average or median of already solved positions.

## Independent per-epoch fits

- 6,693/6,693 epochs converged, using 27-32 satellites each.
- Median of epoch ECEF solutions, converted to latitude/longitude/height: [22.313115570103378, 114.19179657769847, 31.481834900565445].
- Median pseudorange residual RMS: 9.412 m.
- 95% horizontal scatter radius about the per-epoch median: 8.801 m.
- Largest remaining final linearized correction norm: 1.29e-08 m.

The median above is a descriptive summary of independent epoch solutions, distinct from the combined stationary least-squares estimate.

## Interpretation limits

The supplied cor_pseudo was already processed, but its exact correction history is unknown. The run adds no further atmosphere or Earth-rotation correction. It uses one receiver clock per epoch and no separate constellation offsets. The combined fit assumes a stationary receiver. Fit residuals and repeatability do not establish accuracy without a surveyed reference.

## Files

- positions.csv: independent epoch positions, clock offsets, fit RMS and local ENU scatter.
- residuals.csv: both per-epoch and stationary-fit residuals for every input observation, with input-order indices.
- stationary_epoch_clocks.csv: clock offsets from the combined fit.
- first_epoch_iterations.csv: every four-parameter update for the first epoch.
- stationary_iterations.csv: shared-position solver iterations.
- summary.json: complete machine-readable summary and input hash.

## Reproduce

From the repository root, with NumPy installed:

    python analysis/spp_position/run_least_squares.py "C:\Users\User\Downloads\sppStruct_user.mat"

Use --field raw_pseudo and a separate --output directory to test the unprocessed pseudorange column. Its earlier fit was not a credible ground-receiver solution.
