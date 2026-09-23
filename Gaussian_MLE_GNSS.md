# Gaussian Maximum Likelihood and Least-Squares GNSS Positioning

This chapter gives the statistical interpretation of the positioning method in
[`gnss_ls.py`](gnss_ls.py). The physical GNSS model explains how receiver
position produces pseudoranges; the probability model explains how measurement
errors should be penalised. Under a Gaussian error model, maximum-likelihood
estimation becomes least squares.

The central result is

```math
\boxed{
\text{independent Gaussian pseudorange errors}
\quad\Longrightarrow\quad
\text{maximum likelihood} = \text{weighted least squares}
}
```

Ordinary least squares is the special case in which every pseudorange has the
same variance.

## 1. Deterministic and stochastic parts of the model

For satellite $i$, let its known Earth-centred, Earth-fixed (ECEF) position be
$(X_i,Y_i,Z_i)$. The unknown receiver state is

```math
\boldsymbol{\theta}
=
\begin{bmatrix}x&y&z&b\end{bmatrix}^{T},
```

where $(x,y,z)$ is the receiver ECEF position and $b=c\,\delta t$ is receiver
clock bias expressed as a distance in metres.

The geometric distance is

```math
\rho_i(x,y,z)
=
\sqrt{(x-X_i)^2+(y-Y_i)^2+(z-Z_i)^2}.
```

The predicted pseudorange is

```math
h_i(\boldsymbol{\theta})=\rho_i(x,y,z)+b.
```

The observation equation separates this deterministic prediction from random
measurement error:

```math
\boxed{P_i=h_i(\boldsymbol{\theta})+\varepsilon_i}.
```

For a proposed state, the residual is

```math
r_i(\boldsymbol{\theta})
=P_i-h_i(\boldsymbol{\theta}).
```

At the true state, the residual is the measurement error described by the
statistical model. A small residual does not prove that the state is correct;
it means only that the observation is well fitted by that state.

## 2. What a Gaussian error assumption means

First assume

```math
\varepsilon_i\sim\mathcal N(0,\sigma_i^2).
```

This assumption states that the error has zero mean and variance
$\sigma_i^2$. Its density is

```math
p(\varepsilon_i)
=
\frac{1}{\sqrt{2\pi\sigma_i^2}}
\exp\left(-\frac{\varepsilon_i^2}{2\sigma_i^2}\right).
```

The density is largest at zero. The squared error in the exponent means that
positive and negative errors of equal magnitude are equally likely, while
large errors become exponentially less likely.

Because $\varepsilon_i=P_i-h_i(\boldsymbol{\theta})$, the conditional density
of one pseudorange is

```math
p(P_i\mid\boldsymbol{\theta})
=
\frac{1}{\sqrt{2\pi\sigma_i^2}}
\exp\left[
-\frac{\left(P_i-h_i(\boldsymbol{\theta})\right)^2}{2\sigma_i^2}
\right].
```

This expression answers the question: if a candidate receiver state were
correct, how compatible would the measured pseudorange be with it?

## 3. Likelihood is a function of the unknown state

Once the measurements are observed, their numerical values are fixed. The
likelihood treats the candidate state as the variable:

```math
L(\boldsymbol{\theta})=p(\mathbf P\mid\boldsymbol{\theta}).
```

Maximum-likelihood estimation chooses

```math
\hat{\boldsymbol{\theta}}_{\mathrm{ML}}
=
\mathop{\mathrm{arg\,max}}_{\boldsymbol{\theta}}
L(\boldsymbol{\theta}).
```

This is not a probability distribution over receiver position. MLE does not
assign a prior probability to candidate states; that would require a Bayesian
model. MLE selects the state under which the observed data are most likely.

## 4. Independent measurements produce a joint likelihood

If the $M$ satellite errors are independent, the joint density is the product
of their individual densities:

```math
L(\boldsymbol{\theta})
=
\prod_{i=1}^{M}p(P_i\mid\boldsymbol{\theta}).
```

Substitution gives

```math
L(\boldsymbol{\theta})
=
\prod_{i=1}^{M}
\frac{1}{\sqrt{2\pi\sigma_i^2}}
\exp\left[-\frac{r_i(\boldsymbol{\theta})^2}{2\sigma_i^2}\right].
```

Multiplying many small densities is inconvenient and can underflow
numerically, so optimisation normally uses the log-likelihood. The logarithm
is strictly increasing and therefore does not change the maximiser:

```math
\mathop{\mathrm{arg\,max}}L
=
\mathop{\mathrm{arg\,max}}\log L.
```

The log-likelihood is

```math
\log L(\boldsymbol{\theta})
=
-\frac12\sum_{i=1}^{M}\log(2\pi\sigma_i^2)
-\frac12\sum_{i=1}^{M}
\frac{r_i(\boldsymbol{\theta})^2}{\sigma_i^2}.
```

If the standard deviations are known and do not depend on the receiver state,
the first sum is constant with respect to $\boldsymbol{\theta}$. Maximising the
remaining negative term is the same as minimising

```math
\boxed{
J_W(\boldsymbol{\theta})
=
\sum_{i=1}^{M}
\frac{r_i(\boldsymbol{\theta})^2}{\sigma_i^2}
}.
```

This is weighted least squares. The Gaussian exponent is the reason a squared
residual appears in the objective.

## 5. Ordinary least squares is the equal-variance special case

If every measurement has the same standard deviation $\sigma$, then

```math
J_W(\boldsymbol{\theta})
=
\frac{1}{\sigma^2}
\sum_{i=1}^{M}r_i(\boldsymbol{\theta})^2.
```

The positive constant $1/\sigma^2$ changes the numerical value of the
objective but not the state that minimises it. Therefore

```math
\boxed{
\hat{\boldsymbol{\theta}}_{\mathrm{ML}}
=
\mathop{\mathrm{arg\,min}}_{\boldsymbol{\theta}}
\sum_{i=1}^{M}r_i(\boldsymbol{\theta})^2
=
\hat{\boldsymbol{\theta}}_{\mathrm{OLS}}
}.
```

Thus the original unweighted solver has an implicit statistical model:

```math
\boldsymbol{\varepsilon}\sim\mathcal N(\mathbf 0,\sigma^2I).
```

It assumes independent, zero-mean pseudorange errors with the same variance.

If the common variance is unknown, the joint Gaussian MLE still selects the
ordinary least-squares state. For a fixed state,

```math
\hat\sigma^2
=
\frac{1}{M}\sum_{i=1}^{M}r_i^2,
```

and substituting this estimate into the likelihood leaves an objective that is
monotonic in the residual sum of squares.

## 6. Why the weight is inverse variance

The Gaussian weighted objective assigns

```math
w_i=\frac{1}{\sigma_i^2}.
```

A precise observation has small $\sigma_i$ and therefore a large weight. An
uncertain observation has large $\sigma_i$ and a small weight.

For the same five-metre residual:

```math
\frac{5^2}{1^2}=25
\qquad\text{but}\qquad
\frac{5^2}{5^2}=1.
```

The first residual is very surprising if the measurement normally varies by
only one metre. The second is unsurprising if five-metre errors are expected.
The weighting is therefore a statement about probability, not an arbitrary
numerical preference.

## 7. Vector Gaussian model and correlated errors

Collect the observations and predictions into vectors:

```math
\mathbf P
=
\begin{bmatrix}P_1&\cdots&P_M\end{bmatrix}^{T},
\qquad
\mathbf h(\boldsymbol{\theta})
=
\begin{bmatrix}h_1(\boldsymbol{\theta})&\cdots&h_M(\boldsymbol{\theta})\end{bmatrix}^{T}.
```

Let

```math
\boldsymbol{\varepsilon}\sim\mathcal N(\mathbf0,R),
```

where $R$ is the pseudorange covariance matrix. The multivariate Gaussian
density is

```math
p(\mathbf P\mid\boldsymbol{\theta})
=
\frac{1}{(2\pi)^{M/2}|R|^{1/2}}
\exp\left[
-\frac12\mathbf r(\boldsymbol{\theta})^TR^{-1}
\mathbf r(\boldsymbol{\theta})
\right].
```

For known $R$, Gaussian MLE minimises

```math
\boxed{
J_R(\boldsymbol{\theta})
=
\mathbf r(\boldsymbol{\theta})^TR^{-1}
\mathbf r(\boldsymbol{\theta})
}.
```

This is the squared Mahalanobis distance. Three important cases are:

| Covariance model | Statistical assumption | Estimator |
| --- | --- | --- |
| $R=\sigma^2I$ | Independent, equal variance | Ordinary least squares |
| $R=\mathrm{diag}(\sigma_1^2,\ldots,\sigma_M^2)$ | Independent, unequal variance | Weighted least squares |
| Full $R$ | Possibly correlated errors | Generalised least squares |

The implemented extension supports the diagonal case.

## 8. Why Gauss–Newton is required

The Gaussian assumption determines the objective, but does not make the GNSS
range model linear. Receiver coordinates remain inside square roots. At the
current estimate $\boldsymbol{\theta}^{(k)}$, use the first-order expansion

```math
\mathbf h(\boldsymbol{\theta}^{(k)}+\Delta\boldsymbol{\theta})
\approx
\mathbf h(\boldsymbol{\theta}^{(k)})
+H\Delta\boldsymbol{\theta}.
```

The current prefit residual is

```math
\mathbf v
=
\mathbf P-\mathbf h(\boldsymbol{\theta}^{(k)}).
```

Row $i$ of the Jacobian is

```math
H_i
=
\begin{bmatrix}
\dfrac{x-X_i}{\rho_i}&
\dfrac{y-Y_i}{\rho_i}&
\dfrac{z-Z_i}{\rho_i}&1
\end{bmatrix}.
```

After a correction, the residual is approximately

```math
\mathbf v-H\Delta\boldsymbol{\theta}.
```

The Gaussian maximum-likelihood correction therefore solves

```math
\hat{\Delta\boldsymbol{\theta}}
=
\mathop{\mathrm{arg\,min}}_{\Delta\boldsymbol{\theta}}
(\mathbf v-H\Delta\boldsymbol{\theta})^T
R^{-1}
(\mathbf v-H\Delta\boldsymbol{\theta}).
```

Its normal equations are

```math
\boxed{
H^TR^{-1}H\Delta\boldsymbol{\theta}
=
H^TR^{-1}\mathbf v
}.
```

The state is updated with

```math
\boldsymbol{\theta}^{(k+1)}
=
\boldsymbol{\theta}^{(k)}+\Delta\boldsymbol{\theta},
```

and the Jacobian is rebuilt. Gaussian MLE explains the objective; Gauss–Newton
is the numerical method used to minimise that nonlinear objective.

## 9. Whitening instead of forming normal equations

For independent measurements, let

```math
R
=
\mathrm{diag}(\sigma_1^2,\ldots,\sigma_M^2).
```

Define whitened rows and residuals:

```math
H_{w,i}=\frac{H_i}{\sigma_i},
\qquad
v_{w,i}=\frac{v_i}{\sigma_i}.
```

Then

```math
\|\mathbf v_w-H_w\Delta\boldsymbol{\theta}\|_2^2
=
(\mathbf v-H\Delta\boldsymbol{\theta})^TR^{-1}
(\mathbf v-H\Delta\boldsymbol{\theta}).
```

The implementation therefore solves

```python
H_weighted = H / pseudorange_std_m[:, None]
v_weighted = v / pseudorange_std_m
dq, _, rank, singular_values = np.linalg.lstsq(
    H_weighted, v_weighted, rcond=None
)
```

This avoids explicitly calculating $R^{-1}$ or $H^TR^{-1}H$. Equal standard
deviations scale every row by the same constant and therefore reproduce the
ordinary least-squares estimate.

## 10. State covariance and satellite geometry

Near a solution, the linearised Gaussian covariance of the estimated state is

```math
\boxed{
\mathrm{Cov}(\hat{\boldsymbol{\theta}})
\approx
(H^TR^{-1}H)^{-1}
}.
```

For equal variance,

```math
\mathrm{Cov}(\hat{\boldsymbol{\theta}})
\approx
\sigma^2(H^TH)^{-1}.
```

This relationship separates two contributors to uncertainty:

- $R$ describes pseudorange measurement quality.
- $H$ describes satellite geometry.

Precise measurements cannot fully overcome poor geometry, and good geometry
cannot remove large measurement uncertainty. Dilution-of-precision measures
are derived from this same geometry relationship after selecting the relevant
coordinates and scale.

The covariance returned by the educational solver is a local linearised
Gaussian approximation. It is not a guarantee of actual position error and is
only meaningful when the supplied standard deviations and model assumptions
are credible.

## 11. A scalar example before the GNSS case

Suppose three instruments measure one unknown distance $d$:

```math
y=\begin{bmatrix}9&10&14\end{bmatrix}^{T}.
```

With equal Gaussian variance, MLE minimises

```math
J(d)=(9-d)^2+(10-d)^2+(14-d)^2.
```

Setting the derivative to zero gives

```math
\hat d=\frac{9+10+14}{3}=11.
```

Now assume standard deviations $[1,1,4]$. Weighted Gaussian MLE gives

```math
\hat d
=
\frac{\sum_i y_i/\sigma_i^2}{\sum_i1/\sigma_i^2}
=
\frac{9+10+14/16}{1+1+1/16}
\approx9.64.
```

The uncertain third measurement has less influence. GNSS uses the same
statistical principle, but estimates four parameters through a nonlinear range
model rather than one scalar mean.

## 12. Interpreting the solver outputs

The solver reports both physical and normalised fit measures:

- `residuals`: final pseudorange residuals in metres.
- `sum_squared_residuals`: $\sum_i r_i^2$ in square metres.
- `pseudorange_rmse_m`: $\sqrt{\sum_i r_i^2/M}$ in metres.
- `normalized_residuals`: $r_i/\sigma_i$.
- `weighted_sum_squared_residuals`: $\sum_i(r_i/\sigma_i)^2$.
- `normalized_rmse`: $\sqrt{\sum_i(r_i/\sigma_i)^2/M}$.
- `state_covariance`: the local covariance approximation for $[x,y,z,b]$.

The physical RMSE describes pseudorange fit. The normalised objective describes
fit relative to the stated measurement uncertainties. Neither is the same as
position error unless a known reference position is available.

## 13. Limits of the maximum-likelihood interpretation

The equality between Gaussian MLE and least squares depends on assumptions:

1. The deterministic pseudorange model is adequate.
2. Errors have zero mean after corrections.
3. The Gaussian model is a reasonable approximation.
4. The standard deviations or covariance matrix are credible.
5. The assumed independence or correlation structure is correct.
6. The optimiser reaches the relevant minimum.

Real GNSS errors can include atmospheric modelling error, multipath,
non-line-of-sight reception, ephemeris error and outliers. These effects can be
biased, correlated or heavy-tailed. Ordinary or weighted least squares remains
computable, but it may no longer be the correct MLE and its covariance may be
over-optimistic. Robust losses, outlier detection, RAIM and richer stochastic
models are possible later extensions.

Gauss–Newton is also a local method. If it reaches the global minimum of the
Gaussian objective, that solution is the MLE. A poor starting point or weak
geometry can instead cause non-convergence or convergence to an unsuitable
local solution.

## 14. Similar names with different meanings

| Term | Role in this project |
| --- | --- |
| Gaussian distribution | Statistical model for pseudorange errors |
| Maximum likelihood | Principle for selecting the state most compatible with the observations |
| Least squares | Objective produced by the Gaussian likelihood |
| Gauss–Newton | Iterative algorithm for the nonlinear GNSS objective |
| Gaussian elimination | Linear-system algorithm used only in the introductory standard-library example |

Gaussian elimination is not the reason that least squares equals maximum
likelihood. The equivalence comes from the squared error in the Gaussian
probability density.

## 15. Final relationship

The complete reasoning chain is

```math
\boxed{
\begin{aligned}
&P_i=\rho_i(x,y,z)+b+\varepsilon_i\\
&\varepsilon\sim\mathcal N(0,R)\\
&\Downarrow\\
&-\log L(\boldsymbol{\theta})
=\text{constant}+\tfrac12\mathbf r^TR^{-1}\mathbf r\\
&\Downarrow\\
&\hat{\boldsymbol{\theta}}_{\mathrm{ML}}
=\mathop{\mathrm{arg\,min}}\mathbf r^TR^{-1}\mathbf r\\
&\Downarrow\\
&\text{weighted nonlinear least-squares positioning, solved by Gauss--Newton.}
\end{aligned}
}
```

With $R=\sigma^2I$, this reduces to the ordinary least-squares solver. With
different independent variances, it becomes weighted least squares. This is
the precise relationship between Gaussian maximum likelihood and
least-squares GNSS positioning.
